"""Base-only, augmented image -> original BiMC text Flow Matching."""
import torch
import torch.nn.functional as F
from .fm import DeepFlowMatchingNet
from .s1_flow import unit, text_cache

METHOD = 'bimc-original-fused-v1'
DIRECTION = 'image_to_text'


def velocity_from_output(output, x, t, prediction, time_eps):
    if prediction == 'velocity':
        return output
    if prediction == 'endpoint':
        return (output - x) / (1 - t).clamp_min(time_eps)
    raise ValueError('PREDICTION must be velocity or endpoint')


class VelocityField(torch.nn.Module):
    def __init__(self, network, prediction, time_eps):
        super().__init__()
        self.network, self.prediction, self.time_eps = network, prediction, time_eps

    def forward(self, x, t):
        # Both parameterizations use the same truncated time domain. In
        # particular reverse generation never divides by zero at t=1.
        t = t.clamp(max=1-self.time_eps)
        return velocity_from_output(self.network(x, t), x, t, self.prediction, self.time_eps)


def load_model(checkpoint, device):
    if checkpoint.get('flow_direction') != DIRECTION or checkpoint.get('text_prototype_method') != METHOD:
        raise ValueError('Incompatible Flow direction/text prototypes; retrain image-to-text Flow')
    network = DeepFlowMatchingNet(**checkpoint['architecture']).to(device)
    network.load_state_dict(checkpoint['state_dict'])
    return VelocityField(network, checkpoint['prediction'], checkpoint['time_eps'])


def fit(cache, loader, encode_images, *, epochs, lr, weight_decay, blocks, hidden,
        prediction, time_eps, seed, device):
    if cache.get('stage') != 'base' or cache.get('split') != 'train':
        raise ValueError('Flow training accepts base training images only')
    if epochs < 1 or blocks < 1 or lr <= 0 or weight_decay < 0:
        raise ValueError('Invalid Flow training parameters')
    if prediction not in ('velocity', 'endpoint') or not 0 < time_eps < 0.5:
        raise ValueError('Invalid prediction parameterization/time epsilon')
    torch.manual_seed(seed)
    ids, text = text_cache(cache)
    if ids != list(range(len(ids))):
        raise ValueError('Base class IDs must be contiguous and zero-based')
    text = text.to(device)
    dim = text.shape[1]
    arch = dict(in_channels=dim, model_channels=hidden or dim,
                out_channels=dim, num_res_blocks=blocks)
    network = DeepFlowMatchingNet(**arch).to(device).train()
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for epoch in range(epochs):
        total, count = 0., 0
        # Pixel transforms are sampled again whenever the loader is iterated.
        for batch in loader:
            with torch.no_grad():
                x0 = unit(encode_images(batch['image'].to(device))).to(device)
                labels = batch['label'].to(device=device, dtype=torch.long)
                if labels.ndim != 1 or len(labels) != len(x0) or not ((labels >= 0) & (labels < len(ids))).all():
                    raise ValueError('Only base training labels may enter Flow optimization')
                x1 = text[labels]
            t = torch.rand(len(x0), 1, device=device) * (1-time_eps)
            xt = (1-t)*x0 + t*x1
            pred_v = velocity_from_output(network(xt, t), xt, t, prediction, time_eps)
            loss = (pred_v-(x1-x0)).square().sum(-1).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite image-to-text FM loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()*len(x0)
            count += len(x0)
        if not count:
            raise ValueError('Empty base training loader')
        history.append(total/count)
        print(f'epoch={epoch+1} prediction={prediction} fm_loss={history[-1]:.6f}', flush=True)
    return dict(format_version=3, architecture=arch,
                state_dict={k:v.detach().cpu().clone() for k,v in network.state_dict().items()},
                encoder_id=cache['encoder_id'], base_class_ids=ids,
                flow_direction=DIRECTION, text_prototype_method=METHOD,
                prediction=prediction, time_eps=time_eps,
                training=dict(epochs=epochs, lr=lr, weight_decay=weight_decay, seed=seed,
                              base_only=True, online_pixel_augmentation=True), loss_history=history)


@torch.no_grad()
def transport(model, features, steps, reverse=False, solver='euler'):
    if steps < 1:
        raise ValueError('Integration steps must be positive')
    if solver not in ('euler', 'midpoint', 'heun', 'rk4'):
        raise ValueError('SOLVER must be euler, midpoint, heun or rk4')
    x = features.float().clone()
    step = (-1 if reverse else 1) / steps
    for k in range(steps):
        time = 1-k/steps if reverse else k/steps
        def field(state, at):
            t = torch.full((len(state), 1), at, device=state.device)
            return model(state, t)

        k1 = field(x, time)
        if solver == 'euler':
            update = k1
        elif solver == 'midpoint':
            update = field(x + step*k1/2, time + step/2)
        elif solver == 'heun':
            k2 = field(x + step*k1, time + step)
            update = (k1 + k2)/2
        else:
            k2 = field(x + step*k1/2, time + step/2)
            k3 = field(x + step*k2/2, time + step/2)
            k4 = field(x + step*k3, time + step)
            update = (k1 + 2*k2 + 2*k3 + k4)/6
        x = x + step*update
        if not torch.isfinite(x).all():
            raise RuntimeError('Nonfinite Flow integration')
    return F.normalize(x, dim=-1)


@torch.no_grad()
def build_bank(checkpoint, memory, *, steps, num_base, beta_base, beta_inc, device,
               previous_bank=None, solver='euler'):
    if steps < 1 or not all(0 <= b <= 1 for b in (beta_base, beta_inc)):
        raise ValueError('Invalid bank integration/fusion parameters')
    if memory['encoder_id'] != checkpoint['encoder_id']:
        raise ValueError('Encoder mismatch')
    if not 0 <= num_base <= len(memory['class_ids']):
        raise ValueError('Invalid base class count')
    if previous_bank is not None:
        if previous_bank['encoder_id'] != checkpoint['encoder_id']:
            raise ValueError('Encoder mismatch')
        if set(previous_bank['class_ids']) & set(memory['class_ids']):
            raise ValueError('New classes overlap prior bank')
    model = load_model(checkpoint, device).eval()
    # Training direction is image -> text: visual generation integrates backwards.
    generated = transport(model, memory['text'].to(device), steps,
                          reverse=True, solver=solver).cpu()
    beta = torch.full((len(generated), 1), beta_inc)
    beta[:num_base] = beta_base
    current = dict(class_ids=list(memory['class_ids']), text=memory['text'].cpu().clone(),
                   real=memory['real'].cpu().clone(), generated=generated,
                   prototypes=F.normalize((1-beta)*memory['real'].cpu()+beta*generated, dim=-1),
                   encoder_id=checkpoint['encoder_id'])
    if previous_bank is None:
        return current
    return dict(class_ids=previous_bank['class_ids']+current['class_ids'],
                **{k:torch.cat([previous_bank[k],current[k]]) for k in ('text','real','generated','prototypes')},
                encoder_id=checkpoint['encoder_id'])


@torch.no_grad()
def classify_bidirectional(model, image_features, bank, *, alpha, gamma, steps,
                           solver='euler'):
    if not 0 <= alpha <= 1 or not 0 <= gamma <= 1:
        raise ValueError('alpha and gamma must be in [0,1]')
    z = unit(image_features)
    mapped = transport(model, z, steps, solver=solver)
    semantic_query = F.normalize((1-alpha)*z+alpha*mapped, dim=-1)
    scores = gamma*(z @ bank['prototypes'].to(z.device).T) + (1-gamma)*(semantic_query @ bank['text'].to(z.device).T)
    ids = torch.tensor(bank['class_ids'], device=z.device)
    return ids[scores.argmax(-1)], scores
