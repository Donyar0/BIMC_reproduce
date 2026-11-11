"""Single fused-text -> BiMC fused-prototype flow. Adapted from FlowComposer s1_flow.py.
CLIP stays outside the optimizer; only base training images may enter fit().
"""
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
if __package__:
    from .fm import DeepFlowMatchingNet
else:
    from fm import DeepFlowMatchingNet


def unit(x):
    x = x.detach().float()
    if x.ndim != 2 or not torch.isfinite(x).all() or (x.norm(dim=-1) < 1e-8).any():
        raise ValueError('Features must be finite, nonzero [N,D] tensors')
    return F.normalize(x, dim=-1)


def ids_checked(ids):
    if not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('class_ids must be a nonempty list of unique integer IDs')
    return ids


def text_cache(cache):
    ids = ids_checked(cache['class_ids'])
    if not isinstance(cache.get('encoder_id'), str) or not cache['encoder_id']:
        raise ValueError('encoder_id must identify the frozen CLIP and preprocessing')
    text = unit(cache['text_prototypes']).cpu()
    if text.shape[0] != len(ids):
        raise ValueError('Text rows must match class_ids order')
    return ids, text


def output_to_velocity(model_output, xt, t, prediction='velocity'):
    """Convert the direct network output to the velocity supervised by FM.

    For the linear path xt=(1-t)*x0+t*x1, a predicted endpoint x1_hat
    implies v_hat=(x1_hat-xt)/(1-t). The caller keeps t strictly below 1.
    """
    if prediction == 'velocity':
        return model_output
    if prediction == 'endpoint':
        return (model_output - xt) / (1 - t)
    raise ValueError('prediction must be velocity or endpoint')


def fit(cache, *, epochs=100, batch_size=64, lr=2e-4, weight_decay=1e-4,
        blocks=24, hidden=None, target='prototype', ce_weight=0.0,
        temperature=0.07, prediction='velocity', endpoint_t_max=0.875,
        seed=0, device='cpu'):
    if cache.get('stage') != 'base' or cache.get('split') != 'train':
        raise ValueError('Training accepts only stage=base, split=train caches')
    if epochs < 1 or batch_size < 1 or blocks < 1 or lr <= 0 or weight_decay < 0 or ce_weight < 0 or temperature <= 0:
        raise ValueError('Invalid training hyperparameters')
    if target not in ('prototype', 'sample'):
        raise ValueError('target must be prototype or sample')
    if prediction not in ('velocity', 'endpoint'):
        raise ValueError('prediction must be velocity or endpoint')
    if not 0 < endpoint_t_max < 1:
        raise ValueError('endpoint_t_max must be strictly between 0 and 1')
    torch.manual_seed(seed)
    ids, text = text_cache(cache)
    fused = unit(cache['fused_prototypes']).cpu()
    if fused.shape != text.shape:
        raise ValueError('BiMC fused-prototype rows and dimensions must match text/class_ids')
    # BIMC computes the normalized image/text fusion using its own BETA.
    # Both training modes use this fused target, never the visual-only prototype.
    if target == 'prototype':
        sources, targets, y = text, fused, torch.arange(len(ids))
    else:
        images = unit(cache['image_features']).cpu()
        labels = cache['image_labels'].detach().cpu()
        if labels.dtype != torch.long or labels.ndim != 1 or len(labels) != len(images):
            raise ValueError('image_labels must be int64 [N]')
        if images.shape[1] != text.shape[1] or set(labels.tolist()) != set(ids):
            raise ValueError('Only listed base classes are allowed; each needs images')
        row = {c: i for i, c in enumerate(ids)}
        y = torch.tensor([row[c] for c in labels.tolist()])
        # Sample mode repeats each class-level fusion target by its image count.
        # It does NOT use individual image features as Flow endpoints.
        sources, targets = text[y], fused[y]
    loader = DataLoader(TensorDataset(sources, targets, y), batch_size=batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    arch = dict(in_channels=text.shape[1], model_channels=hidden or text.shape[1],
                out_channels=text.shape[1], num_res_blocks=blocks)
    model = DeepFlowMatchingNet(**arch).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    candidates = fused.to(device)
    history = []
    for epoch in range(epochs):
        total, count = 0., 0
        for x0, x1, label in loader:
            x0, x1, label = x0.to(device), x1.to(device), label.to(device)
            t = torch.rand(len(x0), 1, device=device)
            if prediction == 'endpoint':
                # Avoid the 1/(1-t) singularity and match the last 8-step
                # Euler evaluation time t=7/8 by default.
                t = t * endpoint_t_max
            xt = (1-t)*x0 + t*x1
            model_output = model(xt, t)
            pred_v = output_to_velocity(model_output, xt, t, prediction)
            # Both parameterizations are trained using the same velocity MSE.
            loss = ((pred_v - (x1-x0))**2).sum(-1).mean()
            # Optional CE classifies the implied endpoint for both modes.
            if ce_weight:
                pred_endpoint = (model_output if prediction == 'endpoint'
                                 else xt + (1-t)*pred_v)
                endpoint = F.normalize(pred_endpoint, dim=-1)
                loss = loss + ce_weight*F.cross_entropy(endpoint @ candidates.T / temperature, label)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()*len(x0)
            count += len(x0)
        history.append(total/count)
        print(f'epoch={epoch+1} loss={history[-1]:.6f}', flush=True)
    return dict(format_version=4, flow_target='bimc_fused', prediction=prediction,
                state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                architecture=arch, encoder_id=cache['encoder_id'], base_class_ids=ids,
                training=dict(epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                              target=target, target_space='bimc_fused',
                              ce_weight=ce_weight, temperature=temperature,
                              prediction=prediction, endpoint_t_max=endpoint_t_max, seed=seed),
                loss_history=history)


@torch.no_grad()
def generate(checkpoint, cache, steps=8, device='cpu', existing=None):
    """Base or incremental text only. No images, labels, or optimizer needed."""
    if steps < 1:
        raise ValueError('steps must be positive')
    if cache['encoder_id'] != checkpoint['encoder_id']:
        raise ValueError('Encoder mismatch')
    ids, text = text_cache(cache)
    model = DeepFlowMatchingNet(**checkpoint['architecture']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval().requires_grad_(False)
    if text.shape[1] != model.in_channels:
        raise ValueError('Feature dimension mismatch')
    prediction = checkpoint.get('prediction', checkpoint.get('training', {}).get('prediction', 'velocity'))
    if prediction not in ('velocity', 'endpoint'):
        raise ValueError('Unsupported prediction mode in checkpoint')
    x = text.to(device)
    for k in range(steps):
        t = torch.full((len(x),1), k/steps, device=device)
        model_output = model(x, t)
        velocity = output_to_velocity(model_output, x, t, prediction)
        x = x + velocity/steps
    result = dict(class_ids=ids, prototypes=unit(x).cpu(), encoder_id=cache['encoder_id'],
                  prediction=prediction, steps=steps)
    if existing is not None:
        old_ids = ids_checked(existing['class_ids'])
        old = unit(existing['prototypes'])
        if existing['encoder_id'] != result['encoder_id'] or old.shape != (len(old_ids), x.shape[1]):
            raise ValueError('Incompatible existing bank')
        if set(old_ids) & set(ids):
            raise ValueError('Refusing to overwrite existing class prototypes')
        if existing.get('prediction', prediction) != prediction or existing.get('steps', steps) != steps:
            raise ValueError('Cannot append prototypes generated with different prediction/steps')
        result['class_ids'] = old_ids + ids
        # Preserve old prototypes exactly, without regenerating or renormalizing.
        result['prototypes'] = torch.cat([existing['prototypes'].cpu(), result['prototypes']])
    return result


@torch.no_grad()
def classify(image_features, bank):
    images = unit(image_features)
    prototypes = unit(bank['prototypes']).to(images.device)
    ids = ids_checked(bank['class_ids'])
    if len(prototypes) != len(ids) or images.shape[1] != prototypes.shape[1]:
        raise ValueError('Incompatible bank dimensions')
    scores = images @ prototypes.T
    return torch.tensor(ids, device=images.device)[scores.argmax(-1)], scores


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train', help='Base training only')
    train.add_argument('--cache', required=True)
    train.add_argument('--out', required=True)
    for name, default, typ in [('epochs',100,int),('batch-size',64,int),('lr',2e-4,float),
                             ('weight-decay',1e-4,float),('blocks',24,int),('hidden',None,int),
                             ('ce-weight',0.,float),('temperature',0.07,float),('seed',0,int)]:
        train.add_argument('--'+name, default=default, type=typ)
    train.add_argument('--target', choices=['prototype','sample'], default='prototype')
    train.add_argument('--prediction', choices=['velocity','endpoint'], default='velocity')
    train.add_argument('--endpoint-t-max', type=float, default=0.875)
    gen = sub.add_parser('generate', help='Frozen inference, including incremental sessions')
    gen.add_argument('--checkpoint', required=True)
    gen.add_argument('--cache', required=True)
    gen.add_argument('--out', required=True)
    gen.add_argument('--existing-bank')
    gen.add_argument('--steps', type=int, default=8)
    evaluate = sub.add_parser('classify')
    evaluate.add_argument('--cache', required=True, help='encoder_id and image_features; image_labels optional')
    evaluate.add_argument('--bank', required=True)
    evaluate.add_argument('--out', required=True)
    for parser in (train,gen,evaluate):
        parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = vars(p.parse_args())
    command, out = args.pop('command'), args.pop('out')
    cache = load(args.pop('cache'))
    if command == 'train':
        result = fit(cache, **args)
    elif command == 'generate':
        checkpoint = load(args.pop('checkpoint'))
        existing_path = args.pop('existing_bank')
        result = generate(checkpoint, cache, existing=load(existing_path) if existing_path else None, **args)
    else:
        bank = load(args['bank'])
        if cache['encoder_id'] != bank['encoder_id']:
            raise ValueError('Encoder mismatch')
        pred, scores = classify(cache['image_features'].to(args['device']), bank)
        result = dict(predictions=pred.cpu(), scores=scores.cpu(), class_ids=bank['class_ids'])
        if 'image_labels' in cache:
            labels = cache['image_labels'].cpu()
            if labels.shape != pred.shape or not set(labels.tolist()).issubset(set(bank['class_ids'])):
                raise ValueError('Evaluation labels must belong to bank classes')
            result['accuracy'] = (pred.cpu()==labels).float().mean().item()
            print('accuracy:', result['accuracy'])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out)


if __name__ == '__main__':
    main()


