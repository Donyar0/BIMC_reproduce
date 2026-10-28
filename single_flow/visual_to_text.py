"""Single visual -> text flow: base-only training and frozen image inference.
Uses the unchanged FlowComposer network in fm.py and BIMC text prototypes.
"""
import argparse
import math
import pickle
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from .fm import DeepFlowMatchingNet
from .s1_flow import unit, ids_checked, text_cache, load


def fit(cache, *, epochs=100, batch_size=64, lr=2e-4, weight_decay=1e-4,
        blocks=24, hidden=None, source='sample', ce_weight=0.1,
        temperature=0.07, seed=0, device='cpu'):
    if cache.get('stage') != 'base' or cache.get('split') != 'train':
        raise ValueError('Training accepts only stage=base, split=train caches')
    if epochs < 1 or batch_size < 1 or blocks < 1 or lr <= 0 or weight_decay < 0 or ce_weight < 0 or temperature <= 0:
        raise ValueError('Invalid training hyperparameters')
    if source not in ('prototype', 'sample'):
        raise ValueError('source must be prototype or sample')
    torch.manual_seed(seed)
    ids, text = text_cache(cache)
    if source == 'prototype':
        visual = unit(cache['visual_prototypes']).cpu()
        if visual.shape != text.shape:
            raise ValueError('Visual prototype rows/dimensions must match text')
        sources, targets, y = visual, text, torch.arange(len(ids))
    else:
        images = unit(cache['image_features']).cpu()
        labels = cache['image_labels'].detach().cpu()
        if labels.dtype != torch.long or labels.ndim != 1 or len(labels) != len(images):
            raise ValueError('image_labels must be int64 [N]')
        if images.shape[1] != text.shape[1] or set(labels.tolist()) != set(ids):
            raise ValueError('Only listed base classes are allowed; each needs images')
        row = {c: i for i, c in enumerate(ids)}
        y = torch.tensor([row[c] for c in labels.tolist()])
        sources, targets = images, text[y]
    loader = DataLoader(TensorDataset(sources, targets, y), batch_size=batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    arch = dict(in_channels=text.shape[1], model_channels=hidden or text.shape[1],
                out_channels=text.shape[1], num_res_blocks=blocks)
    model = DeepFlowMatchingNet(**arch).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    candidates = text.to(device)
    history = []
    for epoch in range(epochs):
        total, count = 0., 0
        for x0, x1, label in loader:
            x0, x1, label = x0.to(device), x1.to(device), label.to(device)
            t = torch.rand(len(x0), 1, device=device)
            xt = (1-t)*x0 + t*x1
            pred_v = model(xt, t)
            loss = ((pred_v - (x1-x0))**2).sum(-1).mean()
            # Compare the estimated endpoint to all base-class text prototypes.
            if ce_weight:
                endpoint = F.normalize(xt + (1-t)*pred_v, dim=-1)
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
    return dict(format_version=1, direction="visual_to_text", state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                architecture=arch, encoder_id=cache['encoder_id'], base_class_ids=ids,
                training=dict(epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                              source=source, ce_weight=ce_weight, temperature=temperature, seed=seed),
                loss_history=history)



def load_flow(checkpoint, device='cpu'):
    if checkpoint.get('direction') != 'visual_to_text':
        raise ValueError('Expected visual_to_text checkpoint; retrain for this direction')
    model = DeepFlowMatchingNet(**checkpoint['architecture']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval().requires_grad_(False)
    return model


def load_checkpoint(path, trusted=False):
    """Use unrestricted pickle only when explicitly trusting a local checkpoint."""
    try:
        return torch.load(path, map_location='cpu', weights_only=not trusted)
    except pickle.UnpicklingError as exc:
        raise ValueError(
            'Checkpoint could not be read with weights_only=True. For your own '
            'trusted checkpoint, use --flow_trusted_checkpoint (FLOW.TRUST_CHECKPOINT). '
            'Do not enable this for an untrusted file.'
        ) from exc


def validate_alphas(alphas):
    if not alphas:
        raise ValueError('At least one Flow alpha is required')
    result = []
    for alpha in alphas:
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise ValueError('Flow alpha must be a finite number in [0, 1]')
        alpha = float(alpha)
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError('Flow alpha must be a finite number in [0, 1]')
        if alpha in result:
            raise ValueError('Duplicate Flow alphas are not allowed')
        result.append(alpha)
    return result


@torch.no_grad()
def build_text_bank(checkpoint, cache, existing=None):
    """Append text candidates; no image features, Flow evaluation or training."""
    if checkpoint.get('direction') != 'visual_to_text':
        raise ValueError('Expected visual_to_text checkpoint')
    if cache['encoder_id'] != checkpoint['encoder_id']:
        raise ValueError('Encoder mismatch')
    ids, text = text_cache(cache)
    if text.shape[1] != checkpoint['architecture']['in_channels']:
        raise ValueError('Feature dimension mismatch')
    result = dict(kind='text_candidates', class_ids=list(ids), prototypes=text,
                  encoder_id=cache['encoder_id'])
    if existing is not None:
        old_ids = ids_checked(existing['class_ids'])
        old = unit(existing['prototypes'])
        if (existing.get('kind') != 'text_candidates' or
                existing['encoder_id'] != result['encoder_id'] or
                old.shape != (len(old_ids), text.shape[1])):
            raise ValueError('Incompatible text bank')
        if set(old_ids) & set(ids):
            raise ValueError('Refusing to overwrite existing text candidates')
        result['class_ids'] = old_ids + ids
        result['prototypes'] = torch.cat([existing['prototypes'].cpu(), text])
    return result


@torch.no_grad()
def transform_images(model, image_features, steps=1):
    """Euler integration from t=0 to 1; normalize only the input and endpoint."""
    if type(steps) is not int or steps < 1:
        raise ValueError('steps must be a positive integer')
    model.eval().requires_grad_(False)
    x = unit(image_features).to(next(model.parameters()).device)
    if x.shape[1] != model.in_channels:
        raise ValueError('Feature dimension mismatch')
    for k in range(steps):
        t = torch.full((len(x), 1), k / steps, device=x.device)
        x = x + model(x, t) / steps
    return unit(x)


@torch.no_grad()
def classify_alphas(model, image_features, bank, steps=1, alphas=(1.0,)):
    """Share one full Flow endpoint across normalized residual mixtures.

    alpha=0 is CLIP with the same text bank; alpha=1 is the original Flow.
    Alpha does not rescale Euler velocities or change the integration horizon.
    """
    alphas = validate_alphas(alphas)
    if bank.get('kind') != 'text_candidates':
        raise ValueError('Expected text candidate bank, not pseudo-visual prototypes')
    ids = ids_checked(bank['class_ids'])
    text = unit(bank['prototypes'])
    if text.shape != (len(ids), model.in_channels):
        raise ValueError('Incompatible text bank dimensions')
    original = unit(image_features).to(next(model.parameters()).device)
    if original.shape[1] != model.in_channels:
        raise ValueError('Feature dimension mismatch')
    endpoint = transform_images(model, image_features, steps) if any(alphas) else None
    text = text.to(original.device)
    class_ids = torch.tensor(ids, device=original.device)
    result = {}
    for alpha in alphas:
        if alpha == 0:
            mixed = original
        elif alpha == 1:
            mixed = endpoint
        else:
            mixed = unit((1 - alpha) * original + alpha * endpoint)
        scores = mixed @ text.T
        result[alpha] = (class_ids[scores.argmax(-1)], scores)
    return result


@torch.no_grad()
def classify(model, image_features, bank, steps=1, alpha=1.0):
    return classify_alphas(model, image_features, bank, steps, [alpha])[alpha]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train')
    train.add_argument('--source', choices=['sample', 'prototype'], default='sample')
    for name, default, typ in [('epochs',100,int),('batch-size',64,int),('lr',2e-4,float),
                             ('weight-decay',1e-4,float),('blocks',24,int),('hidden',None,int),
                             ('ce-weight',0.1,float),('temperature',0.07,float),('seed',0,int)]:
        train.add_argument('--'+name, default=default, type=typ)
    bank = sub.add_parser('build-bank')
    bank.add_argument('--checkpoint', required=True)
    bank.add_argument('--existing-bank')
    evaluate = sub.add_parser('classify')
    evaluate.add_argument('--checkpoint', required=True)
    evaluate.add_argument('--bank', required=True)
    evaluate.add_argument('--steps', type=int, default=1)
    evaluate.add_argument('--alpha', type=float, default=1.0)
    evaluate.add_argument('--batch-size', type=int, default=256)
    for parser in (train, bank, evaluate):
        parser.add_argument('--cache', required=True)
        parser.add_argument('--out', required=True)
        parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = vars(p.parse_args())
    command, out = args.pop('command'), args.pop('out')
    cache = load(args.pop('cache'))
    if command == 'train':
        result = fit(cache, **args)
    else:
        checkpoint = load(args.pop('checkpoint'))
        if cache['encoder_id'] != checkpoint['encoder_id']:
            raise ValueError('Encoder mismatch')
        if command == 'build-bank':
            existing = args.pop('existing_bank')
            result = build_text_bank(checkpoint, cache, load(existing) if existing else None)
        else:
            bank = load(args['bank'])
            if bank['encoder_id'] != checkpoint['encoder_id']:
                raise ValueError('Encoder mismatch')
            if args['batch_size'] < 1 or len(cache['image_features']) == 0:
                raise ValueError('Positive batch size and nonempty images required')
            model = load_flow(checkpoint, args['device'])
            outputs = [classify(model, batch, bank, args['steps'], args['alpha'])
                       for batch in cache['image_features'].split(args['batch_size'])]
            pred = torch.cat([p.cpu() for p, _ in outputs])
            scores = torch.cat([s.cpu() for _, s in outputs])
            result = dict(predictions=pred, scores=scores, class_ids=bank['class_ids'], alpha=args['alpha'])
            if 'image_labels' in cache:
                labels = cache['image_labels'].cpu()
                if (labels.dtype != torch.long or labels.shape != pred.shape or
                        not set(labels.tolist()).issubset(set(bank['class_ids']))):
                    raise ValueError('Evaluation labels must be int64 [N] and belong to bank')
                result['accuracy'] = (pred == labels).float().mean().item()
                print('accuracy:', result['accuracy'])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out)


if __name__ == '__main__':
    main()

