"""Single visual -> text flow: base training and optional incremental support finetuning.
Uses the unchanged FlowComposer network in fm.py and BIMC text prototypes.
"""
import argparse
import math
import pickle
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from .fm import DeepFlowMatchingNet
from .s1_flow import unit, ids_checked, text_cache, load


def fit(cache, *, epochs=100, batch_size=64, lr=2e-4, weight_decay=1e-4,
        blocks=24, hidden=None, source='sample', seed=0, device='cpu'):
    if cache.get('stage') != 'base' or cache.get('split') != 'train':
        raise ValueError('Training accepts only stage=base, split=train caches')
    if epochs < 1 or batch_size < 1 or blocks < 1 or lr <= 0 or weight_decay < 0:
        raise ValueError('Invalid training hyperparameters')
    if source not in ('prototype', 'sample'):
        raise ValueError('source must be prototype or sample')
    torch.manual_seed(seed)
    ids, text = text_cache(cache)
    if source == 'prototype':
        visual = unit(cache['visual_prototypes']).cpu()
        if visual.shape != text.shape:
            raise ValueError('Visual prototype rows/dimensions must match text')
        sources, targets = visual, text
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
    loader = DataLoader(TensorDataset(sources, targets), batch_size=batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    arch = dict(in_channels=text.shape[1], model_channels=hidden or text.shape[1],
                out_channels=text.shape[1], num_res_blocks=blocks)
    model = DeepFlowMatchingNet(**arch).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for epoch in range(epochs):
        total, count = 0., 0
        for x0, x1 in loader:
            x0, x1 = x0.to(device), x1.to(device)
            t = torch.rand(len(x0), 1, device=device)
            xt = (1-t)*x0 + t*x1
            pred_v = model(xt, t)
            loss = ((pred_v - (x1-x0))**2).sum(-1).mean()
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
                              source=source, seed=seed),
                loss_history=history)




def finetune_incremental(checkpoint, cache, *, session_id,
                         epochs=10, batch_size=25, lr=1e-5, weight_decay=0.,
                         seed=0, device='cpu'):
    """Continue the existing visual->text Flow on current-session 5-shot support.

    FM target: correct novel-class fused text. Only FM is optimized.
    Never uses previous session images or test images. Input checkpoint is not mutated.
    """
    if cache.get('stage') != 'incremental' or cache.get('split') != 'train':
        raise ValueError('Incremental training requires stage=incremental, split=train')
    if type(session_id) is not int or session_id < 1:
        raise ValueError('session_id must be a positive integer')
    if (type(epochs) is not int or epochs < 1 or type(batch_size) is not int or
            batch_size < 1 or not (0 < lr < float('inf')) or
            not (0 <= weight_decay < float('inf'))):
        raise ValueError('Invalid incremental hyperparameters')
    if checkpoint.get('direction') != 'visual_to_text':
        raise ValueError('Expected visual_to_text checkpoint')
    if cache.get('encoder_id') != checkpoint.get('encoder_id'):
        raise ValueError('Incremental encoder mismatch')
    current_ids, current_text = text_cache(cache)
    dim = checkpoint['architecture']['in_channels']
    if current_text.shape[1] != dim:
        raise ValueError('Text feature dimensions do not match')
    if set(current_ids) & set(checkpoint['base_class_ids']):
        raise ValueError('Incremental classes must not overlap base classes')
    updates = checkpoint.get('incremental_updates', [])
    if updates:
        if updates[-1]['session_id'] + 1 != session_id:
            raise ValueError('Sessions must be trained in chronological order')
        if any(set(current_ids) & set(record['class_ids']) for record in updates):
            raise ValueError('Classes from a previously learned session cannot be retrained')
    elif session_id != 1:
        raise ValueError('First incremental update must be session 1')
    prior_ids = [cid for record in updates for cid in record['class_ids']]
    if len(set(prior_ids)) != len(prior_ids):
        raise ValueError('Previous incremental class IDs contain duplicates')
    num_seen_classes = len(checkpoint['base_class_ids']) + len(prior_ids) + len(current_ids)
    row = {cid: i for i, cid in enumerate(current_ids)}

    images = unit(cache['image_features']).cpu()
    labels = cache['image_labels'].detach().cpu()
    if labels.ndim != 1 or labels.dtype != torch.long or len(labels) != len(images):
        raise ValueError('Support image_labels must be int64 [N]')
    if images.shape[1] != dim or set(labels.tolist()) != set(current_ids):
        raise ValueError('Support labels must contain only all current novel classes')
    y = torch.tensor([row[int(cid)] for cid in labels.tolist()], dtype=torch.long)
    targets = current_text[y]
    loader = DataLoader(TensorDataset(images, targets), batch_size=batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(seed + session_id))
    model = load_flow(checkpoint, device).train().requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for epoch in range(epochs):
        total, n = 0., 0
        for x0, x1 in loader:
            x0, x1 = x0.to(device), x1.to(device)
            t = torch.rand(len(x0), 1, device=device)
            xt = (1 - t) * x0 + t * x1
            v = model(xt, t)
            fm = ((v - (x1 - x0)) ** 2).sum(-1).mean()
            loss = fm
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite incremental training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            size = len(x0)
            total += loss.item() * size
            n += size
        record = dict(loss=total / n, fm=total / n)
        history.append(record)
        print(f'inc_session={session_id} epoch={epoch+1} fm={record["fm"]:.6f}', flush=True)

    updated = dict(checkpoint)
    updated['state_dict'] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    updated['incremental_updates'] = list(updates) + [dict(
        session_id=session_id, class_ids=list(current_ids), num_samples=len(images),
        num_seen_classes=num_seen_classes, epochs=epochs, batch_size=batch_size,
        lr=lr, weight_decay=weight_decay, 
        loss_history=history)]
    updated['trained_through_session'] = session_id
    model.eval().requires_grad_(False)
    return updated


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
def classify_alphas(model, image_features, candidate_text, steps=1,
                    alphas=(1.0,), candidate_class_ids=None):
    """Full visual->text Flow inference, with candidate text computed at runtime.

    No text bank is created, loaded, or saved. alpha=1 is mandatory.
    """
    if validate_alphas(alphas) != [1.0]:
        raise ValueError('Full Flow inference only: alpha must be 1')
    if not isinstance(candidate_text, torch.Tensor) or candidate_text.ndim != 2:
        raise ValueError('candidate_text must be a [num_seen_classes, dim] tensor')
    if candidate_text.shape[0] == 0 or candidate_text.shape[1] != model.in_channels:
        raise ValueError('Incompatible candidate text dimensions')
    if candidate_class_ids is None:
        candidate_class_ids = list(range(len(candidate_text)))
    ids = ids_checked(candidate_class_ids)
    if len(ids) != len(candidate_text):
        raise ValueError('candidate_class_ids must match candidate_text rows')
    endpoint = transform_images(model, image_features, steps=steps)
    text = unit(candidate_text).to(endpoint.device)
    scores = endpoint @ text.T
    class_ids = torch.tensor(ids, dtype=torch.long, device=endpoint.device)
    return {1.0: (class_ids[scores.argmax(-1)], scores)}


@torch.no_grad()
def classify(model, image_features, candidate_text, steps=1, candidate_class_ids=None):
    return classify_alphas(model, image_features, candidate_text,
                           steps=steps, candidate_class_ids=candidate_class_ids)[1.0]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train')
    train.add_argument('--source', choices=['sample', 'prototype'], default='sample')
    for name, default, typ in [('epochs',100,int),('batch-size',64,int),('lr',2e-4,float),
                               ('weight-decay',1e-4,float),('blocks',24,int),('hidden',None,int),
                               ('seed',0,int)]:
        train.add_argument('--'+name, default=default, type=typ)
    evaluate = sub.add_parser('classify')
    evaluate.add_argument('--checkpoint', required=True)
    evaluate.add_argument('--candidate-cache', required=True,
                          help='Existing cache containing all seen class_ids/text_prototypes; no bank')
    evaluate.add_argument('--steps', type=int, default=1)
    evaluate.add_argument('--batch-size', type=int, default=256)
    for parser in (train, evaluate):
        parser.add_argument('--cache', required=True)
        parser.add_argument('--out', required=True)
        parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = vars(p.parse_args())
    command, out = args.pop('command'), args.pop('out')
    cache = load(args.pop('cache'))
    if command == 'train':
        result = fit(cache, **args)
    else:
        checkpoint = load_checkpoint(args['checkpoint'])
        candidates = load(args['candidate_cache'])
        if (cache.get('encoder_id') != checkpoint['encoder_id'] or
                candidates.get('encoder_id') != checkpoint['encoder_id']):
            raise ValueError('Encoder mismatch')
        ids, texts = text_cache(candidates)
        if args['batch_size'] < 1 or len(cache['image_features']) == 0:
            raise ValueError('Positive batch size and nonempty images required')
        model = load_flow(checkpoint, args['device'])
        outputs = [classify(model, batch, texts, steps=args['steps'],
                            candidate_class_ids=ids)
                   for batch in cache['image_features'].split(args['batch_size'])]
        pred = torch.cat([p.cpu() for p, _ in outputs])
        scores = torch.cat([s.cpu() for _, s in outputs])
        result = dict(predictions=pred, scores=scores, class_ids=ids, alpha=1.0)
        if 'image_labels' in cache:
            labels = cache['image_labels'].cpu()
            if (labels.dtype != torch.long or labels.shape != pred.shape or
                    not set(labels.tolist()).issubset(set(ids))):
                raise ValueError('Evaluation labels must be int64 [N] and belong to candidates')
            result['accuracy'] = (pred == labels).float().mean().item()
            print('accuracy:', result['accuracy'])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out)


if __name__ == '__main__':
    main()
