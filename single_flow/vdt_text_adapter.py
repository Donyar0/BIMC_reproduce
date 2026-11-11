"""VDT LinearTextAdapter integration for text-to-visual Flow.

Uses the exact LinearTextAdapter, packing and text_ce objective from
feature/bimc-vdt-self-attention. The base trained text adapter remains frozen
for all incremental sessions; no incremental images train any network here.
"""
import hashlib
import json
import torch
import torch.nn.functional as F

from models.vdt_linear import LinearTextAdapter, pack_description_bank
from models.vdt_objective import linear_adapter_ce


@torch.no_grad()
def encode_vdt_descriptions(model, class_names, class_ids, gpt_path):
    """Concatenate class Prompt + each description *before* CLIP text encoding."""
    import models.clip.clip as clip
    ids = [int(c) for c in class_ids]
    if len(class_names) != len(ids) or not ids or ids != sorted(set(ids)):
        raise ValueError('Description class IDs must be unique, ordered and aligned')
    with open(gpt_path, 'r', encoding='utf-8') as handle:
        descriptions = json.load(handle)
    descriptions = {
        str(k).lower().replace('_', ' ').replace('-', ' '): v
        for k, v in descriptions.items()
    }
    features, labels = [], []
    for name, cid in zip(class_names, ids):
        key = str(name).lower().replace('_', ' ').replace('-', ' ')
        class_descriptions = descriptions[key]
        if not class_descriptions:
            raise ValueError('No description found for class '+key)
        prompt = model.template[0].format(key)
        sentences = [prompt+' '+str(desc) for desc in class_descriptions]
        chunks = []
        for start in range(0, len(sentences), 64):
            tokens = clip.tokenize(sentences[start:start+64], truncate=True).to(model.device)
            chunks.append(F.normalize(model.clip_model.encode_text(tokens).float(), dim=-1))
        current = torch.cat(chunks, dim=0)
        features.append(current)
        labels.append(torch.full((len(current),), cid, dtype=torch.long,
                                 device=current.device))
    return torch.cat(features), torch.cat(labels)


@torch.no_grad()
def adapted_prototypes(adapter, descriptions, labels, class_ids):
    """One L2-normalized prototype per class using a frozen LinearTextAdapter."""
    if adapter is None or adapter.training or any(p.requires_grad for p in adapter.parameters()):
        raise ValueError('Text adapter must be trained and frozen before inference')
    packed, mask, actual_ids = pack_description_bank(descriptions, labels)
    if actual_ids.tolist() != [int(c) for c in class_ids]:
        raise ValueError('Adapted text row/class ID mismatch')
    return adapter(packed, mask).detach().float()


def train_base_text_adapter(descriptions, description_labels,
                            image_features, image_labels, base_ids, cfg, seed, device):
    """Image-to-adapted-text CE on base TRAIN only; no FM loss here."""
    ids = [int(c) for c in base_ids]
    if not ids or ids != list(range(len(ids))):
        raise ValueError('Base classes must be contiguous and zero-based')
    if (int(cfg.EPOCHS) < 1 or int(cfg.BATCH_SIZE) < 1 or
            float(cfg.LR) <= 0 or float(cfg.WEIGHT_DECAY) < 0):
        raise ValueError('Invalid text adapter hyperparameters')
    desc = descriptions.detach().float().to(device)
    packed, mask, actual_ids = pack_description_bank(
        desc, description_labels.detach().long().to(device))
    if actual_ids.tolist() != ids:
        raise ValueError('Base descriptions do not cover all base classes')
    images = F.normalize(image_features.detach().float().to(device), dim=-1)
    labels = image_labels.detach().long().to(device)
    if (images.ndim != 2 or images.shape[-1] != desc.shape[-1] or
            len(images) == 0 or sorted(set(labels.tolist())) != ids):
        raise ValueError('Base image labels/features do not match text classes')
    rng_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=rng_devices), torch.enable_grad():
        torch.manual_seed(int(seed))
        adapter = LinearTextAdapter(
            embed_dim=desc.shape[-1], ratio=float(cfg.RATIO),
            dropout=float(cfg.DROPOUT)).to(device).train()
        optimizer = torch.optim.AdamW(
            adapter.parameters(), lr=float(cfg.LR),
            weight_decay=float(cfg.WEIGHT_DECAY))
        for epoch in range(int(cfg.EPOCHS)):
            order = torch.randperm(len(images), device=images.device)
            total = 0.0
            for start in range(0, len(images), int(cfg.BATCH_SIZE)):
                index = order[start:start+int(cfg.BATCH_SIZE)]
                optimizer.zero_grad(set_to_none=True)
                prototypes = adapter(packed, mask)
                loss = linear_adapter_ce(
                    images[index], labels[index], prototypes, mode='text_ce')
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite text adapter CE')
                loss.backward()
                optimizer.step()
                total += loss.detach().item() * len(index)
            print('Text Adapter base epoch={}/{} CE={:.6f}'.format(
                epoch+1, cfg.EPOCHS, total/len(images)), flush=True)
    adapter.eval().requires_grad_(False)
    return adapter


def replace_flow_cache_text(cache, adapted_text, beta):
    """Override BOTH x0 and BiMC-fusion x1; leave visual prototypes intact."""
    if not (0 <= float(beta) <= 1):
        raise ValueError('BETA must be in [0,1]')
    text = F.normalize(adapted_text.detach().float().cpu(), dim=-1)
    visual = cache['visual_prototypes'].detach().float().cpu()
    if (text.ndim != 2 or text.shape != visual.shape or
            text.shape[0] != len(cache['class_ids']) or
            not torch.isfinite(text).all() or not torch.isfinite(visual).all()):
        raise ValueError('Adapter/visual prototype dimensions or values mismatch')
    result = dict(cache)
    result['text_prototypes'] = text.clone()
    result['fused_prototypes'] = F.normalize(
        float(beta)*text+(1-float(beta))*visual, dim=-1)
    return result


def fingerprint(adapter):
    h = hashlib.sha256()
    for name, value in sorted(adapter.state_dict().items()):
        item = value.detach().cpu().contiguous()
        h.update(name.encode('utf-8'))
        h.update(str(tuple(item.shape)).encode('utf-8'))
        h.update(item.numpy().tobytes())
    return h.hexdigest()


def save_adapter(adapter, encoder_id, base_ids, path):
    """Metadata plus tensors, compatible with torch.load(weights_only=True)."""
    torch.save({
        'encoder_id': encoder_id, 'base_class_ids': [int(c) for c in base_ids],
        'architecture': {
            'embed_dim': adapter.projection.in_features,
            'ratio': adapter.ratio, 'dropout': adapter.dropout.p},
        'state_dict': {k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()},
        'sha256': fingerprint(adapter),
    }, path)
