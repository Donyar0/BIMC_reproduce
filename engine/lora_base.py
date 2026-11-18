"""Base-session vision-only CLIP LoRA training and merge."""
import copy
import time
from contextlib import nullcontext
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import models.clip.clip as clip
from models.clip.model import LoRAMultiheadAttention


class AlignedAnchorDataset(Dataset):
    """Return a centered image and the index of its optional horizontal flip."""

    def __init__(self, source, center_transform):
        self.source = copy.copy(source)
        self.source.transform = center_transform

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        item = self.source[index]
        view = int(torch.rand(()) < 0.5)
        if view:
            item['image'] = item['image'].flip(-1)
        item['anchor_view'] = view
        return item


def cache_base_anchor_features(teacher, dataset, data_manager, cfg, device, feature_dim):
    """Cache center and flipped teacher views for the selected Base images."""
    anchor_dataset = copy.copy(dataset)
    anchor_dataset.transform = data_manager.test_transform
    loader = DataLoader(anchor_dataset,
                        batch_size=cfg.DATALOADER.TRAIN.BATCH_SIZE_BASE,
                        shuffle=False, drop_last=False,
                        **data_manager.loader_kwargs())
    # Allocate outside inference_mode so indexed targets can be used by autograd.
    num_views = 2 if cfg.LORA.ALIGN_ANCHOR_VIEW else 1
    cache = torch.empty((len(dataset), num_views, feature_dim),
                        device=device, dtype=torch.bfloat16)
    started = time.time()
    use_cuda = torch.device(device).type == 'cuda'
    with torch.inference_mode():
        for batch in loader:
            images = batch['image'].to(device, non_blocking=True)
            indices = batch['idx'].to(device, non_blocking=True)
            inputs = (torch.cat((images, images.flip(-1)), dim=0)
                      if num_views == 2 else images)
            autocast = (torch.cuda.amp.autocast(dtype=torch.bfloat16)
                        if use_cuda else nullcontext())
            with autocast:
                features = teacher(inputs)
            features = F.normalize(features.float(), dim=-1)
            by_sample = features.reshape(num_views, len(images), feature_dim).permute(1, 0, 2)
            cache[indices] = by_sample.to(torch.bfloat16)
    if use_cuda:
        torch.cuda.synchronize(device)
    print(f'Cached {len(dataset)} Base anchor features ({num_views} views/image) '
          f'in {time.time() - started:.1f}s', flush=True)
    return cache


def encode_base_prompts(clip_model, tokens, num_classes, num_templates):
    features = F.normalize(clip_model.encode_text(tokens).float(), dim=-1)
    return F.normalize(features.reshape(num_classes, num_templates, -1).mean(1), dim=-1)


def train_base_vision_lora(model, data_manager, cfg, output):
    if cfg.TRAINER.BiMC.PREC != 'fp32':
        raise ValueError('Base LoRA training requires BiMC.PREC: fp32')
    if cfg.LORA.ANCHOR_WEIGHT < 0 or cfg.LORA.TEXT_ANCHOR_WEIGHT < 0:
        raise ValueError('LoRA anchor weights must be nonnegative')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Base LoRA bfloat16 autocast requires a CUDA GPU with bfloat16 support')
    base_ids = list(data_manager.class_index_in_task[0])
    if base_ids != list(range(len(base_ids))):
        raise ValueError('Base labels must be zero-based and contiguous')
    # Q/V updates start at zero, so the optional teacher is the original CLIP vision tower.
    teacher = (copy.deepcopy(model.clip_model.visual).eval().requires_grad_(False)
               if cfg.LORA.ANCHOR_WEIGHT > 0 else None)
    model.clip_model.requires_grad_(False)
    lora_layers = [block.attn for block in model.clip_model.visual.transformer.resblocks
                   if isinstance(block.attn, LoRAMultiheadAttention)]
    if not lora_layers:
        raise ValueError('No visual LoRA layers installed')
    parameters = []
    for layer in lora_layers:
        for parameter in (layer.q_A, layer.q_B, layer.v_A, layer.v_B):
            parameter.requires_grad_(True)
            parameters.append(parameter)
    if cfg.LORA.TEXT_ENABLED:
        text_layers = [block.attn for block in model.clip_model.transformer.resblocks
                       if isinstance(block.attn, LoRAMultiheadAttention)]
        if not text_layers:
            raise ValueError('Text LoRA was enabled but no text attention layers were installed')
        for layer in text_layers:
            for parameter in (layer.q_A, layer.q_B, layer.v_A, layer.v_B):
                parameter.requires_grad_(True)
                parameters.append(parameter)
    names = [data_manager.class_names[c] for c in base_ids]
    prompts = [template.format(name.replace('_', ' ').replace('-', ' '))
               for name in names for template in model.template]
    tokens = clip.tokenize(prompts).to(model.device)
    with torch.no_grad():
        text_anchor = encode_base_prompts(model.clip_model, tokens, len(names), len(model.template))
        scale = model.clip_model.logit_scale.exp().float().detach()
    source_dataset = data_manager.get_dataset(0, source='train', mode='train')
    dataset = (AlignedAnchorDataset(source_dataset, data_manager.test_transform)
               if cfg.LORA.ALIGN_ANCHOR_VIEW else source_dataset)
    if cfg.LORA.ALIGN_ANCHOR_VIEW:
        print('Base LoRA augmentation: center crop with random horizontal flip', flush=True)
    loader = DataLoader(dataset, batch_size=cfg.DATALOADER.TRAIN.BATCH_SIZE_BASE,
                        shuffle=True, drop_last=False,
                        **data_manager.loader_kwargs())
    anchor_cache = None
    if cfg.LORA.ANCHOR_WEIGHT > 0 and cfg.LORA.CACHE_ANCHOR:
        anchor_cache = cache_base_anchor_features(
            teacher, source_dataset, data_manager, cfg, model.device, text_anchor.shape[-1])
        del teacher
    optimizer = torch.optim.AdamW(parameters, lr=cfg.LORA.LR,
                                  weight_decay=cfg.LORA.WEIGHT_DECAY)
    total_steps = cfg.LORA.EPOCHS * len(loader)
    if total_steps < 1:
        raise ValueError('Empty Base LoRA training loader')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    model.clip_model.visual.train()
    if cfg.LORA.TEXT_ENABLED:
        model.clip_model.transformer.train()
    history = []
    step = 0
    for epoch in range(cfg.LORA.EPOCHS):
        total, count = 0., 0
        for batch in loader:
            images, labels = model.parse_batch(batch)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                image = model.extract_img_feature(images, tta=False)
                text = (encode_base_prompts(model.clip_model, tokens, len(names), len(model.template))
                        if cfg.LORA.TEXT_ENABLED else text_anchor)
                if anchor_cache is None and cfg.LORA.ANCHOR_WEIGHT > 0:
                    with torch.no_grad():
                        reference = teacher(images)
            # LoRA weights, logits, losses and optimizer updates remain float32.
            with torch.cuda.amp.autocast(enabled=False):
                image = F.normalize(image.float(), dim=-1)
                logits = scale * image @ text.T
                ce = F.cross_entropy(logits, labels)
                if cfg.LORA.ANCHOR_WEIGHT > 0:
                    if anchor_cache is not None:
                        indices = batch['idx'].to(model.device, non_blocking=True)
                        views = (batch['anchor_view'].to(model.device, non_blocking=True)
                                 if cfg.LORA.ALIGN_ANCHOR_VIEW else torch.zeros_like(indices))
                        reference = anchor_cache[indices, views].float()
                    else:
                        reference = F.normalize(reference.float(), dim=-1)
                    loss = ce + cfg.LORA.ANCHOR_WEIGHT * F.mse_loss(image, reference)
                else:
                    loss = ce
                if cfg.LORA.TEXT_ENABLED and cfg.LORA.TEXT_ANCHOR_WEIGHT > 0:
                    loss = loss + cfg.LORA.TEXT_ANCHOR_WEIGHT * F.mse_loss(text, text_anchor)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite Base LoRA loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            total += loss.item() * len(labels)
            count += len(labels)
        history.append(total / count)
        print(f'Base LoRA epoch={epoch+1}/{cfg.LORA.EPOCHS} loss={history[-1]:.6f}', flush=True)
    model.merge_vision_lora()
    torch.save({key: value.detach().cpu().clone()
                for key, value in model.clip_model.visual.state_dict().items()},
               output / 'base_lora_merged_visual.pt')
    if cfg.LORA.TEXT_ENABLED:
        torch.save({key: value.detach().cpu().clone()
                    for key, value in model.clip_model.transformer.state_dict().items()},
                   output / 'base_lora_merged_text.pt')
    print(f'Base {"Vision+Text" if cfg.LORA.TEXT_ENABLED else "Vision"} LoRA '
          f'merged after {step} updates', flush=True)
    return history
