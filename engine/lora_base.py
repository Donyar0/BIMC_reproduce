"""Base-session vision-only CLIP LoRA training and merge."""
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.clip.model import LoRAMultiheadAttention


def train_base_vision_lora(model, data_manager, cfg, output):
    if cfg.TRAINER.BiMC.PREC != 'fp32':
        raise ValueError('Base LoRA training requires BiMC.PREC: fp32')
    base_ids = list(data_manager.class_index_in_task[0])
    if base_ids != list(range(len(base_ids))):
        raise ValueError('Base labels must be zero-based and contiguous')
    # The model is still at its pretrained state: Q/V updates are initialized to zero.
    teacher = copy.deepcopy(model.clip_model.visual).eval().requires_grad_(False)
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
    names = [data_manager.class_names[c] for c in base_ids]
    with torch.no_grad():
        text, _ = model.inference_text_feature(names, model.template, 0)
        text = F.normalize(text.float(), dim=-1).to(model.device)
        scale = model.clip_model.logit_scale.exp().float().detach()
    dataset = data_manager.get_dataset(0, source='train', mode='train')
    loader = DataLoader(dataset, batch_size=cfg.DATALOADER.TRAIN.BATCH_SIZE_BASE,
                        shuffle=True, num_workers=cfg.DATALOADER.NUM_WORKERS,
                        pin_memory=True, drop_last=False)
    optimizer = torch.optim.AdamW(parameters, lr=cfg.LORA.LR,
                                  weight_decay=cfg.LORA.WEIGHT_DECAY)
    total_steps = cfg.LORA.EPOCHS * len(loader)
    if total_steps < 1:
        raise ValueError('Empty Base LoRA training loader')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    model.clip_model.visual.train()
    history = []
    step = 0
    for epoch in range(cfg.LORA.EPOCHS):
        total, count = 0., 0
        for batch in loader:
            images, labels = model.parse_batch(batch)
            image = F.normalize(model.clip_model.encode_image(images).float(), dim=-1)
            logits = scale * image @ text.T
            ce = F.cross_entropy(logits, labels)
            with torch.no_grad():
                reference = F.normalize(teacher(images).float(), dim=-1)
            anchor = F.mse_loss(image, reference)
            loss = ce + cfg.LORA.ANCHOR_WEIGHT * anchor
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
    print(f'Base Vision LoRA merged after {step} updates', flush=True)
    return history
