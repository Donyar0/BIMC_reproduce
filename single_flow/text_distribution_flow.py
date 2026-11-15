"""Text/visual Gaussian flow with distribution-only incremental replay."""
import math
import copy
import torch
import torch.nn.functional as F
from .fm import DeepFlowMatchingNet
from .image_text_flow import scheduled_lr, transport
from .s1_flow import unit

METHOD = 'gaussian-text-visual-v2'
DIRECTION = 'text_to_visual'


def distributions(stats, base=None, shrink=0.1, transfer_temperature=16.):
    """Only class statistics are retained; no image exemplars are stored."""
    ids = [int(x) for x in stats['class_index']]
    text = unit(stats['text_proto'].detach().float()).cpu()
    desc = unit(stats['description_features'].detach().float()).cpu()
    desc_labels = stats['description_targets'].detach().long().cpu()
    features = unit(stats['images_features'].detach().float()).cpu()
    labels = stats['images_targets'].detach().long().cpu()
    visual = unit(torch.stack([features[labels == c].mean(0) for c in ids]))
    dim = text.shape[1]
    text_sigma, empirical = [], []
    for row, c in enumerate(ids):
        group = desc[desc_labels == c]
        if len(group) < 2:
            raise ValueError(f'Class {c} needs at least two descriptions')
        residual = group-text[row]
        residual -= (residual @ text[row]).unsqueeze(1)*text[row]
        text_sigma.append(residual.square().sum(-1).mean().sqrt()/math.sqrt(dim))
        images = features[labels == c]
        centered = images-images.mean(0)
        empirical.append(centered.T @ centered / max(len(images)-1, 1))
    sigma = torch.stack(text_sigma).clamp_min(1e-4)
    empirical = torch.stack(empirical)
    if base is None:
        pooled = empirical.mean(0)
        cov = (1-shrink)*empirical+shrink*pooled.unsqueeze(0)
        base_cov, base_text = cov.clone(), text.clone()
    else:
        base_cov, base_text = base['base_cov'], base['base_text']
        weights = torch.softmax(transfer_temperature*text@base_text.T, dim=-1)
        transferred = torch.einsum('cb,bij->cij', weights, base_cov)
        counts = torch.tensor([(labels == c).sum().item() for c in ids], dtype=torch.float32)
        support_weight = ((counts-1).clamp_min(0)/(counts-1+dim)).view(-1,1,1)
        cov = support_weight*empirical+(1-support_weight)*transferred
    diag = cov.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-6)
    eye = torch.eye(dim).unsqueeze(0)
    cov = cov + (0.05*diag).view(-1,1,1)*eye + 1e-6*eye
    calibrated=unit(stats['image_proto'].detach().float()).cpu()
    return dict(class_ids=ids, text=text, real=visual,
                bank_real=calibrated, visual_target=calibrated,
                support_features=features, support_labels=labels,
                sigma=sigma, cov=cov,
                base_cov=base_cov, base_text=base_text,
                session_lengths=[len(ids)],
                description_features=desc, description_labels=desc_labels)


def combine(old, new):
    if set(old['class_ids']) & set(new['class_ids']):
        raise ValueError('Repeated class IDs')
    return dict(class_ids=old['class_ids']+new['class_ids'],
                **{key:torch.cat((old[key],new[key])) for key in
                   ('text','real','visual_target','sigma','cov','description_features','description_labels')},
                base_cov=old['base_cov'], base_text=old['base_text'],
                support_features=new['support_features'], support_labels=new['support_labels'],
                session_lengths=old['session_lengths'] + new['session_lengths'])


def sample_old_rows(session_lengths, count, device):
    """Split old replay between Base and earlier incremental sessions."""
    past = session_lengths[:-1]
    if not past or any(length <= 0 for length in past) or count < 0:
        raise ValueError('Old-class replay needs nonempty historical sessions')
    base_count = count if len(past) == 1 else count // 2
    base_rows = torch.randint(past[0], (base_count,), device=device)
    inc_count = count-base_count
    if not inc_count:
        return base_rows
    sizes = torch.tensor(past[1:], device=device)
    session = torch.randint(len(sizes), (inc_count,), device=device)
    starts = past[0] + torch.cat((sizes.new_zeros(1), sizes.cumsum(0)[:-1]))
    offset = (torch.rand(inc_count, device=device) * sizes[session]).long()
    rows = torch.cat((base_rows, starts[session]+offset))
    return rows[torch.randperm(count, device=device)]


def sample_pairs(memory, rows, noise_scale, visual_scale, max_radius, device, chol_all=None):
    text = memory['text'].to(device)[rows]
    sigma = memory['sigma'].to(device)[rows].unsqueeze(1)
    x0 = unit(text+torch.randn_like(text)*sigma*noise_scale)
    real = memory['real'].to(device)[rows]
    chol = (chol_all if chol_all is not None else torch.linalg.cholesky(memory['cov'].to(device)))[rows]
    noise = torch.bmm(chol, torch.randn(len(rows), real.shape[1], 1, device=device)).squeeze(-1)
    norm = noise.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    noise = noise*(max_radius/norm).clamp(max=1)
    x1 = unit(real+visual_scale*noise)
    return x0,x1


def sample_funnel_pairs(memory, rows, old, noise_scale, device, chol_all):
    """Forward half targets fixed visual centers; reverse half ends at images."""
    count=len(rows)
    forward=torch.randperm(count,device=device)[:count//2]
    reverse_mask=torch.ones(count,dtype=torch.bool,device=device)
    reverse_mask[forward]=False
    centers=memory['text'].to(device)[rows]
    x0=centers.clone()
    x1=memory['visual_target'].to(device)[rows].clone()
    if len(forward):
        sigma=memory['sigma'].to(device)[rows[forward]].unsqueeze(1)
        x0[forward]=unit(centers[forward]+noise_scale*sigma*torch.randn_like(centers[forward]))
    reverse_old=reverse_mask & old
    if reverse_old.any():
        picked=rows[reverse_old]
        chol=chol_all[picked]
        noise=torch.bmm(chol,torch.randn(len(picked),x1.shape[1],1,device=device)).squeeze(-1)
        x1[reverse_old]=unit(memory['visual_target'].to(device)[picked]+noise)
    reverse_current=reverse_mask & ~old
    if reverse_current.any():
        support=memory['support_features'].to(device)
        labels=memory['support_labels'].to(device)
        class_ids=memory['class_ids']
        chosen=[]
        for row in rows[reverse_current].tolist():
            options=torch.nonzero(labels==class_ids[row],as_tuple=False).flatten()
            if not len(options):
                raise ValueError(f'Missing real support feature for class {class_ids[row]}')
            chosen.append(options[torch.randint(len(options),(1,),device=device)])
        x1[reverse_current]=support[torch.cat(chosen)]
    return x0,x1,reverse_mask


def _train(checkpoint, memory, *, steps, batch_size, lr, weight_decay, seed, device,
           noise_scale, visual_scale, max_radius, replay_fraction,
           warmup_ratio, min_lr_ratio, blocks=None, hidden=None, distill_weight=0.,
           replay_strategy='base_incremental_half', contrast_weight=0.,
           contrast_temperature=20.,funnel_mode=False):
    if steps < 1 or batch_size < 1 or lr <= 0 or weight_decay < 0 or not 0 <= replay_fraction <= 1:
        raise ValueError('Invalid training parameters')
    torch.manual_seed(seed)
    dim = memory['text'].shape[1]
    if checkpoint is None:
        arch = dict(in_channels=dim, model_channels=hidden or dim,
                    out_channels=dim, num_res_blocks=blocks)
        network = DeepFlowMatchingNet(**arch).to(device)
    else:
        arch = checkpoint['architecture']
        network = DeepFlowMatchingNet(**arch).to(device)
        network.load_state_dict(checkpoint['state_dict'])
    if distill_weight < 0 or contrast_weight < 0 or contrast_temperature <= 0:
        raise ValueError('Invalid auxiliary loss settings')
    if replay_strategy not in ('base_incremental_half', 'class_uniform'):
        raise ValueError('Unknown replay strategy')
    prev_model = copy.deepcopy(network).eval() if checkpoint is not None and distill_weight else None
    if prev_model is not None:
        prev_model.requires_grad_(False)
    network.train()
    opt = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=weight_decay)
    nclass = len(memory['class_ids'])
    current_count = memory['session_lengths'][-1]
    old_count = nclass-current_count
    history=[]
    distill_history=[]
    contrast_history=[]
    chol_all = torch.linalg.cholesky(memory['cov'].to(device))
    visual_means = unit(memory['real'].to(device)) if contrast_weight else None
    for step in range(steps):
        if old_count == 0:
            rows = torch.randint(nclass,(batch_size,),device=device)
            old = torch.zeros(batch_size,dtype=torch.bool,device=device)
        else:
            # Exact per-batch split, including late sessions with many old classes.
            old = torch.zeros(batch_size,dtype=torch.bool,device=device)
            old[:round(batch_size*replay_fraction)] = True
            old = old[torch.randperm(batch_size,device=device)]
            rows = torch.empty(batch_size,dtype=torch.long,device=device)
            if replay_strategy == 'class_uniform':
                rows[old] = torch.randint(old_count,(int(old.sum()),),device=device)
            else:
                rows[old] = sample_old_rows(memory['session_lengths'],int(old.sum()),device)
            rows[~old] = torch.randint(old_count,nclass,(int((~old).sum()),),device=device)
        if funnel_mode:
            x0,x1,reverse_mask=sample_funnel_pairs(memory,rows,old,noise_scale,device,chol_all)
            t=torch.rand(batch_size,1,device=device)*0.5
            t[reverse_mask]+=0.5
        else:
            x0,x1=sample_pairs(memory,rows,noise_scale,visual_scale,max_radius,device,chol_all)
            t=torch.rand(batch_size,1,device=device)
        xt = (1-t)*x0+t*x1
        velocity = network(xt,t)
        fm_loss = (velocity-(x1-x0)).square().sum(-1).mean()
        loss = fm_loss
        if prev_model is not None and old.any():
            with torch.no_grad():
                previous_velocity = prev_model(xt[old],t[old])
            distill_loss = (velocity[old]-previous_velocity).square().sum(-1).mean()
            loss = loss + distill_weight*distill_loss
            distill_history.append(distill_loss.item())
        if contrast_weight:
            # Prototype columns follow memory row order; rows are the class targets.
            end_hat = unit(xt+(1-t)*velocity)
            contrast_loss = F.cross_entropy(contrast_temperature*end_hat@visual_means.T,
                                            rows)
            loss = loss + contrast_weight*contrast_loss
            contrast_history.append(contrast_loss.item())
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite Flow loss')
        scheduled_lr(opt,step,steps,lr,warmup_ratio,min_lr_ratio)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        history.append(loss.item())
    result = dict(format_version=2,architecture=arch,
                  state_dict={k:v.detach().cpu().clone() for k,v in network.state_dict().items()},
                  flow_direction=DIRECTION,method=METHOD,
                  loss_history=history if checkpoint is None else checkpoint['loss_history']+history,
                  distill_weight=distill_weight, replay_strategy=replay_strategy,
                  distill_history=distill_history,
                  contrast_weight=contrast_weight,
                  contrast_temperature=contrast_temperature,
                  contrast_history=contrast_history, funnel_mode=funnel_mode)
    if distill_history:
        print(f'adapt fm+distill_loss={history[-1]:.6f} '
              f'distill_loss={sum(distill_history)/len(distill_history):.6f}',flush=True)
    if contrast_history:
        print(f'contrast_loss={sum(contrast_history)/len(contrast_history):.6f}',flush=True)
    return result


def fit(memory, *, epochs, steps_per_epoch, batch_size, lr, weight_decay, blocks, hidden,
        seed, device, noise_scale=1., visual_scale=1., max_radius=0.5,
        warmup_ratio=0.05,min_lr_ratio=0.01,
        contrast_weight=0.,contrast_temperature=20.,funnel_mode=False):
    return _train(None,memory,steps=epochs*steps_per_epoch,batch_size=batch_size,lr=lr,
                  weight_decay=weight_decay,seed=seed,device=device,noise_scale=noise_scale,
                  visual_scale=visual_scale,max_radius=max_radius,replay_fraction=0.,
                  warmup_ratio=warmup_ratio,min_lr_ratio=min_lr_ratio,blocks=blocks,hidden=hidden,
                  contrast_weight=contrast_weight,contrast_temperature=contrast_temperature,
                  funnel_mode=funnel_mode)


def adapt(checkpoint,memory,*,steps,batch_size,lr,weight_decay,seed,device,
          noise_scale,visual_scale,max_radius,replay_fraction,
          warmup_ratio=0.05,min_lr_ratio=0.01,distill_weight=0.,
          replay_strategy='base_incremental_half',contrast_weight=0.,
          contrast_temperature=20.,funnel_mode=False):
    return _train(checkpoint,memory,steps=steps,batch_size=batch_size,lr=lr,
                  weight_decay=weight_decay,seed=seed,device=device,noise_scale=noise_scale,
                  visual_scale=visual_scale,max_radius=max_radius,replay_fraction=replay_fraction,
                  warmup_ratio=warmup_ratio,min_lr_ratio=min_lr_ratio,
                  distill_weight=distill_weight,replay_strategy=replay_strategy,
                  contrast_weight=contrast_weight,contrast_temperature=contrast_temperature,
                  funnel_mode=funnel_mode)


def load_model(checkpoint,device):
    if checkpoint.get('flow_direction') != DIRECTION or checkpoint.get('method') != METHOD:
        raise ValueError('Incompatible Flow checkpoint')
    model=DeepFlowMatchingNet(**checkpoint['architecture']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval()


@torch.no_grad()
def build_bank(checkpoint,current,*,steps,num_base,beta_base,beta_inc,device,
               previous_bank=None,solver='euler',bank_mode='flow'):
    ids=current['class_ids']
    if previous_bank is not None and set(ids)&set(previous_bank['class_ids']):
        raise ValueError('Repeated class IDs')
    text=current['text']
    real=(current['bank_real'] if bank_mode in ('bimc_incremental','pure_visual','calibrated_flow20')
          else current['real'])
    generated=transport(load_model(checkpoint,device),text.to(device),steps,solver=solver).cpu()
    beta=torch.full((len(ids),1),beta_inc)
    beta[:num_base]=beta_base
    prototypes=unit((1-beta)*real+beta*generated)
    if bank_mode == 'pure_visual':
        prototypes=unit(real)
    elif bank_mode == 'calibrated_flow20':
        prototypes=unit(0.8*real+0.2*generated)
    elif bank_mode == 'bimc_incremental' and previous_bank is not None:
        prototypes=unit(0.3*current['bank_text']+0.7*real)
    elif bank_mode not in ('flow','bimc_incremental'):
        raise ValueError('Unknown visual Bank mode')
    new=dict(class_ids=ids,text=text,real=real,generated=generated,
             prototypes=prototypes,
             description_features=current['description_features'],
             description_labels=current['description_labels'])
    if previous_bank is None:
        return new
    return dict(class_ids=previous_bank['class_ids']+ids,
                **{key:torch.cat((previous_bank[key],new[key])) for key in
                   ('text','real','generated','prototypes','description_features','description_labels')})


@torch.no_grad()
def classify_bidirectional(model,image_features,bank,*,alpha,gamma,steps,
                           solver='euler',cycle_scale=0.01,ensemble_alpha=0.8,
                           num_base=0,covariance=None,cov_inv=None,
                           cycle_gate=False,cosine_temperature=1.,
                           knn_temperature=25.,cov_divisor=512.,return_components=False,
                           gamma_base=None,gamma_inc=None,score_mode='probability'):
    if not (0<=alpha<=1 and 0<=gamma<=1 and cycle_scale>0 and
            cosine_temperature>0 and knn_temperature>0 and cov_divisor>0 and
            0<=ensemble_alpha<=1):
        raise ValueError('Invalid scoring parameters')
    z=unit(image_features)
    t_hat=transport(model,z,steps,reverse=True,solver=solver)
    if score_mode == 'cosine_sum':
        visual=z@bank['prototypes'].to(z.device).T
        text=t_hat@bank['text'].to(z.device).T
        descriptions=unit(bank['description_features'].to(z.device))
        labels=bank['description_labels'].to(z.device)
        similarities=t_hat@descriptions.T
        desc_max=torch.stack([similarities[:,labels==c].max(-1).values
                              for c in bank['class_ids']],dim=-1)
        text=text+0.5*desc_max
        scores=visual+text
        error=torch.zeros(len(z),1,device=z.device)
        ids=torch.tensor(bank['class_ids'],device=z.device)
        if return_components:
            return ids[scores.argmax(-1)],scores,error,dict(
                visual=visual,text=text,flow=scores)
        return ids[scores.argmax(-1)],scores,error
    z_reconstructed=transport(model,t_hat,steps,solver=solver)
    error=(z-z_reconstructed).square().sum(-1,keepdim=True)
    base_weight=gamma if gamma_base is None else gamma_base
    inc_weight=gamma if gamma_inc is None else gamma_inc
    if not 0<=base_weight<=1 or not 0<=inc_weight<=1:
        raise ValueError('Invalid per-group gamma')
    gamma_columns=torch.full((1,len(bank['class_ids'])),inc_weight,device=z.device)
    gamma_columns[:,:num_base]=base_weight
    gamma_z=1-(1-gamma_columns)*torch.exp(-error/cycle_scale) if cycle_gate else gamma_columns
    text_query=unit((1-alpha)*z+alpha*t_hat)
    visual_prob=F.softmax(cosine_temperature*(z@bank['prototypes'].to(z.device).T),dim=-1)
    text_prob=F.softmax(cosine_temperature*(text_query@bank['text'].to(z.device).T),dim=-1)
    flow_prob=gamma_z*visual_prob+(1-gamma_z)*text_prob
    if ensemble_alpha<1:
        if covariance is None and cov_inv is None:
            raise ValueError('Covariance is required for geometry priors')
        inv = cov_inv.to(z.device) if cov_inv is not None else torch.linalg.pinv(covariance.to(z.device).float().mean(0))
        dist=z[:,None,:]-bank['real'].to(z.device)[None,:,:]
        cov_logits=-torch.einsum('bcd,de,bce->bc',dist,inv,dist)
        prob_cov=F.softmax(cov_logits/cov_divisor,dim=-1)
        desc=unit(bank['description_features'].to(z.device))
        desc_labels=bank['description_labels'].to(z.device)
        similarities=z@desc.T
        knn=torch.stack([similarities[:,desc_labels==c].max(-1).values
                         for c in bank['class_ids']],dim=-1)
        prob_knn=F.softmax(knn_temperature*knn,dim=-1)
        priors=torch.cat((prob_cov[:,:num_base],prob_knn[:,num_base:]),dim=-1)
        scores=ensemble_alpha*flow_prob+(1-ensemble_alpha)*priors
    else:
        scores=flow_prob
    ids=torch.tensor(bank['class_ids'],device=z.device)
    if return_components:
        return ids[scores.argmax(-1)],scores,error,dict(
            visual=visual_prob,text=text_prob,flow=flow_prob,
            cov=prob_cov if ensemble_alpha<1 else None,
            knn=prob_knn if ensemble_alpha<1 else None,
            gamma=gamma_z)
    return ids[scores.argmax(-1)],scores,error
