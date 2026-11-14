"""Text/visual Gaussian flow with distribution-only incremental replay."""
import math
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
    return dict(class_ids=ids, text=text, real=visual, sigma=sigma, cov=cov,
                base_cov=base_cov, base_text=base_text,
                description_features=desc, description_labels=desc_labels)


def combine(old, new):
    if set(old['class_ids']) & set(new['class_ids']):
        raise ValueError('Repeated class IDs')
    return dict(class_ids=old['class_ids']+new['class_ids'],
                **{key:torch.cat((old[key],new[key])) for key in
                   ('text','real','sigma','cov','description_features','description_labels')},
                base_cov=old['base_cov'], base_text=old['base_text'])


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


def _train(checkpoint, memory, *, steps, batch_size, lr, weight_decay, seed, device,
           noise_scale, visual_scale, max_radius, replay_fraction,
           warmup_ratio, min_lr_ratio, blocks=None, hidden=None):
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
    network.train()
    opt = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=weight_decay)
    nbase = len(memory['base_text'])
    nclass = len(memory['class_ids'])
    history=[]
    chol_all = torch.linalg.cholesky(memory['cov'].to(device))
    for step in range(steps):
        if nclass == nbase:
            rows = torch.randint(nclass,(batch_size,),device=device)
        else:
            old = torch.rand(batch_size,device=device)<replay_fraction
            rows = torch.empty(batch_size,dtype=torch.long,device=device)
            rows[old] = torch.randint(nclass-(nclass-nbase),(int(old.sum()),),device=device)
            rows[~old] = torch.randint(nbase,nclass,(int((~old).sum()),),device=device)
        x0,x1 = sample_pairs(memory,rows,noise_scale,visual_scale,max_radius,device,chol_all)
        t = torch.rand(batch_size,1,device=device)
        xt = (1-t)*x0+t*x1
        loss = (network(xt,t)-(x1-x0)).square().sum(-1).mean()
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
                  loss_history=history if checkpoint is None else checkpoint['loss_history']+history)
    return result


def fit(memory, *, epochs, steps_per_epoch, batch_size, lr, weight_decay, blocks, hidden,
        seed, device, noise_scale=1., visual_scale=1., max_radius=0.5,
        warmup_ratio=0.05,min_lr_ratio=0.01):
    return _train(None,memory,steps=epochs*steps_per_epoch,batch_size=batch_size,lr=lr,
                  weight_decay=weight_decay,seed=seed,device=device,noise_scale=noise_scale,
                  visual_scale=visual_scale,max_radius=max_radius,replay_fraction=0.,
                  warmup_ratio=warmup_ratio,min_lr_ratio=min_lr_ratio,blocks=blocks,hidden=hidden)


def adapt(checkpoint,memory,*,steps,batch_size,lr,weight_decay,seed,device,
          noise_scale,visual_scale,max_radius,replay_fraction,
          warmup_ratio=0.05,min_lr_ratio=0.01):
    return _train(checkpoint,memory,steps=steps,batch_size=batch_size,lr=lr,
                  weight_decay=weight_decay,seed=seed,device=device,noise_scale=noise_scale,
                  visual_scale=visual_scale,max_radius=max_radius,replay_fraction=replay_fraction,
                  warmup_ratio=warmup_ratio,min_lr_ratio=min_lr_ratio)


def load_model(checkpoint,device):
    if checkpoint.get('flow_direction') != DIRECTION or checkpoint.get('method') != METHOD:
        raise ValueError('Incompatible Flow checkpoint')
    model=DeepFlowMatchingNet(**checkpoint['architecture']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval()


@torch.no_grad()
def build_bank(checkpoint,current,*,steps,num_base,beta_base,beta_inc,device,
               previous_bank=None,solver='euler'):
    ids=current['class_ids']
    if previous_bank is not None and set(ids)&set(previous_bank['class_ids']):
        raise ValueError('Repeated class IDs')
    text=current['text']
    real=current['real']
    generated=transport(load_model(checkpoint,device),text.to(device),steps,solver=solver).cpu()
    beta=torch.full((len(ids),1),beta_inc)
    beta[:num_base]=beta_base
    new=dict(class_ids=ids,text=text,real=real,generated=generated,
             prototypes=unit((1-beta)*real+beta*generated),
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
                           num_base=0,covariance=None,cov_inv=None):
    if not (0<=alpha<=1 and 0<=gamma<=1 and cycle_scale>0 and 0<=ensemble_alpha<=1):
        raise ValueError('Invalid scoring parameters')
    z=unit(image_features)
    t_hat=transport(model,z,steps,reverse=True,solver=solver)
    z_reconstructed=transport(model,t_hat,steps,solver=solver)
    error=(z-z_reconstructed).square().sum(-1,keepdim=True)
    gamma_z=1-(1-gamma)*torch.exp(-error/cycle_scale)
    text_query=unit((1-alpha)*z+alpha*t_hat)
    visual_prob=F.softmax(z@bank['prototypes'].to(z.device).T,dim=-1)
    text_prob=F.softmax(text_query@bank['text'].to(z.device).T,dim=-1)
    flow_prob=gamma_z*visual_prob+(1-gamma_z)*text_prob
    if ensemble_alpha<1:
        if covariance is None:
            raise ValueError('Covariance is required for geometry priors')
        inv = cov_inv.to(z.device) if cov_inv is not None else torch.linalg.pinv(covariance.to(z.device).float().mean(0))
        dist=z[:,None,:]-bank['real'].to(z.device)[None,:,:]
        cov_logits=-torch.einsum('bcd,de,bce->bc',dist,inv,dist)
        prob_cov=F.softmax(cov_logits/512,dim=-1)
        desc=unit(bank['description_features'].to(z.device))
        desc_labels=bank['description_labels'].to(z.device)
        similarities=z@desc.T
        knn=torch.stack([similarities[:,desc_labels==c].max(-1).values
                         for c in bank['class_ids']],dim=-1)
        prob_knn=F.softmax(knn,dim=-1)
        priors=torch.cat((prob_cov[:,:num_base],prob_knn[:,num_base:]),dim=-1)
        scores=ensemble_alpha*flow_prob+(1-ensemble_alpha)*priors
    else:
        scores=flow_prob
    ids=torch.tensor(bank['class_ids'],device=z.device)
    return ids[scores.argmax(-1)],scores,error
