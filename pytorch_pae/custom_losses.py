import torch
import torch.nn.functional as F


def masked_chi2(recon, features, data, device):
    
    chi2 = (recon-features)**2*data['noise'].to(device).float() #data['noise']=1/var
    chi2 = torch.masked_select(chi2, data['mask'].to(device).bool())
    
    return torch.mean(chi2)


def contrastive_loss(feats,tau):
    cos_sim   = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
    mask      = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(mask, -9e15)
       
    pos_mask  = mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)

    cos_sim   = cos_sim / tau
    nll       = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll       = nll.mean()
    
    return nll
    
