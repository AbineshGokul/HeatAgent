import math
import numpy as np
import torch
import torch.nn.functional as F

def generate_attention_and_deviation_maps(model, image_tensor, prototype_bank, image_size=224):
    """Build patch activation and distance-to-normal-prototype maps (DINOv2 tokens)."""
    model.eval(); x=image_tensor.detach().clone().requires_grad_(True)
    features=model.backbone.forward_features(x)
    logits=model.classifier(features[:,0,:]).squeeze(-1); logits.sum().backward()
    patches=features[:,5:,:].detach()  # CLS + four register tokens excluded
    batch,count,_=patches.shape; side=int(math.sqrt(count))
    if side*side!=count: raise ValueError(f"Non-square patch grid: {count}")
    heat=torch.relu(patches).mean(dim=-1).view(batch,1,side,side)
    heat=F.interpolate(heat,size=(image_size,image_size),mode="bilinear",align_corners=False)
    heat=(heat-heat.amin((2,3),keepdim=True))/(heat.amax((2,3),keepdim=True)-heat.amin((2,3),keepdim=True)+1e-8)
    p=F.normalize(patches,dim=-1); bank=F.normalize(prototype_bank,dim=-1)
    sim=torch.einsum("bnc,kc->bnk",p,bank)
    dev=(1-sim.max(dim=-1).values).view(batch,1,side,side)
    dev=F.interpolate(dev,size=(image_size,image_size),mode="bilinear",align_corners=False)
    dev=(dev-dev.amin((2,3),keepdim=True))/(dev.amax((2,3),keepdim=True)-dev.amin((2,3),keepdim=True)+1e-8)
    return torch.sigmoid(logits).detach(),heat.squeeze(1).detach(),dev.squeeze(1).detach()

def compute_verifier_score(heatmap, deviation_map, bone_mask):
    """Heuristic verifier score; not clinically validated."""
    bone=(bone_mask>128).astype(np.float32)
    bone_mass=float((heatmap*bone).sum()/(heatmap.sum()+1e-8)) if bone.sum() else 0.0
    distribution=heatmap/(heatmap.sum()+1e-8)
    entropy=-np.sum((distribution+1e-12)*np.log(distribution+1e-12))/np.log(distribution.size)
    focus=float(np.clip(1.0-entropy*1.2,0,1))
    agreement=float(np.clip(1.0-np.mean(np.abs(heatmap-(1.0-deviation_map))),0,1))
    score=float(np.clip(.45*bone_mass+.30*focus+.25*agreement,0,1))
    return score,{"bone_mass":bone_mass,"focus":focus,"deviation_agreement":agreement}
