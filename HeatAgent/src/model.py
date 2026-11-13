import math
import torch
from torch import nn
import timm

class LoRALinear(nn.Module):
    """Frozen linear projection with a trainable low-rank residual."""
    def __init__(self, original, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        self.original_linear = original
        self.original_linear.weight.requires_grad = False
        if original.bias is not None: original.bias.requires_grad = False
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, original.in_features))
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, rank))
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    def forward(self, x):
        update = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return self.original_linear(x) + self.scaling * update

def inject_lora_to_vit(backbone, rank=8, alpha=16, dropout=0.1):
    for p in backbone.parameters(): p.requires_grad = False
    for module in backbone.modules():
        if hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear):
            module.qkv = LoRALinear(module.qkv, rank, alpha, dropout)
    return backbone

class DINOv2LoRAFractureClassifier(nn.Module):
    def __init__(self, model_name="vit_base_patch14_reg4_dinov2.lvd142m", image_size=224,
                 lora_rank=8, lora_alpha=16, lora_dropout=0.1, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0,
                                          img_size=image_size, dynamic_img_pad=True)
        self.backbone = inject_lora_to_vit(self.backbone,lora_rank,lora_alpha,lora_dropout)
        self.classifier = nn.Sequential(nn.LayerNorm(self.backbone.num_features),
                                        nn.Dropout(0.2), nn.Linear(self.backbone.num_features,1))
    def forward(self, x): return self.classifier(self.backbone(x)).squeeze(-1)
