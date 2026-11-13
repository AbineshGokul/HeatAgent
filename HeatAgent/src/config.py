from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class Config:
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    backbone_name: str = "vit_base_patch14_reg4_dinov2.lvd142m"
    num_workers: int = 2
    seed: int = 42
    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
CFG = Config()
