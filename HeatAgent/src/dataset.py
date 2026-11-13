import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from .preprocessing import read_radiograph

class FractureDataset(Dataset):
    """Loads rows from the GRAZ manifest produced by the Kaggle notebook."""
    def __init__(self, manifest, image_size=224, train=False):
        self.frame = manifest.reset_index(drop=True).copy()
        aug = [T.RandomHorizontalFlip(0.5), T.RandomRotation(10)] if train else []
        self.transform = T.Compose([T.ToPILImage(), T.Resize((image_size,image_size)), *aug,
            T.ToTensor(), T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
    def __len__(self): return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = read_radiograph(row.img_path)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return self.transform(image), torch.tensor(float(row.label),dtype=torch.float32), str(row.get("filestem",index))
