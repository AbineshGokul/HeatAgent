import argparse
import cv2
import torch
from torchvision import transforms as T
from .config import CFG
from .model import DINOv2LoRAFractureClassifier
from .preprocessing import read_radiograph

def load_model(checkpoint_path):
    model=DINOv2LoRAFractureClassifier(CFG.backbone_name,CFG.image_size,CFG.lora_rank,CFG.lora_alpha,CFG.lora_dropout,pretrained=False).to(CFG.device)
    checkpoint=torch.load(checkpoint_path,map_location=CFG.device,weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval(); return model

def predict_image(image_path,checkpoint_path):
    model=load_model(checkpoint_path); image=read_radiograph(image_path); rgb=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    transform=T.Compose([T.ToPILImage(),T.Resize((CFG.image_size,CFG.image_size)),T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])
    tensor=transform(rgb).unsqueeze(0).to(CFG.device)
    with torch.no_grad(): probability=torch.sigmoid(model(tensor)).item()
    return {"fracture_probability":probability,"predicted_label":"fracture_suspected" if probability>=.5 else "no_fracture_suspected"}

if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--image",required=True); parser.add_argument("--checkpoint",required=True)
    args=parser.parse_args(); print(predict_image(args.image,args.checkpoint))
