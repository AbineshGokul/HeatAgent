import argparse
from pathlib import Path
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm
from .config import CFG
from .dataset import FractureDataset
from .model import DINOv2LoRAFractureClassifier

def run_training(manifest_csv, output_dir="outputs", epochs=10):
    torch.manual_seed(CFG.seed)
    out=Path(output_dir); ckpt_dir=out/"checkpoints"; ckpt_dir.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(manifest_csv)
    train_df=frame[frame.split=="train"]; val_df=frame[frame.split=="val"]
    train_loader=DataLoader(FractureDataset(train_df,CFG.image_size,True),batch_size=CFG.batch_size,shuffle=True,num_workers=CFG.num_workers,pin_memory=True)
    val_loader=DataLoader(FractureDataset(val_df,CFG.image_size,False),batch_size=CFG.batch_size,shuffle=False,num_workers=CFG.num_workers,pin_memory=True)
    model=DINOv2LoRAFractureClassifier(CFG.backbone_name,CFG.image_size,CFG.lora_rank,CFG.lora_alpha,CFG.lora_dropout).to(CFG.device)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=CFG.learning_rate,weight_decay=CFG.weight_decay)
    criterion=nn.BCEWithLogitsLoss(); best_auc=0.0; history=[]
    for epoch in range(1,epochs+1):
        model.train(); train_losses=[]
        for x,y,_ in tqdm(train_loader,desc=f"Epoch {epoch}/{epochs} train"):
            x,y=x.to(CFG.device),y.to(CFG.device); optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(x),y); loss.backward(); optimizer.step(); train_losses.append(loss.item())
        model.eval(); val_losses=[]; probs=[]; targets=[]
        with torch.no_grad():
            for x,y,_ in tqdm(val_loader,desc=f"Epoch {epoch}/{epochs} val"):
                x,y=x.to(CFG.device),y.to(CFG.device); logits=model(x)
                val_losses.append(criterion(logits,y).item()); probs.extend(torch.sigmoid(logits).cpu().tolist()); targets.extend(y.cpu().tolist())
        auc=roc_auc_score(targets,probs) if len(set(targets))>1 else float("nan")
        f1=f1_score(targets,[int(p>=0.5) for p in probs],zero_division=0)
        row={"epoch":epoch,"train_loss":sum(train_losses)/len(train_losses),"val_loss":sum(val_losses)/len(val_losses),"val_auc":auc,"val_f1":f1}
        history.append(row); print(row)
        torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"best_val_auc":max(best_auc,auc)},ckpt_dir/"last_checkpoint.pt")
        if auc>best_auc:
            best_auc=auc
            torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"best_val_auc":best_auc},ckpt_dir/"best_checkpoint.pt")
    pd.DataFrame(history).to_csv(out/"training_history.csv",index=False)
    print(f"Best validation AUC: {best_auc:.4f}")
    return model,pd.DataFrame(history)

if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--manifest",required=True); parser.add_argument("--output",default="outputs"); parser.add_argument("--epochs",type=int,default=CFG.epochs)
    args=parser.parse_args(); run_training(args.manifest,args.output,args.epochs)
