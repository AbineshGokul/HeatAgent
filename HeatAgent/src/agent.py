import cv2
import numpy as np
from .preprocessing import get_bone_mask_fallback

def apply_clahe(image):
    return cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8)).apply(image)

def zoom_bone_roi(image,mask):
    points=np.argwhere(mask>128)
    if not len(points): return image
    y0,x0=points.min(axis=0); y1,x1=points.max(axis=0); h,w=image.shape
    y0,x0=max(0,y0-15),max(0,x0-15); y1,x1=min(h,y1+15),min(w,x1+15)
    crop=image[y0:y1,x0:x1]
    return cv2.resize(crop,(w,h)) if crop.size else image

def run_verifier_retry_loop(image,assess_fn,threshold=0.45,max_steps=2):
    """assess_fn(image) -> (fracture_probability, verifier_score, details)."""
    current=image.copy(); probability,score,details=assess_fn(current)
    trace=[{"step":0,"action":"initial_inspection","probability":probability,"verifier_score":score,"details":details}]
    actions=[("clahe_enhancement",lambda im:apply_clahe(im)),("zoom_bone_roi",lambda im:zoom_bone_roi(im,get_bone_mask_fallback(im)))]
    for step,(name,action) in enumerate(actions[:max_steps],1):
        if score>=threshold: break
        current=action(current); probability,score,details=assess_fn(current)
        trace.append({"step":step,"action":name,"probability":probability,"verifier_score":score,"details":details})
    decision="REFER_FOR_EXPERT_REVIEW" if score<threshold else ("FRACTURE_SUSPECTED" if probability>=0.5 else "NO_FRACTURE_SUSPECTED")
    return {"probability":probability,"verifier_score":score,"decision":decision,"trace":trace}
