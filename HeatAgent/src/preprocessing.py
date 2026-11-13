import cv2
import numpy as np

def normalize_16to8bit(image, lower_p=0.5, upper_p=99.5):
    """Percentile-window 16-bit radiographs to uint8."""
    if image is None: raise ValueError("Image could not be read")
    if image.ndim == 3: image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype == np.uint8: return image
    image = image.astype(np.float32)
    low, high = np.percentile(image, [lower_p, upper_p])
    if high <= low: return np.zeros(image.shape, dtype=np.uint8)
    return (np.clip((image-low)/(high-low), 0, 1)*255).astype(np.uint8)

def read_radiograph(path):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None: raise FileNotFoundError(f"Unable to read image: {path}")
    return normalize_16to8bit(raw)

def get_bone_mask_fallback(image_8bit):
    """CLAHE + Otsu heuristic mask; not a clinical segmentation model."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image_8bit)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
