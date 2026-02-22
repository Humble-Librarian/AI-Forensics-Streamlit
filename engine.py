import os
import cv2
import torch
import numpy as np
import datetime
import gc
import gdown
from PIL import Image, ImageDraw
from torchvision import transforms
from facenet_pytorch import MTCNN
import config
from models import SpatialXception, SRMXception, DeepfakeLSTM

def smart_load_state_dict(model, path, device, log_func=print):
    """
    Intelligently loads a state dict into a model by trying multiple structural strategies.
    Handles DataParallel prefixes and backbone-only checkpoints.
    
    Args:
        model (nn.Module): The target model instance.
        path (str): Path to the .pth weight file.
        device (torch.device): Device to load the tensors onto.
        log_func (callable): Function for logging status messages.
        
    Returns:
        bool: True if loading was successful, False otherwise.
    """
    try:
        if not os.path.exists(path):
            return False
            
        state_dict = torch.load(path, map_location=device)
        
        # Check for DataParallel/DistributedDataParallel prefixing
        if all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        # Strategy 1: Direct Load
        try:
            model.load_state_dict(state_dict, strict=True)
            return True
        except:
            # Strategy 2: Sub-module Load Fallback
            for target_module in ['model', 'backbone']:
                if hasattr(model, target_module):
                    try:
                        load_target = getattr(model, target_module)
                        load_target.load_state_dict(state_dict, strict=True)
                        return True
                    except:
                        continue
        return False
    except Exception as e:
        log_func(f"Loading Error ({os.path.basename(path)}): {e}")
        return False

def load_all_models(device, log_func=print):
    """
    Initializes all neural architectures and downloads/loads their weights.
    
    Args:
        device (torch.device): GPU or CPU device.
        log_func (callable): UI logging function.
        
    Returns:
        dict: A dictionary of initialized and loaded models, or None on failure.
    """
    models = {}
    
    try:
        # Download missing models
        for key in config.GDRIVE_IDS:
            if not os.path.exists(config.MODEL_FILES[key]):
                log_func(f"Downloading {key} model...")
                gdown.download(id=config.GDRIVE_IDS[key], output=config.MODEL_FILES[key], quiet=False)

        # Spatial
        models['spatial'] = SpatialXception(num_classes=2).to(device)
        if not smart_load_state_dict(models['spatial'], config.MODEL_FILES['spatial'], device, log_func):
            return None
        models['spatial'].eval()
        
        # SRM
        models['srm'] = SRMXception(num_classes=1).to(device)
        if not smart_load_state_dict(models['srm'], config.MODEL_FILES['srm'], device, log_func):
            return None
        models['srm'].eval()
        
        # LSTM
        models['lstm'] = DeepfakeLSTM(input_size=4096).to(device)
        if not smart_load_state_dict(models['lstm'], config.MODEL_FILES['lstm'], device, log_func):
            return None
        models['lstm'].eval()
        
        models['mtcnn'] = MTCNN(keep_all=False, select_largest=True, device=device, margin=14)
        return models
    except Exception as e:
        log_func(f"Critical System Initialization Error: {e}")
        return None

def extract_frames(video_path, seq_length):
    """
    Uniformly samples N frames from a video file.
    
    Args:
        video_path (str): Path to the video file.
        seq_length (int): Number of frames to extract.
        
    Returns:
        list: List of PIL Image objects.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total-1, seq_length, dtype=int) if total > seq_length else range(total)
    
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx in indices: 
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        idx += 1
    cap.release()
    while len(frames) < seq_length and len(frames) > 0: 
        frames.append(frames[-1].copy())
    return frames

def run_inference(video_path, seq_length, models, device, log_func, progress_callback=None):
    """
    Executes the full deepfake detection pipeline: face detection -> feature extraction -> fusion -> temporal analysis.
    
    Args:
        video_path (str): Path to the video file.
        seq_length (int): Number of frames to process.
        models (dict): Dictionary of loaded model instances.
        device (torch.device): Processing device.
        log_func (callable): Logging function for the UI.
        progress_callback (callable): UI progress bar update function.
        
    Returns:
        dict: Analysis results including scores and per-frame probability maps.
    """
    
    trans_spatial = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    trans_srm = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)), 
        transforms.ToTensor()
    ])

    frames = extract_frames(video_path, seq_length)
    if not frames: return None

    batch_s, batch_f = [], []
    log_func(">>> EXTRACTING FACIAL BIOMETRICS...")

    thumb_saved = False
    for i, f in enumerate(frames):
        boxes, _ = models['mtcnn'].detect(f)
        
        if boxes is not None: 
            face = f.crop(boxes[0])
            face_final = face.resize((config.IMG_SIZE, config.IMG_SIZE), getattr(Image, 'Resampling', Image).BILINEAR)
        else: 
            face_final = f.resize((config.IMG_SIZE, config.IMG_SIZE), getattr(Image, 'Resampling', Image).BILINEAR)
        
        if not thumb_saved and boxes is not None:
             draw = ImageDraw.Draw(f)
             draw.rectangle(boxes[0].tolist(), outline="#ef4444", width=4)
             f.save("temp_thumb.jpg")
             thumb_saved = True
             
        batch_s.append(trans_spatial(face_final))
        batch_f.append(trans_srm(face_final))
        if progress_callback:
            progress_callback((i + 1) / seq_length)

    if not thumb_saved and frames:
        frames[0].save("temp_thumb.jpg")

    inp_s = torch.stack(batch_s).to(device)
    inp_f = torch.stack(batch_f).to(device)

    log_func(">>> EXECUTING NEURAL INFERENCE (BATCHED)...")
    
    with torch.no_grad():
        feat_s, log_s = models['spatial'](inp_s)
        feat_f, log_f = models['srm'](inp_f)
        
        probs_s = torch.softmax(log_s, dim=1)[:, 1].cpu().numpy()
        probs_f = torch.sigmoid(log_f).cpu().squeeze().numpy()
        
        if len(probs_f.shape) == 0:
            probs_f = np.array([probs_f])
            
        val_s = float(np.mean(probs_s))
        val_f = float(np.mean(probs_f))
        
        log_func(">>> COMPUTING TEMPORAL CONSISTENCY...")
        combined = torch.cat((feat_s, feat_f), dim=1).unsqueeze(0)
        val_t = torch.sigmoid(models['lstm'](combined)).item()

    frame_scores = ((probs_s + probs_f) / 2).tolist()
    final = (0.4 * val_s) + (0.4 * val_f) + (0.2 * val_t)
    
    log_func(">>> ANALYSIS COMPLETE.")
    
    results = {
        "final": final, "s": val_s, "f": val_f, "t": val_t,
        "filename": os.path.basename(video_path),
        "frame_scores": frame_scores,
        "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "seq_length": seq_length
    }
    
    gc.collect()
    return results
