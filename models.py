import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# --- SRM FILTER LAYER ---
class SRMConv2d(nn.Module):
    """
    SRM Filter Layer for extracting high-frequency noise artifacts.
    Implements 3 standard SRM kernels for spatial richness modeling.
    
    This layer uses fixed, non-trainable weights to compute image residuals 
    based on the SRM (Spatial Rich Model) filters, which helps in identifying 
    subtle manipulation footprints.
    """
    def __init__(self):
        super(SRMConv2d, self).__init__()
        self.channels = 3
        q1 = [[0, 0, 0, 0, 0],
              [0, -1, 2, -1, 0],
              [0, 2, -4, 2, 0],
              [0, -1, 2, -1, 0],
              [0, 0, 0, 0, 0]]
        q2 = [[-1, 2, -2, 2, -1],
              [2, -6, 8, -6, 2],
              [-2, 8, -12, 8, -2],
              [2, -6, 8, -6, 2],
              [-1, 2, -2, 2, -1]]
        q3 = [[0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 1, -2, 1, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]]
        
        q = torch.FloatTensor([q1, q2, q3])
        self.weight = nn.Parameter(q.unsqueeze(1), requires_grad=False)

    def forward(self, x):
        """Extract noise maps from the input image."""
        r = F.conv2d(x[:,0:1,:,:], self.weight, padding=2)
        g = F.conv2d(x[:,1:2,:,:], self.weight, padding=2)
        b = F.conv2d(x[:,2:3,:,:], self.weight, padding=2)
        noise_map = torch.cat([r, g, b], dim=1)
        return noise_map

# --- SPATIAL MODEL ---
class SpatialXception(nn.Module):
    """
    Xception-based Spatial Deepfake Detector.
    Analyzes pixel-level anomalies using a pretrained Xception backbone.
    
    The Xception architecture is effective at learning spatial artifacts
    left by deepfake generation processes (e.g., blending boundaries, 
    resolution inconsistencies).
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.model = timm.create_model('xception', pretrained=pretrained, num_classes=num_classes)
        
    def forward(self, x):
        """Returns pooled features and classification logits."""
        features = self.model.forward_features(x)
        # Global average pool to get feature vector
        pooled_features = torch.mean(features, dim=[2, 3])
        logits = self.model.fc(pooled_features)
        return pooled_features, logits

# --- SRM MODEL ---
class SRMXception(nn.Module):
    """
    Frequency-domain Deepfake Detector.
    Uses SRM filters to extract noise residuals before processing with Xception.
    
    By discarding image content and focusing on high-frequency noise, 
    this model captures traces of upsampling and GAN-generated textures.
    """
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        self.srm = SRMConv2d()
        self.compress = nn.Conv2d(9, 3, kernel_size=1) 
        self.backbone = timm.create_model('xception', pretrained=pretrained)
        if num_classes > 0:
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        else:
            self.backbone.fc = nn.Identity()
        
    def forward(self, x):
        """Returns pooled noise features and classification logits."""
        noise = self.srm(x)
        noise = self.compress(noise)
        features = self.backbone.forward_features(noise)
        # Global average pool to get feature vector
        pooled_features = torch.mean(features, dim=[2, 3])
        logits = self.backbone.fc(pooled_features)
        return pooled_features, logits

# --- TEMPORAL MODEL (BiLSTM) ---
class DeepfakeLSTM(nn.Module):
    """
    Bi-Directional LSTM for Temporal Deepfake Detection.
    Analyzes sequence of spatial/frequency features for frame-to-frame inconsistencies.
    
    BiLSTM captures both forward and backward temporal dependencies,
    making it ideal for spotting flickering or unnatural movements in videos.
    """
    def __init__(self, input_size=4096, hidden_size=128, num_layers=2):
        super(DeepfakeLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # BiLSTM output dim is hidden_size * 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        """Returns temporal consistency score."""
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :] 
        out = self.fc(last_time_step)
        return out
