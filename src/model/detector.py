import torch
import torch.nn as nn
import torchvision
from src.config import config

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Visual pathway: Using a ResNet34 backbone modified for multiple frames
        self.visual_encoder = torchvision.models.resnet34(
            weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1
        )
        self.visual_encoder.conv1 = nn.Conv2d(
            3 * config.FRAME_COUNT, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.visual_encoder.fc = nn.Identity()

        # Lip sync analysis pathway
        self.lip_encoder = nn.Sequential(
            nn.Conv2d(3 * config.FRAME_COUNT, 128, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # Temporal modeling via LSTM
        self.lstm = nn.LSTM(
            input_size=config.VISUAL_FEATURES + 256,
            hidden_size=config.LSTM_UNITS,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # Audio pathway: if audio is available, encode it; otherwise use a zero tensor
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(40, 256, 5),
            nn.ReLU(),
            nn.Conv1d(256, config.AUDIO_FEATURES, 5),
            nn.AdaptiveAvgPool1d(1)
        )

        # Classifier: combines visual and audio features
        self.classifier = nn.Sequential(
            nn.Linear(2 * config.LSTM_UNITS + config.AUDIO_FEATURES, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, video, audio=None):
        B = video['full'].shape[0]
        full_video = video['full']
        # full_video shape: [B, T, C, H, W]
        B, T, C, H, W = full_video.shape
        vis_features = self.visual_encoder(full_video.reshape(B, T * C, H, W))

        lip_video = video['lip']
        lip_features = self.lip_encoder(lip_video.reshape(B, T * C, 64, 64))

        # Combine visual features and process them with LSTM for temporal modeling
        combined_vis = torch.cat([vis_features, lip_features], dim=1)
        lstm_out, _ = self.lstm(combined_vis.unsqueeze(1))
        lstm_features = lstm_out.squeeze(1)

        # Audio branch: if audio tensor is nonzero, process it; otherwise, use zeros.
        if audio is not None and torch.sum(audio) != 0:
            audio_features = self.audio_encoder(audio).squeeze(-1)
        else:
            audio_features = torch.zeros(B, config.AUDIO_FEATURES, device=full_video.device)

        combined = torch.cat([lstm_features, audio_features], dim=1)
        return self.classifier(combined)
