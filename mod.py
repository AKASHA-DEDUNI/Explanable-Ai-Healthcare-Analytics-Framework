# model.py
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class MultiModalPneumoniaModel(nn.Module):
    def __init__(self, metadata_input_dim=2, text_embedding_dim=384):
        super().__init__()
        # Load ResNet-50 with pretrained weights
        self.cnn = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()  # Remove the final fully connected layer

        # Metadata branch
        self.meta_fc = nn.Sequential(
            nn.Linear(metadata_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Text embedding branch
        self.text_fc = nn.Sequential(
            nn.Linear(text_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Classifier combining all features
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 32 + 64, 128),  # ResNet-50 outputs 2048-dim features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, image, metadata, text_emb):
        img_feat = self.cnn(image)        # 2048-dim feature from ResNet-50
        meta_feat = self.meta_fc(metadata)
        text_feat = self.text_fc(text_emb)
        combined = torch.cat([img_feat, meta_feat, text_feat], dim=1)
        return self.classifier(combined)
