import torch
import torch.nn as nn

import numpy as np
from utils import frames_from_video_file

# Model definition
class SlowfastVideoClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SlowfastVideoClassifier, self).__init__()
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r101', pretrained=True)
        proj = self.model.blocks[-1].proj
        self.model.blocks[-1].proj = nn.Linear(proj.in_features, num_classes)

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.model(x)

        return x

def load_slowfast_model(model_path: str) -> SlowfastVideoClassifier:
    """Load the model for inference"""
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SlowfastVideoClassifier(num_classes=4).to(device)

    # Freeze parameters for inference
    for param in model.parameters():
        param.requires_grad = False

    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device)['model_state_dict'])
    return model

def get_slowfast_video_frames(video_path: str, output_size:tuple[int,int]=(224, 224)) -> list[torch.FloatTensor]:
    """Get video frames for slow and fast streams"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """Get video frames"""
    frames = frames_from_video_file(video_path, 32, output_size, 1)
    frames = torch.FloatTensor(frames)

    fast_frames = frames.clone()

    slow_frames = torch.index_select(frames, 1, torch.linspace(0, 31, 8).long())

    video_frames = [slow_frames.unsqueeze(0).to(device), fast_frames.unsqueeze(0).to(device)]
    return video_frames

def slowfast_pred_to_label(prediction: torch.FloatTensor) -> str:
    """Convert prediction to label"""
    pred = torch.argmax(prediction, dim=1).cpu().numpy()

    classes = ['short pass', 'dribble', 'diving', 'throw']
    id2label = {i: label for i, label in enumerate(classes)}

    return id2label[pred[0]]
