import torch
import torch.nn as nn

import numpy as np
from utils import frames_from_video_file


# Model definition
class HieraVideoClassifier(nn.Module):
    def __init__(self, num_classes):
        super(HieraVideoClassifier, self).__init__()
        self.model = torch.hub.load('facebookresearch/hiera', model='hiera_base_plus_16x224', pretrained=True)
        proj = self.model.head.projection
        self.model.head.projection = nn.Linear(proj.in_features, num_classes)

        # Freeze the pre-trained parameters
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.blocks[4:].parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.model(x)

        return x

def load_hiera_model(model_path: str) -> HieraVideoClassifier:
    """Load the model for inference"""
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = HieraVideoClassifier(num_classes=4).to(device)

    # Freeze parameters for inference
    for param in model.parameters():
        param.requires_grad = False

    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device)['model_state_dict'])
    return model

def get_hiera_video_frames(video_path: str, output_size:tuple[int,int]=(224, 224)) -> torch.FloatTensor:
    """Get video frames"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """Get video frames"""
    frames = frames_from_video_file(video_path, 16, output_size, 4)
    frames = torch.FloatTensor(frames)

    video_frames = frames.unsqueeze(0).to(device)

    return video_frames

def hiera_pred_to_label(prediction: torch.FloatTensor) -> str:
    """Convert prediction to label"""
    pred = torch.argmax(prediction, dim=1).cpu().numpy()

    classes = ['short pass', 'dribble', 'diving', 'throw']
    id2label = {i: label for i, label in enumerate(classes)}

    return id2label[pred[0]]