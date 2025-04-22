import torch
import torch.nn as nn

import numpy as np
import cv2
import random

# Model definition
class VideoClassifier(nn.Module):
    def __init__(self, num_classes):
        super(VideoClassifier, self).__init__()
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r101', pretrained=True)
        proj = self.model.blocks[-1].proj
        self.model.blocks[-1].proj = nn.Linear(proj.in_features, num_classes)

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.model(x)

        return x

def load_model(model_path: str) -> VideoClassifier:
    """Load the model for inference"""
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VideoClassifier(num_classes=4).to(device)

    # Freeze parameters for inference
    for param in model.parameters():
        param.requires_grad = False

    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device)['model_state_dict'])
    return model

def format_frames(frame: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    """Format frames to tensor with specified size"""
    frame = cv2.resize(frame, output_size)
    frame = frame / 255.0
    return frame

def frames_from_video_file(video_path: str, n_frames: int, output_size: tuple[int, int]=(224, 224), 
                           frame_step:int=4) -> np.ndarray:
    """Extract frames from video file"""
    random.seed(42)

    result = []
    src = cv2.VideoCapture(str(video_path))

    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)
    need_length = 1 + (n_frames - 1) * frame_step

    # Select start frame
    if need_length > video_length:
        start = 0
    else:
        max_start = video_length - need_length
        start = random.randint(0, max_start + 1)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    ret, frame = src.read()
    result.append(format_frames(frame, output_size))

    # Append frames
    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret:
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))

    src.release()

    result = np.array(result)
    result = np.transpose(result, (3, 0, 1, 2))  # (T, C, H, W) format
    return result

def get_video_frames(video_path: str, output_size:tuple[int,int]=(224, 224)) -> list[torch.FloatTensor]:
    """Get video frames for slow and fast streams"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """Get video frames"""
    frames = frames_from_video_file(video_path, 32, output_size, 1)
    frames = torch.FloatTensor(frames)

    fast_frames = frames.clone()

    slow_frames = torch.index_select(frames, 1, torch.linspace(0, 31, 8).long())

    video_frames = [slow_frames.unsqueeze(0).to(device), fast_frames.unsqueeze(0).to(device)]
    return video_frames

def pred_to_label(prediction: torch.FloatTensor) -> str:
    """Convert prediction to label"""
    pred = torch.argmax(prediction, dim=1).cpu().numpy()

    classes = ['short pass', 'dribble', 'diving', 'throw']
    id2label = {i: label for i, label in enumerate(classes)}

    return id2label[pred[0]]

