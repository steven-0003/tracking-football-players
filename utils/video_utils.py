import cv2
from tqdm import tqdm
import random
import numpy as np

def read_video(path: str) -> list[cv2.Mat]:
    """Reads a video file and returns its frames as a list of numpy arrays."""
    capture = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frames.append(frame)
    
    return frames

def save_video(path: str, frames, w, h, num_frames) -> None:
    """Saves a list of frames as a video file."""
    f = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path, f, 24, (w, h))
    for frame in tqdm(frames,desc=f"Saving video in {path}", total=num_frames):
        out.write(frame)
    out.release()

def format_frames(frame, output_size):
    """Format frames to tensor with specified size"""
    frame = cv2.resize(frame, output_size)
    frame = frame / 255.0
    return frame

def frames_from_video_file(video_path, n_frames, output_size=(224, 224), frame_step=4):
    """Extract frames from video file"""
    result = []
    src = cv2.VideoCapture(str(video_path))

    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)
    need_length = 1 + (n_frames - 1) * frame_step

    if need_length > video_length:
        start = 0
    else:
        max_start = video_length - need_length
        start = random.randint(0, max_start + 1)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    ret, frame = src.read()
    result.append(format_frames(frame, output_size))

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