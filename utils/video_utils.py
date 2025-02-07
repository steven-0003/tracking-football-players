import cv2
from tqdm import tqdm
import supervision as sv

def read_video(path: str):
    capture = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frames.append(frame)
    
    return frames

def save_video(path: str, frames, w, h, num_frames) -> None:
    f = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path, f, 24, (w, h))
    for frame in tqdm(frames,desc=f"Saving video in {path}", total=num_frames):
        out.write(frame)
    out.release()