import cv2
from tqdm import tqdm

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