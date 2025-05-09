from utils import save_video
from trackers import Tracker
from player_ball_assigner import PlayerBallAssigner
from keypoints import KeypointDetector
from action_recognition import load_model, get_video_frames, pred_to_label, VideoClassifier
from heatmaps import generate_heatmaps

import numpy as np
from pathlib import Path
import os
import supervision as sv
import torch

import tkinter as tk
from tkinter.messagebox import showinfo
import tkinter.filedialog as fd

def main(filename):
    if not filename:
        showinfo(title="Error", message="No file selected.")
        return
    
    if not os.path.exists(filename):
        showinfo(title="Error", message="File does not exist.")
        return

    video_path = filename
    video_name = Path(video_path).stem

    Path('stubs').mkdir(parents=True, exist_ok=True)

    video_info = sv.VideoInfo.from_video_path(video_path)
    num_frames = video_info.total_frames
    w, h = video_info.resolution_wh
    fps = video_info.fps

    frames = sv.get_video_frames_generator(video_path)

    # Initialise tracker
    tracker = Tracker('models/best.pt', 'models/keypoints/best.pt', fps)
    tracks = tracker.get_object_tracks(frames, num_frames, read=True, path=f'stubs/track_stubs_{video_name}.pkl')

    player_assigner = PlayerBallAssigner()
    team_possession = player_assigner.get_team_possession(num_frames, tracks)

    frames = sv.get_video_frames_generator(video_path)

    # Draw keypoints
    keypoint_detector = KeypointDetector('models/keypoints/best.pt')
    pitch_frames = keypoint_detector.draw_2d_pitch(tracks, num_frames)
    
    # Draw output
    frames = sv.get_video_frames_generator(video_path)
    output_frames = tracker.draw_annotations(frames, tracks, team_possession, pitch_frames)

    Path(f'output/{video_name}').mkdir(parents=True, exist_ok=True)
    save_video(f'output/{video_name}/output.avi', output_frames, w, h, num_frames)

    generate_heatmaps(video_name, tracks)

    showinfo(title="Complete", message=f"Analysis complete. Video output and heatmaps saved to output/{video_name}")

def action_recognition(filename):
    if not filename:
        showinfo(title="Error", message="No file selected.")
        return
    
    if not os.path.exists(filename):
        showinfo(title="Error", message="File does not exist.")
        return

    model = load_model('models/action/best.pt')
    with torch.no_grad():
        model.eval()
        slowfast_frames  = get_video_frames(filename)
        predicted = model(slowfast_frames)
        action = pred_to_label(predicted)
    showinfo(title="Action Recognition", message=f"Predicted action: {action}")


def select_file():
    filetypes = (
        ('Video files', '*.mp4'),
        ('All files', '*.*')
    )

    filename = fd.askopenfilename(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes)
    
    showinfo(title="Selected File", message=f"Selected file: {filename}")

    for widget in root.winfo_children():
        if widget.cget('text') != "Open Video File":
            widget.pack_forget()
    
    run_button = tk.Button(root, text="Run Analysis", command=lambda: main(filename))
    run_button.pack(pady=20)

    action_recognition_button = tk.Button(root, text="Run Action Recognition", command=lambda: action_recognition(filename))
    action_recognition_button.pack(pady=20)
    

if __name__ == "__main__":
    root = tk.Tk()

    open_button = tk.Button(root, text="Open Video File", command=select_file)
    open_button.pack(pady=20)

    root.mainloop()