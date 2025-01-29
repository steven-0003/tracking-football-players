from ultralytics import YOLO
import numpy as np
import pandas as pd
import supervision as sv
from collections import deque
from tqdm import tqdm

from player_ball_assigner import PlayerBallAssigner

class BallTracker:
    def __init__(self, path, buffer_size=20):
        self.model = YOLO(path)
        self.buffer = deque(maxlen=buffer_size)
        self.player_assigner = PlayerBallAssigner()

    def callback(self, patch: np.ndarray) -> sv.Detections: 
        result = self.model(patch, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)
    
    def update(self, detections: sv.Detections) -> sv.Detections:
        xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        self.buffer.append(xy)

        if len(detections) == 0:
            return detections

        centroid = np.mean(np.concatenate(self.buffer), axis=0)
        distances = np.linalg.norm(xy - centroid, axis=1)
        index = np.argmin(distances)
        return detections[[index]]
    
    def get_ball_tracks(self, frames, num_frames, tracks):
        slicer = sv.InferenceSlicer(
            callback=self.callback,
            overlap_filter_strategy=sv.OverlapFilter.NONE,
            slice_wh=(640, 640)
        )
        
        for frame_num, frame in enumerate(tqdm(frames, desc="Running Ball Detection Model", total=num_frames)):
            detection = slicer(frame).with_nms(threshold=0.1)
            detection = self.update(detection)
            if detection.xyxy.size > 0:
                bbox = detection.xyxy[0]
                tracks["ball"][frame_num][1] = {"bbox": bbox}

        tracks["ball"] = self.interpolate_ball_positions(tracks["ball"])
    
    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        ball_positions_df = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        ball_positions_df = ball_positions_df.interpolate()
        ball_positions_df = ball_positions_df.bfill()

        ball_positions = [{1: {"bbox": x}} for x in ball_positions_df.to_numpy().tolist()]

        return ball_positions
