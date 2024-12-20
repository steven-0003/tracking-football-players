import pickle
import cv2
import numpy as np
import os
import sys
sys.path.append('../')
from utils import measure_distance

class CameraMovementEstimator:
    def __init__(self, frame) -> None:
        self.min_distance = 5
        self.first_frame = frame

        grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(grayscale)
        mask_features[:,0:20] = 1
        mask_features[:,900:1050] = 1

        self.features = {
            "maxCorners": 100,
            "qualityLevel": 0.3,
            "minDistance": 3,
            "blockSize": 7,
            "mask": mask_features
        }

        self.lk_params = {
            "winSize": (15,15),
            "maxLevel": 2,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)
        }

    def adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0]-camera_movement[0], position[1]-camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted

    def get_camera_movement(self, frames, num_frames, read=False, path=None):
        if read and path is not None and os.path.exists(path):
            with open(path,'rb') as f:
                camera_movement = pickle.load(f)
            return camera_movement

        camera_movement = [[0,0]] * num_frames
        grayscale = cv2.cvtColor(self.first_frame,cv2.COLOR_BGR2GRAY)
        features = cv2.goodFeaturesToTrack(grayscale, **self.features)

        for frame_num, frame in enumerate(frames):
            frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(grayscale, frame_gray, features, None, **self.lk_params)

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            for i, (new, old) in enumerate(zip(new_features, features)):
                new_feature_point = new.ravel()
                old_feature_point = old.ravel()

                distance = measure_distance(new_feature_point, old_feature_point)

                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x = new_feature_point[0] - old_feature_point[0]
                    camera_movement_y = new_feature_point[1] - old_feature_point[1]

            if max_distance > self.min_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                features = cv2.goodFeaturesToTrack(frame_gray,**self.features)

            grayscale = frame_gray.copy()

        if path is not None:
            with open(path,'wb') as f:
                pickle.dump(camera_movement, f)
        
        return camera_movement


    def draw_camera_movement(self, frames, camera_movement_per_frame):
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()

            cv2.rectangle(overlay, (0,0), (500,100), (255,255,255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

            dx, dy = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame, f"Camera movement X: {dx:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 3)
            frame = cv2.putText(frame, f"Camera movement Y: {dy:.2f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 3)

            yield frame
    