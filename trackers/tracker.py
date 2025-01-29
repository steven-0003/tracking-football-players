from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import cv2
import pickle
import os
from tqdm import tqdm
from sklearn.ensemble import IsolationForest

import sys
sys.path.append('../')

from utils import get_bbox_center, get_bbox_width, get_foot_position
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner

class Tracker:
    def __init__(self, path) -> None:
        self.model = YOLO(path)
        self.tracker = sv.ByteTrack(lost_track_buffer=1000, track_activation_threshold=0.5, frame_rate=24)
        self.team_assigner = TeamAssigner()
        self.player_assigner = PlayerBallAssigner()
        self.anomaly_model = IsolationForest(n_estimators=100, max_samples='auto', contamination='auto')

    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]
                    if object == 'ball':
                        position = get_bbox_center(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        ball_positions_df = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        ball_positions_df = ball_positions_df.interpolate()
        ball_positions_df = ball_positions_df.bfill()

        ball_positions = [{1: {"bbox": x}} for x in ball_positions_df.to_numpy().tolist()]

        return ball_positions

    def get_object_tracks(self, frames, num_frames, read=False, path=None) -> dict:
        if read and path is not None and os.path.exists(path):
            with open(path,'rb') as f:
                tracks = pickle.load(f)
            
            return tracks

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, frame in enumerate(tqdm(frames, desc="Running Player Detection Model", total=num_frames)):
            detection = self.model(frame, conf=0.1, verbose=False)

            cls_names = detection[0].names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection[0])

            # Convert goalkeeper to player
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "Goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["Player"]

            # Tracking
            detections_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detections_with_tracks:
                bbox = frame_detection[0]
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["Player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv["Referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0]
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv["Ball"]:
                    if tracks["ball"][frame_num]:
                        if frame_num >=100:
                            past_positions = tracks["ball"][frame_num-100:frame_num]
                        else:
                            past_positions = tracks['ball'][0:frame_num]
                        ball_positions = [x.get(1,{}).get('bbox',[]) for x in past_positions]
                        ball_positions_df = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])
                        ball_positions_df.dropna(inplace=True)
                        X = ball_positions_df[['x1','y1','x2','y2']].values
                        weights = np.arange(0.1, 1, 0.9/len(X))
                        self.anomaly_model.fit(X, sample_weight=weights)
                        old_bbox = tracks['ball'][frame_num][1]['bbox']
                        old_score = self.anomaly_model.decision_function(old_bbox.reshape(1,-1))
                        new_score = self.anomaly_model.decision_function(bbox.reshape(1,-1))

                        if new_score > old_score:
                            tracks["ball"][frame_num][1]['bbox'] = bbox

                    else: 
                        tracks["ball"][frame_num][1] = {"bbox": bbox}
            
            # Team classification
            if frame_num == 0:
                self.team_assigner.assign_team_colour(frame, tracks["players"][0])
            
            for player_id, track in tracks["players"][frame_num].items():
                team = self.team_assigner.get_player_team(frame, track['bbox'], player_id)
                tracks['players'][frame_num][player_id]['team'] = team
                tracks['players'][frame_num][player_id]['team_colour'] = self.team_assigner.team_colours[team]

        # Interpolate ball positions
        tracks["ball"] = self.interpolate_ball_positions(tracks["ball"])

        if path is not None:
            with open(path,'wb') as f:
                pickle.dump(tracks, f)
        
        return tracks

    def detect_frames(self, frames):
        for frame in frames:
            yield self.model(frame,conf=0.1)

    def draw_annotations(self, frames, tracks, team_possession):
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ref_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # Draw players
            for track_id, player in player_dict.items():
                colour = player.get("team_colour", (0,0,255))
                frame = self.draw_elipse(frame, player["bbox"],colour,track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], (0,0,255))

            # Draw referees
            for _, referee in ref_dict.items():
                frame = self.draw_elipse(frame, referee["bbox"],(0,255,255))

            # Draw ball
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"],(0,255,0))
                
            # Draw team possession
            frame = self.draw_team_possession(frame, frame_num, team_possession)
            
            yield frame
        
    
    def draw_team_possession(self, frame, frame_num, team_possession):
        overlay = frame.copy()

        cv2.rectangle(overlay, (1350,850), (1900,970), (255,255,255), cv2.FILLED)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

        team_possession_til_frame = team_possession[:frame_num+1]
        team_1_num_frames = team_possession_til_frame[team_possession_til_frame==0].shape[0]
        team_2_num_frames = team_possession_til_frame[team_possession_til_frame==1].shape[0]

        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Possession: {team_1*100:.2f}%", (1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 3)
        cv2.putText(frame, f"Team 2 Possession: {team_2*100:.2f}%", (1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 3)

        return frame
        
    
    def draw_triangle(self, frame, bbox, colour):
        y = int(bbox[1])
        x,_ = get_bbox_center(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20]
        ])
        cv2.drawContours(frame, [triangle_points], 0, colour, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)

        return frame

    def draw_elipse(self, frame, bbox, colour, track_id=None):
        y2 = int(bbox[3])

        xc, _ = get_bbox_center(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(frame, center=(xc, y2), axes=(int(width), int(0.35*width)), angle=0.0, startAngle=-45, endAngle=235,
                    color=colour, thickness=2, lineType=cv2.LINE_4)
        
        r_width = 40
        r_height = 20
        r_x1 = xc - r_width//2
        r_x2 = xc + r_width//2
        r_y1 = (y2-r_height//2) + 15
        r_y2 = (y2+r_height//2) + 15

        if track_id is not None:
            cv2.rectangle(frame, (int(r_x1), int(r_y1)), (int(r_x2), int(r_y2)), colour, cv2.FILLED)

            t_x1 = r_x1 + 12
            if(track_id>99):
                t_x1 -= 10
            
            cv2.putText(frame, f"{track_id}", (int(t_x1), int(r_y1+15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        
        return frame
