from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import cv2
import pickle
import os
from tqdm import tqdm

import sys
sys.path.append('../')

from utils import get_bbox_center, get_bbox_width, get_foot_position, remove_short_tracks, player_tracks_by_frames, player_tracks_by_ids
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner

from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

class Tracker:
    def __init__(self, path, keypoint_path, fps) -> None:
        self.model = YOLO(path)
        self.keypoints = YOLO(keypoint_path)
        self.CONFIG = SoccerPitchConfiguration()
        self.tracker = sv.ByteTrack(lost_track_buffer=1000, track_activation_threshold=0.5, frame_rate=fps)
        self.team_assigner = TeamAssigner()
        self.player_assigner = PlayerBallAssigner()

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

    def interpolate_ball_positions(self, positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in positions]
        ball_positions_df = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        transformed_positions = [x.get(1,{}).get('position_transformed',[]) for x in positions]
        transformed_positions_df = pd.DataFrame(transformed_positions,columns=['x','y'])

        # Interpolate missing values
        ball_positions_df = ball_positions_df.interpolate()
        ball_positions_df = ball_positions_df.bfill()

        transformed_positions_df = transformed_positions_df.interpolate()
        transformed_positions_df = transformed_positions_df.bfill()

        ball_positions = [{1: {"bbox": x}} for x in ball_positions_df.to_numpy().tolist()]

        transformed_list = transformed_positions_df.to_numpy().tolist()
        for i in range(len(ball_positions)):
            ball_positions[i][1]["position_transformed"] = transformed_list[i]

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

        # source_target_keypoints = []
        last_position = None
        
        for frame_num, frame in enumerate(tqdm(frames, desc="Running KeyPoint & Player Detection Model", total=num_frames)):
            detection = self.model(frame, conf=0.1, verbose=False)
            keypoint_detection = self.keypoints(frame, verbose=False)[0]
            key_points = sv.KeyPoints.from_ultralytics(keypoint_detection)
            mask = (key_points.xy[0][:, 0] > 1) & (key_points.xy[0][:, 1] > 1)
            source = key_points.xy[0][mask].astype(np.float32)
            target = np.array(self.CONFIG.vertices)[mask].astype(np.float32)
            transform = False
            if source.shape[0] >= 4:
                transform = True
                transformer = ViewTransformer(
                    source=source,
                    target=target
                )

            cls_names = detection[0].names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection[0])

            # Convert goalkeeper to player
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "Goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["Player"]

            # Tracking
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            detections_with_classes = np.concatenate((detection_supervision.xyxy, detection_supervision.class_id.reshape(-1,1)), axis=1)
            detections_with_classes = np.round(detections_with_classes,2)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0]
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["Player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                    if transform:
                        position = get_foot_position(bbox)
                        transformed = transformer.transform_points(np.array([position]))
                        tracks["players"][frame_num][track_id]["position_transformed"] = transformed[0]
                    else:
                        tracks["players"][frame_num][track_id]["position_transformed"] = np.array([])

                if cls_id == cls_names_inv["Referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                    
                    if transform:
                        position = get_foot_position(bbox)
                        transformed = transformer.transform_points(np.array([position]))
                        tracks["referees"][frame_num][track_id]["position_transformed"] = transformed[0]
                    else:
                        tracks["referees"][frame_num][track_id]["position_transformed"] = np.array([])

            for frame_detection in detection_supervision:
                bbox = frame_detection[0]
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv["Ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

                    if transform:
                        position = get_bbox_center(bbox)
                        transformed = transformer.transform_points(np.array([position]))
                        tracks["ball"][frame_num][1]["position_transformed"] = transformed[0]
                    else:
                        tracks["ball"][frame_num][1]["position_transformed"] = np.array([])
            
            # Team classification
            if frame_num == 0:
                self.team_assigner.assign_team_colour(frame, tracks["players"][0])
            
            for player_id, track in tracks["players"][frame_num].items():
                team = self.team_assigner.get_player_team(frame, track['bbox'], player_id)
                tracks['players'][frame_num][player_id]['team'] = team
                tracks['players'][frame_num][player_id]['team_colour'] = self.team_assigner.team_colours[team]

        # Interpolate ball positions
        tracks["ball"] = self.interpolate_ball_positions(tracks["ball"])

        last_position = None
        for f, ball_track in enumerate(tracks["ball"]):
            position = get_bbox_center(ball_track[1].get("bbox", []))

            if last_position is None:
                last_position = np.array(position)
            else:
                distance = np.linalg.norm(np.array(position) - last_position)
                if distance > 100:
                    tracks["ball"][f][1] = {}
                last_position = position

        tracks["ball"] = self.interpolate_ball_positions(tracks["ball"])

        # Remove short tracks
        player_tracks = player_tracks_by_ids(tracks)
        player_tracks = remove_short_tracks(player_tracks, 20)
        tracks["players"] = player_tracks_by_frames(player_tracks)

        if path is not None:
            with open(path,'wb') as f:
                pickle.dump(tracks, f)
        
        return tracks

    def detect_frames(self, frames):
        for frame in frames:
            yield self.model(frame,conf=0.1)

    def draw_annotations(self, frames, tracks, team_possession, pitch_frames):
        for frame_num, f in enumerate(frames):
            frame = f.copy()
            pitch_frame = next(pitch_frames)
            x_offset = frame.shape[1]//2 - pitch_frame.shape[1]//2
            y_offset = frame.shape[0] - pitch_frame.shape[0] - 50
            y1, y2 = y_offset, y_offset + pitch_frame.shape[0]
            x1, x2 = x_offset, x_offset + pitch_frame.shape[1]
            frame[y1:y2,x1:x2] = (frame[y1:y2,x1:x2] * 0.5) + (pitch_frame * 0.5)

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
