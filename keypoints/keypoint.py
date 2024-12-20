from ultralytics import YOLO
import supervision as sv
import numpy as np
import pickle
import os

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer

class KeypointDetector:
    def __init__(self, path):
        self.model = YOLO(path)
        self.CONFIG = SoccerPitchConfiguration()

    def get_keypoints(self, frames, read=False, path=None):
        if read and path is not None and os.path.exists(path):
            with open(path,'rb') as f:
                keypoints = pickle.load(f)
            return keypoints
        
        keypoints = []

        for frame_num, frame in enumerate(frames):
            result = self.model(frame, conf=0.3)[0]
            key_points = sv.KeyPoints.from_ultralytics(result)
    
            filter = key_points.confidence[0] > 0.5
            frame_reference_points = key_points.xy[0][filter]
            frame_reference_key_points = sv.KeyPoints(
                xy=frame_reference_points[np.newaxis, ...])

            pitch_reference_points = np.array(self.CONFIG.vertices)[filter]

            transformer = ViewTransformer(
                source=pitch_reference_points,
                target=frame_reference_points
            )

            pitch_all_points = np.array(self.CONFIG.vertices)
            frame_all_points = transformer.transform_points(points=pitch_all_points)

            frame_all_key_points = sv.KeyPoints(xy=frame_all_points[np.newaxis, ...])

            keypoints.append([frame_all_key_points, frame_reference_key_points])

        if path is not None:
            with open(path,'wb') as f:
                pickle.dump(keypoints, f)
        
        return keypoints

    def draw_keypoints(self, frames, keypoints):
        output_frames = []

        edge_annotator = sv.EdgeAnnotator(
            color=sv.Color.from_hex('#00BFFF'),
            thickness=2, edges=self.CONFIG.edges)
        vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex('#FF1493'),
            radius=8)
        vertex_annotator_2 = sv.VertexAnnotator(
            color=sv.Color.from_hex('#00BFFF'),
            radius=8)
                
        for frame_num, frame in enumerate(frames):
            frame_all_key_points, frame_reference_key_points = keypoints[frame_num]

            annotated_frame = frame.copy()
            annotated_frame = edge_annotator.annotate(
                scene=annotated_frame,
                key_points=frame_all_key_points)
            annotated_frame = vertex_annotator_2.annotate(
                scene=annotated_frame,
                key_points=frame_all_key_points)
            annotated_frame = vertex_annotator.annotate(
                scene=annotated_frame,
                key_points=frame_reference_key_points)
            
            output_frames.append(annotated_frame)

        return output_frames
    
    def get_xy(self, tracks):
        player1_xy = {}
        player2_xy = {}
        referee_xy = {}
        ball_xy = {}

        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                player1_frame = []
                player2_frame = []
                referee_frame = []
                ball_frame = []

                for track_id, track_info in track.items():
                    if object == 'players':
                        if track_info['team'] == 0:
                            player1_frame.append(np.array(track_info['position']))
                        elif track_info['team'] == 1:
                            player2_frame.append(np.array(track_info['position']))
                    elif object == 'referees':
                        referee_frame.append(np.array(track_info['position']))
                    elif object == 'ball':
                        ball_frame.append(np.array(track_info['position']))

                if player1_frame:
                    player1_xy[frame_num] = np.array(player1_frame)

                if player2_frame:
                    player2_xy[frame_num] = np.array(player2_frame)

                if referee_frame:
                    referee_xy[frame_num] = np.array(referee_frame)

                if ball_frame:
                    ball_xy[frame_num] = np.array(ball_frame)

        return player1_xy, player2_xy, referee_xy, ball_xy
    
    def draw_2d_pitch(self, frames, player1_xy, player2_xy, referee_xy, ball_xy):
        for frame_num, frame in enumerate(frames):
            result = self.model(frame, conf=0.3)[0]
            key_points = sv.KeyPoints.from_ultralytics(result)
    
            filter = key_points.confidence[0] > 0.5
            frame_reference_points = key_points.xy[0][filter]
            pitch_reference_points = np.array(self.CONFIG.vertices)[filter]

            transformer = ViewTransformer(
                source=frame_reference_points,
                target=pitch_reference_points
            )

            pitch_ball_xy = transformer.transform_points(points=ball_xy.get(frame_num, np.array([])))
            pitch_player1_xy = transformer.transform_points(points=player1_xy.get(frame_num, np.array([])))
            pitch_player2_xy = transformer.transform_points(points=player2_xy.get(frame_num, np.array([])))
            pitch_referee_xy = transformer.transform_points(points=referee_xy.get(frame_num, np.array([])))

            annotated_frame = draw_pitch(self.CONFIG)

            annotated_frame = draw_points_on_pitch(
                                config=self.CONFIG,
                                xy=pitch_ball_xy,
                                face_color=sv.Color.WHITE,
                                edge_color=sv.Color.BLACK,
                                radius=10,
                                pitch=annotated_frame)
            
            annotated_frame = draw_points_on_pitch(
                                config=self.CONFIG,
                                xy=pitch_player1_xy,
                                face_color=sv.Color.from_hex('00BFFF'),
                                edge_color=sv.Color.BLACK,
                                radius=16,
                                pitch=annotated_frame)
            
            annotated_frame = draw_points_on_pitch(
                                config=self.CONFIG,
                                xy=pitch_player2_xy,
                                face_color=sv.Color.from_hex('FF1493'),
                                edge_color=sv.Color.BLACK,
                                radius=16,
                                pitch=annotated_frame)
            
            annotated_frame = draw_points_on_pitch(
                                config=self.CONFIG,
                                xy=pitch_referee_xy,
                                face_color=sv.Color.from_hex('FFD700'),
                                edge_color=sv.Color.BLACK,
                                radius=16,
                                pitch=annotated_frame)
            
            yield annotated_frame
