from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
from typing import Generator

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.configs.soccer import SoccerPitchConfiguration

class KeypointDetector:
    def __init__(self, path):
        self.model = YOLO(path)
        self.CONFIG = SoccerPitchConfiguration()
    
    def draw_2d_pitch(self, tracks: dict, num_frames: int) -> Generator[np.ndarray, None, None]:
        """
        Draws the 2D pitch with player and referee keypoints on it.

        Args:
            tracks (dict): Player, referee, and ball tracks.
            num_frames (int): Total number of frames in the video.

        Yields:
            Generator[np.ndarray]: Annotated 2D pitch frames with player, referee, and ball positions.
        """
        for frame_num in range(num_frames):
            annotated_frame = draw_pitch(self.CONFIG)

            for player_track in tracks['players'][frame_num].values():
                if player_track['team']==0:
                    if player_track['position_transformed'].size>0:
                        annotated_frame = draw_points_on_pitch(
                            config=self.CONFIG,
                            xy=player_track['position_transformed'].reshape(1,-1),
                            face_color=sv.Color.from_hex('00BFFF'),
                            edge_color=sv.Color.BLACK,
                            radius=16,
                            pitch=annotated_frame
                        )

                if player_track['team']==1:
                    if player_track['position_transformed'].size>0:
                        annotated_frame = draw_points_on_pitch(
                            config=self.CONFIG,
                            xy=player_track['position_transformed'].reshape(1,-1),
                            face_color=sv.Color.from_hex('FF1493'),
                            edge_color=sv.Color.BLACK,
                            radius=16,
                            pitch=annotated_frame
                    )
            for referee_track in tracks['referees'][frame_num].values():
                if referee_track['position_transformed'].size>0:
                    annotated_frame = draw_points_on_pitch(
                        config=self.CONFIG,
                        xy=referee_track['position_transformed'].reshape(1,-1),
                        face_color=sv.Color.from_hex('FFD700'),
                        edge_color=sv.Color.BLACK,
                        radius=16,
                        pitch=annotated_frame
                    )

            ball_tracks = tracks["ball"]

            if ball_tracks:
                ball_pos = ball_tracks[frame_num].get(1, {}).get('position_transformed', [])
            else:
                ball_pos = []
            if len(ball_pos)>0:
                annotated_frame = draw_points_on_pitch(
                    config=self.CONFIG,
                    xy=np.array(ball_pos).reshape(1,-1),
                    face_color=sv.Color.WHITE,
                    edge_color=sv.Color.BLACK,
                    radius=10,
                    pitch=annotated_frame
                )
            final = cv2.resize(annotated_frame,(390,240),interpolation=cv2.INTER_LINEAR)
            yield final
            