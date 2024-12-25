from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator
from keypoints import KeypointDetector

from sports.annotators.soccer import draw_pitch

import numpy as np
import cv2
import supervision as sv

def main():
    video_name = "08fd33_4"
    video_path = f'input_videos/{video_name}.mp4'
    frames = sv.get_video_frames_generator(video_path)
    first_frame = next(frames)
    frames = sv.get_video_frames_generator(video_path)

    # Initialise tracker
    tracker = Tracker('models/best.pt')
    tracks, team_possession = tracker.get_object_tracks(frames, read=True, path=f'stubs/track_stubs_{video_name}.pkl')

    # Initialise keypoints
    keypoint_detector = KeypointDetector('models/keypoints/best.pt')
    # keypoints = keypoint_detector.get_keypoints(frames, read=True, path=f'stubs/keypoint_stubs_{video_name}.pkl')

    # Get object positions
    tracker.add_position_to_tracks(tracks)

    # Camera movement estimator
    frames = sv.get_video_frames_generator(video_path)
    camera_movement_estimator = CameraMovementEstimator(first_frame)
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(frames,read=True,
                                                                              path=f'stubs/camera_stubs_{video_name}.pkl')
    camera_movement_estimator.adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    
    # View transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    frames = sv.get_video_frames_generator(video_path)

    # Draw keypoints
    player1_xy, player2_xy, referee_xy, ball_xy = keypoint_detector.get_xy(tracks)
    pitch_frames = keypoint_detector.draw_2d_pitch(frames, player1_xy, player2_xy, referee_xy, ball_xy)

    # Draw output
    frames = sv.get_video_frames_generator(video_path)
    output_frames = tracker.draw_annotations(frames, tracks, team_possession)

    # Draw camera movement
    output_frames = camera_movement_estimator.draw_camera_movement(output_frames, camera_movement_per_frame)

    # Draw speed and distance
    output_frames = speed_and_distance_estimator.draw_speed_and_distance(output_frames, tracks)

    first_pitch_frame = draw_pitch(keypoint_detector.CONFIG)

    save_video(f'output_videos/{video_name}_output.avi', output_frames, first_frame)
    save_video(f'output_videos/{video_name}_pitch.avi', pitch_frames, first_pitch_frame)

if __name__ == "__main__":
    main()