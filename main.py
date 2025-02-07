from utils import read_video, save_video
from trackers import Tracker, BallTracker
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
    video_name = "LevBor-1stHalf"
    video_path = f'input_videos/{video_name}.mp4'

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

    # Speed and distance estimator
    # speed_and_distance_estimator = SpeedAndDistanceEstimator()
    # speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    frames = sv.get_video_frames_generator(video_path)

    # Draw keypoints
    keypoint_detector = KeypointDetector('models/keypoints/best.pt')
    pitch_frames = keypoint_detector.draw_2d_pitch2(tracks, num_frames)
    
    # Draw output
    frames = sv.get_video_frames_generator(video_path)
    output_frames = tracker.draw_annotations(frames, tracks, team_possession)

    # Draw speed and distance
    # output_frames = speed_and_distance_estimator.draw_speed_and_distance(output_frames, tracks)

    pitch_frame = draw_pitch(keypoint_detector.CONFIG)

    save_video(f'output_videos/{video_name}_output.avi', output_frames, w, h, num_frames)
    save_video(f'output_videos/{video_name}_pitch.avi', pitch_frames, pitch_frame.shape[1], pitch_frame.shape[0], num_frames)

if __name__ == "__main__":
    main()