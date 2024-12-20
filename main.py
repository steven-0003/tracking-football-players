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
    #frames = read_video(f'input_videos/{video_name}.mp4')
    frames = sv.get_video_frames_generator(f'input_videos/{video_name}.mp4')
    first_frame = next(frames)
    frames = sv.get_video_frames_generator(f'input_videos/{video_name}.mp4')
    num_frames = len(list(frames))
    frames = sv.get_video_frames_generator(f'input_videos/{video_name}.mp4')

    # Initialise tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(frames, read=True, path=f'stubs/track_stubs_{video_name}.pkl')

    # Interpolate ball positions 
    tracks["ball"] = tracker.interpolate_ball_positions(tracks['ball'])

    # Initialise keypoints
    keypoint_detector = KeypointDetector('models/keypoints/best.pt')
    # keypoints = keypoint_detector.get_keypoints(frames, read=True, path=f'stubs/keypoint_stubs_{video_name}.pkl')

    # Get object positions
    tracker.add_position_to_tracks(tracks)

    # Camera movement estimator
    frames = sv.get_video_frames_generator(f'input_videos/{video_name}.mp4')
    camera_movement_estimator = CameraMovementEstimator(first_frame)
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(frames,num_frames,read=True,
                                                                              path=f'stubs/camera_stubs_{video_name}.pkl')
    camera_movement_estimator.adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    
    # View transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_colour(first_frame,tracks['players'][0])

    player_assigner = PlayerBallAssigner()
    team_possession = []

    f = sv.get_video_frames_generator(f'input_videos/{video_name}.mp4')
    frames = list(f)

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_colour'] = team_assigner.team_colours[team]

        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_possession.append(tracks['players'][frame_num][assigned_player]['team'])
        elif len(team_possession)>0:
            team_possession.append(team_possession[-1])
           
    del f
    del frames
    team_possession = np.array(team_possession)

    frames = sv.get_video_frames_generator(f'input_videos/{video_name}.mp4')

    # Draw keypoints
    player1_xy, player2_xy, referee_xy, ball_xy = keypoint_detector.get_xy(tracks)
    pitch_frames = keypoint_detector.draw_2d_pitch(frames, player1_xy, player2_xy, referee_xy, ball_xy)

    # Draw output
    frames = sv.get_video_frames_generator(f'input_videos/{video_name}.mp4')
    output_frames = tracker.draw_annotations(frames, tracks, team_possession)

    # Draw camera movement
    output_frames = camera_movement_estimator.draw_camera_movement(output_frames, camera_movement_per_frame)

    # Draw speed and distance
    speed_and_distance_estimator.draw_speed_and_distance(output_frames, tracks)

    first_pitch_frame = draw_pitch(keypoint_detector.CONFIG)

    save_video(f'output_videos/{video_name}_output.avi', output_frames, first_frame)
    save_video(f'output_videos/{video_name}_pitch.avi', pitch_frames, first_pitch_frame)

if __name__ == "__main__":
    main()