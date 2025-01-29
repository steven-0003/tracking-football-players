import numpy as np
import sys

sys.path.append('../')
from utils import get_bbox_center, measure_distance

class PlayerBallAssigner:
    def __init__(self) -> None:
        self.max_player_ball_distance = 70
    
    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_bbox_center(ball_bbox)

        min_distance = 9999999
        assigned_player = -1
        
        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = measure_distance((player_bbox[0],player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2],player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance and distance < min_distance:
                min_distance = distance
                assigned_player = player_id
        
        return assigned_player
    
    def get_team_possession(self, num_frames, tracks):
        team_possession = []

        for frame_num in range(num_frames):
            ball_bbox = tracks['ball'][frame_num][1]['bbox']
            assigned_player = self.assign_ball_to_player(tracks["players"][frame_num], ball_bbox)
            if assigned_player != -1:
                tracks['players'][frame_num][assigned_player]['has_ball'] = True
                team_possession.append(tracks['players'][frame_num][assigned_player]['team'])
            elif len(team_possession)>0:
                team_possession.append(team_possession[-1])

        return np.array(team_possession)