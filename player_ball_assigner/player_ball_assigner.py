import numpy as np
import sys

sys.path.append('../')
from utils import get_bbox_center, measure_distance

class PlayerBallAssigner:
    def __init__(self) -> None:
        self.max_player_ball_distance = 70
    
    def assign_ball_to_player(self, players: dict, ball_bbox: list) -> int:
        """Assigns the ball to the closest player within a certain distance.
        If no player is within the distance, returns -1.

        Args:
            players (dict): Player tracks with their bounding boxes.
            ball_bbox (list): Bounding box of the ball.

        Returns:
            int: The ID of the assigned player or -1 if no player is close enough.
        """
        if len(ball_bbox) == 0:
            return -1
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
    
    def get_team_possession(self, num_frames: int, tracks: dict) -> np.ndarray:
        """Calculates the team possession based on ball assignment to players.
        For each frame, if a player has the ball, their team is recorded. 
        If no player has the ball, the last known team possession is used.

        Args:
            num_frames (int): Total number of frames in the video.
            tracks (dict): Player, referee, and ball tracks.

        Returns:
            np.ndarray: Array indicating the team possession for each frame.
        """
        team_possession = []

        for frame_num in range(num_frames):

            ball_bbox = tracks['ball'][frame_num].get(1, {}).get('bbox', [])
            assigned_player = self.assign_ball_to_player(tracks["players"][frame_num], ball_bbox)
            if assigned_player != -1:
                tracks['players'][frame_num][assigned_player]['has_ball'] = True
                team_possession.append(tracks['players'][frame_num][assigned_player]['team'])
            elif len(team_possession)>0:
                team_possession.append(team_possession[-1])

        return np.array(team_possession)