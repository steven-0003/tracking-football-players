import os
import pickle
import pandas as pd
import numpy as np
from collections import Counter

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer

def get_tracks(video_name):
    path = f'stubs/track_stubs_{video_name}.pkl'
    if not os.path.exists(path):
        raise(Exception(f"No tracks found for {video_name}"))
    
    with open(path, 'rb') as f:
        tracks = pickle.load(f)

        return tracks

def count_ids(tracks):
    ids = set()

    for frame_num, player_track in enumerate(tracks["players"]):
        frame_ids = set(player_track.keys())
        diff = frame_ids.difference(ids)
        ids.update(frame_ids)

        if len(diff)>0:
            print(f"New ids for frame {frame_num}: {diff}")

    print(f"Number of player ids: {len(ids)}")

def player_tracks_by_ids(tracks):
    player_tracks_by_ids = {}

    for frame_num, player_tracks in enumerate(tracks["players"]):
        for id, player_track in player_tracks.items():
            if id not in player_tracks_by_ids.keys():
                player_tracks_by_ids[id] = {}

            player_tracks_by_ids[id][frame_num] = player_track

    return player_tracks_by_ids

def remove_short_tracks(player_tracks: dict, threshold: int) -> dict:
    return {player_id: player_track for player_id, player_track in player_tracks.items() if len(player_track)>threshold}

def player_tracks_by_frames(player_tracks: dict) -> dict:
    tracks_by_frames = {}

    for id, player_track in player_tracks.items():
        for frame_num, track in player_track.items():
            if frame_num not in tracks_by_frames:
                tracks_by_frames[frame_num] = {}
            
            tracks_by_frames[frame_num][id] = track
    
    return tracks_by_frames

def interpolate_player_tracks(player_tracks: dict, cluster_centers) -> dict:
    for player_id in player_tracks.keys():
        player = player_tracks[player_id].copy()

        teams = [player.get(frame,{}).get('team', -1) for frame in player.keys()]
        c = Counter(teams)
        team = c.most_common(1)[0][0]
        team_colour = cluster_centers[team]

        start_frame = min(player.keys())
        end_frame = max(player.keys()) + 1

        positions = [player.get(frame,{}).get('bbox',[]) for frame in range(start_frame, end_frame)]
        transformed = [player.get(frame,{}).get('position_transformed',[]) for frame in range(start_frame, end_frame)]
        positions_df = pd.DataFrame(positions, columns=['x1','y1','x2','y2'])
        transformed_df = pd.DataFrame(transformed, columns=['x','y'])

        # imp = SimpleImputer(strategy='mean')
        # imp = IterativeImputer(max_iter=10, random_state=42)

        # new_positions = imp.fit_transform(positions_df)
        # new_transformed = imp.fit_transform(transformed_df)
        
        positions_df = positions_df.interpolate()
        positions_df = positions_df.bfill()

        transformed_df = transformed_df.interpolate()
        transformed_df = transformed_df.bfill()

        interpolated_postions = positions_df.to_numpy().tolist()
        interpolated_transformed = transformed_df.to_numpy().tolist()

        for i, (pos, transform) in enumerate(zip(interpolated_postions, interpolated_transformed)):
            player[i] = {"bbox": pos, "position_transformed": transform, "team": team, "team_colour": team_colour}


def connect_tracks(player_tracks: dict) -> dict:
    lost_ids = []
    all_ids = []

    for frame_num, player_track in player_tracks.items():
        current_ids = []
        
        for id, track in player_track.items():
            current_ids.append(id)
            lost_id_index = id_in_arr(lost_ids,id)
            if lost_id_index != -1:
                track = lost_ids.pop(lost_id_index)
                track[id] = frame_num
                all_ids.append(track)
            else:
                cur_id_index = id_in_arr(all_ids,id)
                if cur_id_index != -1:
                    all_ids[cur_id_index][id] = frame_num
                else:
                    all_ids.append({id:frame_num})
        print(current_ids)

def id_in_arr(ids: list[dict], id: int) -> int:
    for i, id_track in enumerate(ids):
        if id in id_track.keys():
            return i
    
    return -1

if __name__ == "__main__":
    video_name = "08fd33_4"
    tracks = get_tracks(video_name)
    player_tracks = player_tracks_by_ids(tracks)
    long_player_tracks = remove_short_tracks(player_tracks, 10)
    player = long_player_tracks[18]
    positions = [player.get(frame,{}).get('bbox',[]) for frame in range(min(player.keys()), max(player.keys())+1)]
    positions_df = pd.DataFrame(positions,columns=['x1','y1','x2','y2'])

    imp = IterativeImputer(max_iter=10, random_state=42)
    print(imp.fit_transform(positions_df))

    positions_df = positions_df.interpolate()
    positions_df = positions_df.bfill()

    # connected_tracks = player_tracks_by_frames(long_player_tracks)
    # print(connected_tracks[0])
    # interpolate_player_tracks(long_player_tracks)

    # connect_tracks(connected_tracks)
    # for frame_num, player_track in enumerate(tracks['players']):
    #     print(frame_num, player_track.get(28, {}))

    # print(player_tracks[28].get(76, {}))