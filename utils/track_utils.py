
def player_tracks_by_frames(player_tracks: dict) -> dict:
    tracks_by_frames = {}

    for id, player_track in player_tracks.items():
        for frame_num, track in player_track.items():
            if frame_num not in tracks_by_frames:
                tracks_by_frames[frame_num] = {}
            
            tracks_by_frames[frame_num][id] = track
    
    max_frame = max(tracks_by_frames.keys())
    return [tracks_by_frames.get(frame_num, {}) for frame_num in range(max_frame + 1)]

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

def get_transformed(player_track: dict):
    return [frame.get('position_transformed') for frame in player_track.values()]

