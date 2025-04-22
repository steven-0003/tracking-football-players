
def player_tracks_by_frames(player_tracks: dict, num_frames: int) -> dict:
    """Converts player tracks from a dictionary format to a list format, where each frame contains the player tracks."""
    tracks_by_frames = {i: {} for i in range(num_frames)}

    for id, player_track in player_tracks.items():
        for frame_num, track in player_track.items():
            tracks_by_frames[frame_num][id] = track
    
    return [tracks_by_frames.get(frame_num, {}) for frame_num in range(num_frames)]

def player_tracks_by_ids(tracks) -> dict:
    """Converts player tracks from a list format to a dictionary format, where each player ID contains their tracks."""
    player_tracks_by_ids = {}

    for frame_num, player_tracks in enumerate(tracks["players"]):
        for id, player_track in player_tracks.items():
            if id not in player_tracks_by_ids.keys():
                player_tracks_by_ids[id] = {}

            player_tracks_by_ids[id][frame_num] = player_track

    return player_tracks_by_ids

def remove_short_tracks(player_tracks: dict, threshold: int) -> dict:
    """Removes player tracks that are shorter than a specified threshold."""
    return {player_id: player_track for player_id, player_track in player_tracks.items() if len(player_track)>threshold}

def get_transformed(player_track: dict) -> list:
    """Extracts the transformed positions from player tracks."""
    return [frame.get('position_transformed') for frame in player_track.values()]

