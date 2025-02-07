import cv2
import sys
sys.path.append('../')
from utils import measure_distance, get_foot_position

class SpeedAndDistanceEstimator:
    def __init__(self) -> None:
        self.frame_window = 5
        self.frame_rate = 24

    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}

        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referee":
                continue

            num_frames = len(object_tracks)
            for frame_num in range(0, num_frames, self.frame_window):
                last_frame = min(frame_num+self.frame_window,num_frames-1)

                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    if start_position.size==0 or end_position.size==0:
                        continue
                    
                    distance_covered = measure_distance(start_position,end_position)/100
                    time_elapsed = (last_frame-frame_num)/self.frame_rate
                    speed_ms = distance_covered/time_elapsed
                    speed_kmph = speed_ms * 3.6

                    if object not in total_distance:
                        total_distance[object] = {}

                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0

                    total_distance[object][track_id] += distance_covered

                    for frame_num_batch in range(frame_num,last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue

                        tracks[object][frame_num_batch][track_id]['speed'] = speed_kmph
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]

    def draw_speed_and_distance(self, frames, tracks):
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referee":
                    continue

                for _, track in object_tracks[frame_num].items():
                    if 'speed' in track:
                        speed = track.get('speed', 0)
                        distance = track.get('distance', 0)

                        bbox = track['bbox']
                        position = get_foot_position(bbox)
                        position = list(position)
                        position[1] += 40

                        position = tuple(map(int,position))


                        frame = cv2.putText(frame, f"Speed: {speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 
                                            0.5, (255,255,255),2)
                        frame = cv2.putText(frame, f"Distance: {distance:.2f} m", (position[0],position[1]+20), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)
            
            yield frame
    