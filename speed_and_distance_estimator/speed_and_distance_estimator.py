import cv2
import sys 
sys.path.append('../')  # Add parent directory to the system path to import utility functions
from utils import measure_distance, get_foot_position  # Import utility functions

class SpeedAndDistance_Estimator():
    def __init__(self):
        self.frame_window = 5  # Number of frames over which speed is calculated
        self.frame_rate = 24  # Frames per second (fps) of the video
    
    def add_speed_and_distance_to_tracks(self, tracks):
        """
        Calculate and add speed & total distance traveled for each tracked object.
        """
        total_distance = {}  # Dictionary to store total distance covered by each tracked object

        for object, object_tracks in tracks.items():  # Iterate over all tracked objects
            if object == "ball" or object == "referees":  # Skip ball and referees
                continue 
            number_of_frames = len(object_tracks)  # Total frames available for this object
            
            for frame_num in range(0, number_of_frames, self.frame_window):  # Iterate in steps of frame_window
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)  # Determine last frame in the batch

                for track_id, _ in object_tracks[frame_num].items():  # Iterate over track IDs in the current frame
                    if track_id not in object_tracks[last_frame]:  # Ensure the track ID exists in last frame
                        continue

                    # Get the starting and ending positions of the tracked object
                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    if start_position is None or end_position is None:  # Skip if positions are invalid
                        continue
                    
                    # Calculate distance covered and speed
                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate  # Convert frame difference to seconds
                    speed_meters_per_second = distance_covered / time_elapsed  # Speed in m/s
                    speed_km_per_hour = speed_meters_per_second * 3.6  # Convert to km/h

                    # Initialize total distance dictionary for this object
                    if object not in total_distance:
                        total_distance[object] = {}
                    
                    if track_id not in total_distance[object]:  # Initialize track-specific distance
                        total_distance[object][track_id] = 0
                    
                    total_distance[object][track_id] += distance_covered  # Update total distance covered

                    # Assign speed and distance values for all frames in the batch
                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][frame_num_batch]:  # Skip if track ID is missing in a frame
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour  # Store speed
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]  # Store distance
    
    def draw_speed_and_distance(self, frames, tracks):
        """
        Overlay speed and distance information on each frame.
        """
        output_frames = []  # List to store annotated frames
        for frame_num, frame in enumerate(frames):  # Iterate over frames
            for object, object_tracks in tracks.items():  # Iterate over tracked objects
                if object == "ball" or object == "referees":  # Skip ball and referees
                    continue 
                for _, track_info in object_tracks[frame_num].items():  # Iterate over track info in current frame
                    if "speed" in track_info:  # Check if speed is recorded
                        speed = track_info.get('speed', None)
                        distance = track_info.get('distance', None)
                        if speed is None or distance is None:  # Skip if values are missing
                            continue
                        
                        bbox = track_info['bbox']  # Get bounding box of the tracked object
                        position = get_foot_position(bbox)  # Get position of foot (for players)
                        position = list(position)
                        position[1] += 40  # Adjust position to display text slightly below

                        position = tuple(map(int, position))  # Convert position to integer tuple

                        # Draw speed and distance on the frame
                        cv2.putText(frame, f"{speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1] + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            output_frames.append(frame)  # Append the modified frame to output list
        
        return output_frames  # Return annotated frames