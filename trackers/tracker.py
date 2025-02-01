from ultralytics import YOLO
import supervision as sv
import pickle 
import os
import numpy as np
import pandas as pd
import cv2
import sys

# When Python encounters an import statement, it looks for the specified module in the directories listed in sys.path
sys.path.append('../')
# manually adds the parent directory (../) to the list of paths in sys.path. This ensures that Python can locate modules in the parent directory when importing them.

# Importing utility functions to calculate center and width of bounding boxes
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        # YOLO() function is defined in the ultralytics library
        # Initialize the YOLO model using the provided path
        self.model = YOLO(model_path)
        # loads the neural network architecture and weights stored in the file 'model_path'
        # returns an object that represents the YOLO model. This object provides : model.predict() and model.train() methods to make predictions and train the model on custom dataset respectively.
        
        # Initialize a ByteTrack tracker from the supervision library
        # ByteTrack is a popular multi-object tracking algorithm designed to track objects in video frames.
        self.tracker = sv.ByteTrack()
        
    def add_position_to_tracks(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position
    
    def interpolate_ball_positions(self, ball_positions):
        # Extract the bounding box (bbox) for the ball from each frame
        # ball_positions is a list of dictionaries, where each dictionary represents a frame
        # Each frame dictionary has a key `1` (ball ID), and its value is another dictionary containing the bbox
        # Example: ball_positions = [{1: {"bbox": [x1, y1, x2, y2]}}, {1: {"bbox": []}}, ...]
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]

        # Convert the list of ball positions into a Pandas DataFrame
        # Each row in the DataFrame represents a frame, and columns represent the bbox coordinates
        # Columns: 'x1', 'y1', 'x2', 'y2' (top-left and bottom-right corners of the bbox)
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values in the DataFrame
        # Linear interpolation is used to fill gaps where the ball was not detected
        # For example, if frame 1 has bbox [10, 20, 30, 40] and frame 3 has bbox [50, 60, 70, 80],
        # frame 2 will be interpolated as [30, 40, 50, 60]
        df_ball_positions = df_ball_positions.interpolate()

        # Backward fill any remaining missing values
        # This ensures that if the first few frames have missing values, they are filled using the next available value
        df_ball_positions = df_ball_positions.bfill()

        # Convert the DataFrame back into the original list of dictionaries format
        # Each row in the DataFrame is converted back into a dictionary with the ball ID (1) and its interpolated bbox
        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        # Return the interpolated ball positions
        return ball_positions
    
    def detect_frames(self, frames):
        """
        Detect objects in video frames using the YOLO model.
        Frames are processed in batches to optimize performance.
        """
        batch_size = 20 # Number of frames processed at a time
        detections = []
        
        # Iterate over frames in batches
        for i in range(0, len(frames), batch_size):
            # Perform detection on the current batch with a confidence threshold of 0.1
            # conf=0.1 means that the model will include all detections with a confidence score of 10% or higher.
            # It is a relatively low threshold
            # This is useful when you want to capture all possible detections (even less certain ones) for further processing or filtering.
            
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch  # Append the detections to the results list

        return detections # Return all detections from the frames

    def get_object_tracks(self, frames, read_from_stub=False,stub_path=None):
        """
        Get object tracks for players, referees, and the ball.
        Optionally, load precomputed tracks from a stub file.
        
        """
        # If stub data exists and is specified, load it directly
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        # Otherwise, perform detection on the frames                        
        detections = self.detect_frames(frames)

        # Dictionary to store tracks for players, referees, and the ball
        # tracks = {
        #     "players": [
        #         {1: {"bbox": [100, 200, 150, 250], "class_id": 0, "confidence": 0.92, "center": (125, 225), ...}},
        #         {2: {"bbox": [300, 400, 350, 450], "class_id": 0, "confidence": 0.95, "center": (325, 425), ...}},
        #     ],
        #     "referees": [
        #         {3: {"bbox": [50, 60, 100, 160], "class_id": 1, "confidence": 0.88, "center": (75, 110), ...}},
        #     ],
        #     "ball": [
        #         {1: {"bbox": [400, 500, 450, 550], "class_id": 2, "confidence": 0.99, "center": (425, 525), ...}},
        #     ]
        # }       

        tracks = {
            # These are all lists of dictionaries
            "players": [], # "players": [{track_id: {"bbox": [x1, y1, x2, y2]},"class_id": 0, ...}},],   
            "referees": [],
            "ball": [], # List of dictionaries with ball tracking data
        }

        # Iterate over each frame's detections
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names # Class names detected by YOLO
            # syntax : {id(integer): class_name(string), ...}
            
            cls_names_inv = {v: k for k, v in cls_names.items()} # Reverse mapping of class names
            
            # Convert YOLO detection to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Change "goalkeeper" detections to "player"
            # detection_supervision.class_id = list (or iterable) of class IDs.
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind] = cls_names_inv['player']

            # Update tracker with detections and obtain tracked objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision) 

            # Initialize tracking data for the current frame
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            # Process tracked detections (for players and referees)
            for frame_detection in detection_with_tracks:
                # frame_detection = [bounding_box, confidence, detection_label, class_id, track_id]
                bbox = frame_detection[0].tolist() # Bounding box coordinates
                cls_id = frame_detection[3] # Class ID
                track_id = frame_detection[4] # Unique track ID
                # track_id refers to the unique ID assigned to each detected object. In each frame, the track_id allows the system to know whether a detected object corresponds to the same object in previous frames or not

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox} 

            for frame_detection in detection_supervision: # loop over detection without the tracks, this is for ball, where we don't track it
                bbox = frame_detection[0].tolist() # Bounding box coordinates
                cls_id = frame_detection[3] # Class ID

                # Add ball detections to the tracking dictionary
                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}
                # 1 is used as a fixed track ID to store the bounding box of the ball in the tracks["ball"] dictionary. It’s a way to uniquely identify the ball's bounding box across all frames in the video.   
        
        # Save the tracks to a stub file if a path is provided
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
       
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        """
        Draw an ellipse at the bottom of the bounding box for visualization.
        """
        
        y2 = int(bbox[3]) # Bottom y-coordinate of the bounding box
        x_center, _ = get_center_of_bbox(bbox) # Get the center x-coordinate
        width = get_bbox_width(bbox) # Get the width of the bounding box
        cv2.ellipse(
            frame,
            center=(x_center, y2), # Position at the bottom-center of the bbox
            axes=(int(width), int(0.35*width)), # Size of the ellipse
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
            )
        
        # for little box on ellipse marking player's number
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) + 15
        y2_rect = (y2 + rectangle_height//2) + 15 
        
        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)), 
                          color,
                          cv2.FILLED)
            # Adding padding to the text
            x1_text = x1_rect + 12
            
            # if track_id > 99, then reduce the x1_text value to fit the text in the box
            if track_id > 99:
                x1_text -=10
                
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)), 
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                (0,0,0),
                2
            )
        

        return frame
    
    # draw a triangle icon on top of the ball
    def draw_triangle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20], 
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame
    
    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # Draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame 
    
    def draw_annotations(self, video_frames, tracks, team_ball_control):
        """
        Annotate video frames with tracking data (e.g., draw ellipses for players).
        """
        
        output_video_frames = [] # List to store annotated frames
        
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy() # Make a copy of the current fram
            player_dict = tracks["players"][frame_num] # Player tracking data
            referee_dict = tracks["referees"][frame_num] # Referee tracking data
            ball_dict = tracks["ball"][frame_num] # Ball tracking data

            # Draw ellipses for players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255)) # Get player team color
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
                
                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, player["bbox"],(0,0,255))
                
            # Draw ellipses for referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))
                
            # Draw triangle for ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"],(0,255,0))
                
            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            # Append the annotated frame to the output list
            output_video_frames.append(frame)

        return output_video_frames # Return the annotated frames
           