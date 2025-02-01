import pickle
import cv2
import numpy as np
import os
import sys
sys.path.append('../')  # Add the parent directory to the Python path to import custom utilities
from utils import measure_distance, measure_xy_distance  # Import custom utility functions

class CameraMovementEstimator:
    def __init__(self, frame):
        # Minimum distance threshold to consider significant camera movement
        # If the detected movement is less than 5 pixels, it is ignored.
        self.minimum_distance = 5

        # Parameters for Lucas-Kanade optical flow algorithm
        self.lk_params = dict(
            winSize=(15, 15),  # Size of the search window for optical flow/tracking
            maxLevel=2,  # Maximum pyramid level for optical flow/tracking
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)  # Stopping criteria
        ) # 10, 0.03 means stop after 10 iterations or until the accuracy epsilon=0.03 is reached

        # Convert the first frame to grayscale for feature detection
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create a mask to specify regions of the frame where features should be detected
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1  # Left edge of the frame
        mask_features[:, 900:1050] = 1  # Right edge of the frame
        # : means all rows, 0:20 means 0 to 20 columns(the leftmost part of the image)

        # Parameters for detecting good features to track
        self.features = dict(
            maxCorners=100,  # Maximum number of corners to detect
            qualityLevel=0.3,  # Minimum quality of corners to retain
            minDistance=3,  # Minimum distance between detected corners
            blockSize=7,  # Size of the neighborhood for corner detection
            mask=mask_features  # Mask to restrict feature detection to specific regions
        )

    # to "remove" the camera movement effect so that you can focus purely on the movement of the objects in the scene
    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        """
        Adjust the positions of objects in tracks based on camera movement.
        """
        for object, object_tracks in tracks.items():  # Iterate over each object (e.g., players, ball)
            for frame_num, track in enumerate(object_tracks):  # Iterate over each frame
                for track_id, track_info in track.items():  # Iterate over each track in the frame
                    position = track_info['position']  # Get the original position of the object
                    camera_movement = camera_movement_per_frame[frame_num]  # Get camera movement for the frame
                    # Adjust the position by subtracting the camera movement
                    position_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1])
                    # Store the adjusted position in the tracks dictionary
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        """
        Estimate camera movement for each frame using optical flow.
        """
        # If reading from a stub file (precomputed camera movement), load and return it
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # Initialize a list to store camera movement for each frame
        camera_movement = [[0, 0]] * len(frames)
        # [[movement_x, movement_y], [movement_x, movement_y], ...]

        # Remmeber: frame is a numpy array with shape (H, W, C=3)
        # Convert the first frame to grayscale(for reference) and detect features to track
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY) # it becomes 2D image
        # initial features to track
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)
        # what does ** do? it unpacks the dictionary
        # cv2.goodFeaturesToTrack(old_gray, maxCorners=100, qualityLevel=0.3, minDistance=3, blockSize=7, mask=mask_features)


        # Iterate over the remaining frames to estimate camera movement
        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            # Calculate optical flow between the previous and current frame
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            max_distance = 0  # Track the maximum displacement of features
            camera_movement_x, camera_movement_y = 0, 0  # Initialize camera movement

            # Iterate over the tracked features to find the maximum displacement
            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()  # Flatten the new feature point coordinates
                old_features_point = old.ravel()  # Flatten the old feature point coordinates

                # Calculate the distance between the old and new feature points
                distance = measure_distance(new_features_point, old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    # Calculate the x and y components of the camera movement
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)

            # If the maximum displacement exceeds the threshold, update the camera movement
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                # Detect new features in the current frame for the next iteration
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            # Update the previous frame and features for the next iteration
            old_gray = frame_gray.copy()

        # Save the computed camera movement to a stub file if a path is provided
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        """
        Draw the camera movement information on the frames.
        """
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            # Create an overlay to display the camera movement information
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)  # White background
            alpha = 0.6  # Transparency of the overlay
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Get the camera movement for the current frame
            x_movement, y_movement = camera_movement_per_frame[frame_num]

            # Draw the camera movement information on the frame
            frame = cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            frame = cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            output_frames.append(frame)

        return output_frames