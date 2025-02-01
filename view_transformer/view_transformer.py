import numpy as np 
import cv2

class ViewTransformer():
    def __init__(self):
        # Define the real-world dimensions of the court
        court_width = 68  # Width of the court in real-world units (e.g., meters)
        court_length = 23.32  # Length of the court in real-world units

        # Define the pixel coordinates of the court corners in the image ; Trapazoid
        self.pixel_vertices = np.array([
            [110, 1035],  # Bottom-left corner
            [265, 275],   # Top-left corner
            [910, 260],   # Top-right corner
            [1640, 915]   # Bottom-right corner
        ])
        
        # Define the corresponding real-world coordinates of the court corners ; Rectangle
        self.target_vertices = np.array([
            [0, court_width],   # Bottom-left corner in real-world coordinates
            [0, 0],             # Top-left corner
            [court_length, 0],  # Top-right corner
            [court_length, court_width]  # Bottom-right corner
        ])
        
        # Convert to float32 for OpenCV functions
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        # Compute the perspective transformation matrix
        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        """
        Transforms a given point from image space to real-world coordinates using the perspective transform.
        """
        # Convert the point to integer coordinates
        p = (int(point[0]), int(point[1]))
        
        # Check if the point lies within the trapazoid defined by pixel_vertices
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0 
        if not is_inside:
            return None  # Return None if the point is outside the court
        
        # Reshape the point for transformation
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        
        # Apply the perspective transformation
        transform_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        
        # Reshape the transformed point and return it
        return transform_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        """
        Adds the transformed positions to each tracked object in the dataset.
        """
        for object, object_tracks in tracks.items():  # Iterate over each tracked object (e.g., players, ball)
            for frame_num, track in enumerate(object_tracks):  # Iterate over each frame
                for track_id, track_info in track.items():  # Iterate over each tracking ID in the frame
                    position = track_info['position_adjusted']  # Get the adjusted position of the object
                    position = np.array(position)  # Convert to NumPy array
                    
                    # Transform the position using the perspective transformation
                    position_transformed = self.transform_point(position)
                    
                    # If transformation is successful, convert it to a list
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    
                    # Store the transformed position back in the tracks dictionary
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed