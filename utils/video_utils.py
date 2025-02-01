import cv2

# Reads a video file and converts it into a list of frames (images).
def read_video(video_path):
    # Open the video file using OpenCV's VideoCapture
    cap = cv2.VideoCapture(video_path)
    
    # Initialize an empty list to store the frames.
    frames = [] 
    
    # Loop through the video until there are no more frames.
    while True:
        # Read a single frame from the video.
        # `ret`: A boolean flag indicating if the frame was successfully read.
        # `frame`: The actual frame data (a NumPy array representing an image).
        ret, frame = cap.read()
        
        # If `ret` is False, it means there are no more frames to read, so exit the loop.
        if not ret: 
            break
        
        # Append the successfully read frame to the `frames` list.
        frames.append(frame)
        
    # Return the list of frames as the result.
    return frames

# Saves a sequence of frames back into a video file.
def save_video(output_video_frames, output_video_path):
    # Define the codec (XVID in this case) for saving the video.
    # `fourcc` is a 4-character code that specifies the video codec.
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    
    # Create a VideoWriter object to write the video file.
    # Parameters:
    # `output_video_path`: The path to save the output video.
    # `fourcc`: The codec to use (XVID).
    # `24.0`: The frames per second (FPS) for the output video.
    # `(width, height)`: Resolution of the video frames (determined from the first frame).
    out = cv2.VideoWriter(output_video_path, fourcc, 24.0, (output_video_frames[0].shape[1], output_video_frames[0].shape[0])) # shape[1] is width and shape[0] is height
    
    # Loop through the list of frames and write each frame to the video file.
    for frame in output_video_frames:
        out.write(frame)
        
    # Release the VideoWriter object to finalize the video file.
    out.release()
    
