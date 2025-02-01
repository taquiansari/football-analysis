from utils import read_video, save_video # utils contains utility functions
from trackers import Tracker  # trackers contains the 'Tracker' class used to track objects in the video
import cv2 # OpenCV is a library of programming functions mainly aimed at real-time computer vision
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

def main():
    # Read the video 
    video_frames = read_video('input_videos/08fd33_4.mp4')
    # read_video is a utility function that extracts frame from a video and returns a list of frames
    # A frame is a single, still image that makes up a video. It is a single image from a sequence of images that are displayed on a screen at a particular frame rate(eg. 24 fps).

    # Initialize Tracker object with a pre-trained model
    tracker = Tracker('models/best.pt')

    # Get object tracks from the video frames
    # 'get_object_tracks' retrieves tracking information about objects in the video.
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')  
    # The parameters:
    # `read_from_stub=True`: Indicates that precomputed tracking data(for particular input video) should be loaded instead of performing tracking anew.
    # `stub_path='stubs/track_stubs.pkl'`: Path to a pickle file containing precomputed tracking data (stub). 

    # save cropped image of a player
    # for track_id, player in tracks["players"][0].items():
    #     bbox = player["bbox"]
    #     frame = video_frames[0]
        
    #     # crop bbox from frame
    #     cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
    #     # save the cropped image
    #     cv2.imwrite(f"output_videos/cropped_img.jpg", cropped_image)
        
    #     break
    
    # Get object positions 
    tracker.add_position_to_tracks(tracks)
    
    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame) 
    
    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    
    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
        
    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign Ball Aquisition
    player_assigner =PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)


    # Annotate the video frames with tracking information (e.g., bounding boxes, labels).
    # 'draw_annotations' overlays tracking data on the original frames.
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control) 
    
    # Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)
    
    # Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Save the annotated frames as a video file.
    # 'save_video' is a utility function that takes the annotated frames and saves them as a video.
    save_video(output_video_frames, 'output_videos/output.avi')

# Standard Python entry point to run the script. It means that if the script is executed directly, the main() function will be called. And if the script is imported as a module, the main() function will not be called.
if __name__ == "__main__":
    main()