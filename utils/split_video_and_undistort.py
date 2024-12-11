import os
import cv2
from configs.camera_intrinsic import cameraMatrix, distCoeff

# Undistort Video Frames #
def split_video_to_undistroted_frames(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        print("Error: Unable to open the video file.")
        return

    # Get frame_rate and total_frames of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Frame rate: {fps}, Total frames: {frame_count}")

    # Create a folder to save undistorted frames
    undistorted_folder = 'undistorted_frames' 
    os.makedirs(undistorted_folder, exist_ok=True)

    # Loop through each frame of the video
    for frame_num in range(frame_count):
        # Read the next frame
        success, oriframe = cap.read() 
        if not success:
            print(f"Error: Unable to read frame{frame_num+1}. Skipping.")
            continue

        # Undistort the frame
        undistorted_frame = cv2.fisheye.undistortImage(oriframe, cameraMatrix, distCoeff, None, cameraMatrix)
        
        # Save the undistorted frame
        frame_filename = os.path.join(undistorted_folder, f"frame_{frame_num+1:04d}.png")
        cv2.imwrite(frame_filename, undistorted_frame)  

    cap.release()
    print("Splitting completed.")