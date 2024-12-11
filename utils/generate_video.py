import os
import cv2
import glob # to find file pathnames

# Generate Video from Frames #
def generate_video_from_frames(video_path, fps):
    width = 0
    height = 0
    frames = []

    # Get all frames paths and sort them
    jpg_files = glob.glob("./marked_frames/*.jpg")
    frame_paths = sorted(jpg_files, key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    # Add each frame to the list
    for frame_path in frame_paths:
        frames.append(cv2.imread(frame_path))
    
    # Get the dimensions of frames
    if frames: 
        width = frames[0].shape[1]
        height = frames[0].shape[0]  
    else: # the frames list is empty
        print("Failed to load the image.")

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # the video encoder
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height)) 

    # Write each frame to the video file
    for frame in frames:
        video.write(frame)  

    video.release()
    print("Video generation complete.")