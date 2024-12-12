import os
import cv2
from configs.camera_intrinsic import cameraMatrix, distCoeff

# Capture Video #
def capture_video(output_folder):
    # Open an external camera
    inputVideo = cv2.VideoCapture(1)
    if not inputVideo.isOpened():
        print("Error: Could not open camera at index 1.")

    # Create a folder
    os.makedirs(output_folder, exist_ok=True)

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # video encoder
    fps = inputVideo.get(cv2.CAP_PROP_FPS) 
    width = int(inputVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(inputVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))  
    output_path = os.path.join(output_folder, 'captured_video.avi') 
    outputVideo = cv2.VideoWriter(output_path, fourcc, fps, (width, height)) # VideoWriter object

    while inputVideo.isOpened():
        ret, frame = inputVideo.read() # read a frame of video
        if ret:
            cv2.imshow('Capturing Video', frame) # display the video frame
            outputVideo.write(frame) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    inputVideo.release()
    outputVideo.release()
    cv2.destroyAllWindows() # close all OpenCV windows

# Undistort Video #
def undistort_video(output_folder):
    # Open an external camera
    inputVideo = cv2.VideoCapture(1)
    if not inputVideo.isOpened():
        print("Error: Could not open camera at index 1.")

    # Create a folder
    os.makedirs(output_folder, exist_ok=True)

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = inputVideo.get(cv2.CAP_PROP_FPS) 
    width = int(inputVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(inputVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))  
    output_path1 = os.path.join(output_folder, 'undistorted_video.avi')  
    output_path2 = os.path.join(output_folder, 'origin_video.avi') 
    # VideoWriter object for undistorted video
    file_undistorted_video = cv2.VideoWriter(output_path1, fourcc, fps, (width, height))
    # VideoWriter object for original video
    file_origin_video = cv2.VideoWriter(output_path2, fourcc, fps, (width, height))

    while inputVideo.isOpened():
        # Read a frame of video
        ret, oriframe = inputVideo.read() 
        if ret:
            # Undistort the video frame
            undistorted_frame = cv2.fisheye.undistortImage(oriframe, cameraMatrix, distCoeff, None, cameraMatrix)

            # Save video frames
            file_undistorted_video.write(undistorted_frame)
            file_origin_video.write(oriframe)

            # Display video frames
            cv2.imshow("Original Video", oriframe)
            cv2.imshow("Undistorted Video", undistorted_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    inputVideo.release()
    file_origin_video.release()
    file_undistorted_video.release()
    cv2.destroyAllWindows()

# Program Entry #
if __name__ == '__main__':
    print("Please select an operation: ")
    print("[1] Capture Video [2] Calibrate Camera")

    select = input()  
    # Capture video
    if select == '1':
        capture_video("captured_video")
    # Capture video and undistort it
    elif select == '2':
        undistort_video("captured_undistorted_video")