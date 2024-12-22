import os
import cv2
import numpy as np
from ultralytics import YOLO
from configs.camera_intrinsic import cameraMatrix, distCoeff

# Global Variables #
result = None
frame_count = 0
skip_frames = 20 

# Detect SpeedBumps of a Frame #
def detect_frame(frame):
    global result

    # YOLO model prediction to show speedbumps
    if frame_count % skip_frames == 0:
        result = model.predict(source=frame, save=False, # classes=1,
                show_labels=False, show_conf=False, conf=0.25, iou=0.45)[0]
    
    # Iterate through all speedbumps
    for xyxy in result.boxes.xyxy:  
        # Coordinates in format [xmin, ymin, xmax, ymax]
        corners = xyxy.tolist()
        x_min, y_min, x_max, y_max = map(int, corners)

        # Draw a green rectangle box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)

        # The center point of speedbump
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        H = np.load('configs/H_matrix.npy') 
        mid_point = np.array([[x_mid, y_mid]], dtype=np.float32) 

        # Apply H to the center point
        trans_point = cv2.perspectiveTransform(mid_point.reshape(1, -1, 2), H)
        y_mid_transformed = trans_point[0][0][1] 

        # Distance
        distance = y_mid_transformed

        # Add distance above the rectangle box
        text = "distance=" + str(distance) + "cm"
        cv2.putText(frame, text, (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return frame

# Undistort Video and Detect SpeedBumps #
def detect_video(output_folder):
    global frame_count 

    # Open an external camera
    inputVideo = cv2.VideoCapture(1) 
    inputVideo.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  
    inputVideo.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) 
    if not inputVideo.isOpened():
        print("Error: Could not open camera at index 1.")

    # Create a folder
    os.makedirs(output_folder, exist_ok=True)

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = inputVideo.get(cv2.CAP_PROP_FPS) 
    width = int(inputVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(inputVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))  
    output_path1 = os.path.join(output_folder, 'detected_video.avi')  
    output_path2 = os.path.join(output_folder, 'origin_video.avi') 
    # VideoWriter object for detected video
    file_detected_video = cv2.VideoWriter(output_path1, fourcc, fps, (width, height))
    # VideoWriter object for original video
    file_origin_video = cv2.VideoWriter(output_path2, fourcc, fps, (width, height))

    while True:
        # Read a frame of video
        ret, oriframe = inputVideo.read() 
 
        if ret: 
            # Undistort and detect the frame   
            undistorted_frame = cv2.fisheye.undistortImage(oriframe, cameraMatrix, distCoeff, None, cameraMatrix) 
            detected_frame = detect_frame(undistorted_frame)

            # Save video frames
            file_detected_video.write(detected_frame)
            file_origin_video.write(oriframe)

            # Resize frames for display
            display_frame = cv2.resize(detected_frame, (640, 360))
            display_oriframe = cv2.resize(oriframe, (640, 360))

            # Display video frames
            cv2.imshow("Original Video", display_oriframe)
            cv2.imshow("Undistorted and Detected Video", display_frame)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    inputVideo.release()
    file_origin_video.release()
    file_detected_video.release()
    cv2.destroyAllWindows()

# Program Entry #
if __name__ == '__main__':
    # Load YOLOv8 model
    model = YOLO("speedbump.pt")
    # Capture video and detect speedbumps
    detect_video("captured_detected_video")