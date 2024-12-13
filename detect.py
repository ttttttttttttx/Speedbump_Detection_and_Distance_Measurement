import cv2
import numpy as np
from ultralytics import YOLO
from utils.generate_video import generate_video_from_frames
from utils.split_video_and_undistort import split_video_to_undistroted_frames

# Program Entry #
if __name__ == '__main__':

    # Process video into undistroted frames
    video_path = "./captured_video/test1.mp4"
    split_video_to_undistroted_frames(video_path)

    # Load YOLOv8 model
    model = YOLO("speedbump.pt") # "speedbump.pt" is the model file

    # YOLO model prediction to show speedbumps
    results = model.predict(source="undistorted_frames", save=False, # classes=1,
                            show_labels=False, show_conf=False)

    # Iterate through the results
    for i, result in enumerate(results):
        # Read the image
        image = cv2.imread(result.path)

        # Iterate through all speedbumps in the image
        for xyxy in result.boxes.xyxy:  
            # Coordinates in format [xmin, ymin, xmax, ymax]
            corners = xyxy.tolist()
            x_min, y_min, x_max, y_max = map(int, corners)

            # Draw a green rectangle box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)

            # The center point of speedbump
            x_mid = (x_min + x_max) / 2
            y_mid = (y_min + y_max) / 2
            H = np.load('configs/H_matrix.npy') 
            mid_point = np.array([[x_mid, y_mid]], dtype=np.float32) 

            # Apply H to the center point
            trans_point = cv2.perspectiveTransform(mid_point.reshape(1, -1, 2), H)
            y_mid_transformed = trans_point[0][0][1] # transformed y-coordinate

            # Distance
            distance = y_mid_transformed

            # Add distance above the rectangle box
            text = "distance=" + str(distance) + "cm"
            cv2.putText(image, text, (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

        # Save the image 
        save_path = "marked_frames\\" + str(i) + ".jpg"
        cv2.imwrite(save_path, image)

    # Combine the marked images into a video
    output_video_path = "./generated_video/generated_video.mp4"
    generate_video_from_frames(output_video_path, 20.0)