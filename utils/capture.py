import os
import cv2
from utils.file_operations import clean_folder

# Read Frames from a Video and Save Frames #
def capture(folder, num, cap, isPreClean = False):
    # Clean the folder
    if isPreClean == True:
        clean_folder(folder)

    count = 1 # number of frames
    images_count = len(os.listdir(folder)) # number of images in current folder

    while True:
        # Read a frame from the video
        success, frame = cap.read()
        # Exit the loop
        if (num != -1) and (not success or count > num):
            break

        # Display the current frame
        cv2.imshow('Press \'C\' to capture, \'Q\' to exit..', frame)
        # Wait for a key event
        key = cv2.waitKey(1) & 0xff 

        # Press 'Q' to exit the loop
        if key == ord('q') or key == ord('Q'):
            break
        # Press 'C' to save the frame
        if key == ord('c') or key == ord('C'):
            img_path = f'{folder}/{images_count + count}.png'  
            cv2.imwrite(img_path, frame)  
            print(f'captured at {img_path}') 
            count += 1