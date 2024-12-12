import os
import cv2
import glob
import numpy as np
from utils.capture import capture
from utils.file_operations import write_intri_to_file
from configs.camera_intrinsic import cameraMatrix, distCoeff

# Chessboard Class #
class Board:
    def __init__(self, col, row, size):
        self.COL = col
        self.ROW = row
        self.size = size 
        pass

# Camera Calibration #
def camera_calibration():

    image_points = []  # image coordinate points
    object_points = [] # world coordinate points
    
    K = np.array(np.zeros((3, 3))) # camera intrinsic matrix
    D = np.array(np.zeros((4, 1))) # distortion coefficients

    # Define world coordinate points
    board = Board(9, 6, 23) # initialize chessboard
    board_size = (board.COL, board.ROW)  
    objp = np.zeros((board.COL * board.ROW, 1, 3), np.float32)
    objp[:, 0, :2] = np.mgrid[0:board.COL, 0:board.ROW].T.reshape(-1, 2)
    objp = objp * board.size 

    # Read all calibration images
    images = []
    input_path = "./calibration_images"
    image_paths = glob.glob(input_path + "/*.jpg")  
    # Add all images to the list
    for image_path in image_paths:
        images.append(cv2.imread(image_path)) 
    if not images:
        print("Error: Unable to read input images.")
        exit(-1)  

    # Termination criteria for corner detection
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-9)
    # Parameters for finding chessboard corners
    criteria = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE

    # Iterate over all images
    for i, image in enumerate(images):

        # Find chessboard corners
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
        found, corners = cv2.findChessboardCorners(image_gray, board_size, criteria)
        object_points.append(objp)  
        image_points.append(corners)  

        # If corners are found
        if found:
            # Refine corner coordinates
            cv2.cornerSubPix(image_gray, corners, (3, 3), (-1, -1), subpix_criteria)
            # Draw corners and save
            img_corners = cv2.drawChessboardCorners(image, board_size, corners, True)
            save_path = os.path.join("./calibration_corners_images", f"{i}_corners.jpg")
            cv2.imwrite(save_path, img_corners) 

    # Calibrate using the fisheye model
    rvecs = [np.zeros((1, 1, 3), dtype=np.float32) for i in range(len(image_paths))] # rotation vectors
    tvecs = [np.zeros((1, 1, 3), dtype=np.float32) for i in range(len(image_paths))] # translation vectors
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
    termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-9)
    rms, _, _, _, _ = cv2.fisheye.calibrate(object_points, image_points, image_gray.shape[::-1], K, D,              
                                            rvecs, tvecs, calibration_flags, termination_criteria)

    print("Camera intrinsic matrix (K): \n", K)
    print("\nDistortion coefficients (D): \n", D)
    print("\nRoot Mean Square Error (rms): ", rms) # < 0.5 indicates good calibration

    # Save camera parameters
    output_filename = "./configs/camera_intrinsic.py"
    write_intri_to_file(output_filename, K, D)

# Undistort Images #
def undistort(img_path, save_path, K, D):
    
    img = cv2.imread(img_path) 

    # Undistortion method 1: Using initUndistortRectifyMap and remap
    # DIM = img.shape[:2][::-1]  # Get image dimensions
    # map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    # img_undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Undistortion method 2: Using undistortImage
    img_undistorted = cv2.fisheye.undistortImage(img, K, D, None, K) # undistort using fisheye model

    # Save undistorted images
    os.makedirs(save_path, exist_ok=True) 
    file_name = os.path.splitext(os.path.basename(img_path))[0] # extract file name without extension
    save_file_path = os.path.join(save_path, f"{file_name}_undistorted.jpg")
    cv2.imwrite(save_file_path, img_undistorted) 
    print(f"Undistorted image saved to: {save_file_path}")

# Program Entry #
if __name__ == '__main__':

    print("Please select an operation:")
    print("[1] Capture images [2] Calibrate camera [3] Undistort images")
    select = input() 

    # Capture images #
    if select == '1':
        cap = cv2.VideoCapture(1) # open external camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) 
        if cap.isOpened():
            # capture("./calibration_images", 15, cap, isPreClean=True) # clean folder
            capture("./calibration_images", 15, cap, isPreClean=False) # not clean folder
        else:
            print("Error: Unable to open camera with index 1.")
            exit()

    # Calibrate camera # 
    elif select == '2':
        camera_calibration()

    # Undistort images # 
    elif select == '3':
        # Get a list of all image paths
        image_paths = glob.glob("./marked_points_image/*.jpg")
        # Process each image for undistortion
        for image_path in image_paths:
            undistort(image_path, "./marked_points_image", cameraMatrix, distCoeff)