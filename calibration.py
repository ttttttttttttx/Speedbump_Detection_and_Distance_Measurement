import cv2
import numpy as np
import glob
import os
from utils.capture import Capture
from utils.file_operations import writeIntriToFile
from configs.camera_intrinsic import cameraMatrix, distCoeff


class Board:
    def __init__(self, col, row, size) -> None:
        self.COL = col  # num of corners' cols
        self.ROW = row  # num of corners' rows
        self.size = size  # (mm)
        pass


def CameraCalibration():
    board = Board(9, 6, 21)  # My calibration board parameters
    input_path = "Img_for_calibration"

    # Camera intrinsic
    K = np.array(np.zeros((3, 3)))  # cameraMatrix
    D = np.array(np.zeros((4, 1)))  # distCoeff

    # Compute for each image
    object_points = []  # points in the world
    image_points = []  # points in the image

    # object points
    objp = np.zeros((board.COL * board.ROW, 1, 3), np.float32)
    objp[:, 0, :2] = np.mgrid[0:board.COL, 0:board.ROW].T.reshape(-1, 2)
    objp = objp*board.size

    # Read images from file
    images = []
    image_paths = glob.glob(input_path + "/*.jpg")
    for image_path in image_paths:
        images.append(cv2.imread(image_path))

    if not images:
        print("Error: Could not read input images.")
        exit(-1)

    board_size = (board.COL, board.ROW)

    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW

    for image in images:
        img_h, img_w, _ = image.shape
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        object_points.append(objp)

        found, corners = cv2.findChessboardCorners(image_gray, board_size, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

        image_points.append(corners)

        # if no match
        if len(object_points[0]) != len(image_points[0]):
            print(f"no match in !")
            return

        # Use found in corners to see if you have found a corner
        # If found
        if found:
            # Improved corner accuracy
            cv2.cornerSubPix(image_gray, corners, (3, 3), (-1, -1), subpix_criteria)

            cv2.drawChessboardCorners(image, board_size, corners, True)

            # Displays pictures with corner dots
            # cv2.imshow("Chessboard Corners", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    # Compute camera matrix and distortion coefficients for fisheye
    rvecs = [np.zeros((1, 1, 3), dtype=np.float32) for i in range(len(image_paths))]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float32) for i in range(len(image_paths))]
    rms, _, _, _, _ = \
        cv2.fisheye.calibrate(
            object_points,
            image_points,
            image_gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )

    print("Camera Matrix (K):")
    print(K)
    print("\nDistortion Coefficients (D):")
    print(D)


    # print(rms) #This parameter shows the calibration effect, less than 0.5 indicates a good calibration effect
    output_filename = "./configs/camera_intrinsic.py"
    # store Camera parameter
    writeIntriToFile(output_filename, K, D)

def undistort(img_path, save_path,K,D,imshow=True):
    img = cv2.imread(img_path)
    # option1
    # map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    # img_undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # option2
    img_undistorted = cv2.fisheye.undistortImage(img, K, D, None, K)
    if imshow:
        cv2.imshow("undistorted", img_undistorted)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    os.makedirs(save_path, exist_ok=True)

    # Writes the dedistorted image to a file
    if save_path is not None:
        # Create file directory
        os.makedirs(save_path, exist_ok=True)

        # Extract a file name without an extension
        file_name = os.path.splitext(os.path.basename(img_path))[0]

        # Save the dedistorted image
        save_file_path = os.path.join(save_path, f"{file_name}_undistorted.jpg")
        cv2.imwrite(save_file_path,  img_undistorted)
        print(f"Undistorted image saved to: {save_file_path}")

    return img_undistorted


if __name__ == '__main__':
    print("Please select the action: ")
    print("[1] Capture Image [2] Calibrate Camera [3] Undistort Image")
    selec = input()
    if selec == '1':
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print(f"Error: Could not open camera at index 1")
            exit()
        else:
            # clear first
            Capture("./Img_for_calibration", 5, cap, isPreClean=True)

    elif selec == '2':
        CameraCalibration()

    elif selec == '3':
        # Read images from file
        images = []
        image_paths = glob.glob("./img_for_calibration/*.jpg")
        for image_path in image_paths:
            images.append(cv2.imread(image_path))
        # print("image_size", images[0].shape[1], "x", images[0].shape[0])
        DIM = (640, 480)
        # undistrotion
        for image_path in image_paths:
            undistort(image_path, "./UndistortionImg_of_chessboard", cameraMatrix, distCoeff)
