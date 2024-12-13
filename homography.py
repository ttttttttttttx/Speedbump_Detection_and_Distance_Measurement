import cv2
import numpy as np

# 2D point coordinates in the image
imgps = [] # the clicked corners
# 2D point coordinates in the real world
objps = np.array([[-60, 120], [0, 120], [0, 240], 
                  [0, 300], [120, 240],  [120, 300]])

# Mouse Callback #
def click_corner(event, x, y, flags, param):
    # When left mouse button is pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        imgps.append([x, y])
        xy = "%d,%d" % (x, y)
        # Draw a circle at the clicked location
        cv2.circle(img, (x, y), 7, (0, 0, 255), thickness=-1) 
        # Display the coordinates at the clicked location
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 0), thickness=2)

# Mark Points on an Image #
def mark_points():
    global imgps
    # Close all OpenCV windows
    cv2.destroyAllWindows() 
    # Create a window to display the image
    cv2.namedWindow("groundBoard")
    # Call click_corner() when clicking on "groundBoard"
    cv2.setMouseCallback("groundBoard", click_corner)

    while (1):
        cv2.imshow("groundBoard", img)
        key = cv2.waitKey(1) & 0xff 
        
        if key == ord('q') or key == ord('Q'):
            imgps = np.array(imgps, dtype=np.float32)
            print(imgps)
            # Save the image
            output_path = './marked_points_image/calibrated_image.png'
            cv2.imwrite(output_path, img)
            break

# Program Entry #
if __name__ == '__main__':
    # Mark points on the groundBoard image
    image_path = './marked_points_image/011_undistorted.jpg'
    img = cv2.imread(image_path)
    mark_points()

    # Calculate the homography matrix H
    H, _ = cv2.findHomography(imgps, objps)
    # Save the homography matrix H
    np.save('configs/H_matrix.npy', H)