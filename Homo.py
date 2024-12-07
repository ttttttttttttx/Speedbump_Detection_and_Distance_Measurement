import cv2
import numpy as np
imgps=np.array([[898,682],[1072,677],[1070,810],[857,809],[1284,795],[1248,663]])
objps=np.array([[-45.45, 136.35],[-15.15,136.35], [-15.15,106.05],[-45.45,106.05], [15.15,106.05], [15.15,136.35]])



if __name__ == '__main__':

    H, _ = cv2.findHomography(imgps, objps)
    np.save('configs/H_matrix.npy', H)
    