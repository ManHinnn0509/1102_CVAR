import os

import numpy as np
import cv2
from matplotlib import pyplot as plt

FLAG = cv2.IMREAD_GRAYSCALE

def main():
    p = "./databases/tsukuba"
    # print(os.listdir(p))
    
    imgs = [cv2.imread(f"{p}/{i}", FLAG) for i in os.listdir(p)]
    # print(imgs)

    # Downscale
    # imgs = [cv2.pyrDown(i) for i in imgs]
    
    # 01 <- L ?
    # 02 <- R ?
    img01 = imgs[0]
    img02 = imgs[1]

    """
    cv2.StereoBM_create() the disparity is computed by comparing the sum of absolute differences (SAD) of each 'block' of pixels.
    In semi-global block matching or cv2.StereoSGBM_create() forces similar disparity on neighbouring blocks.
    
    This creates a more complete disparity map but is more computationally expensive.
    """
    
    winSize = 3
    minDisp = 16
    numDisp = 112 - minDisp

    stereo = cv2.StereoSGBM_create(
        minDisparity = minDisp,
        numDisparities = numDisp,
        blockSize = 16,
        P1 = 8 * 3 * winSize * winSize,
        P2 = 32 * 3 * winSize * winSize,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )
    
    """
    stereo = cv2.StereoBM_create(
        numDisparities = 16,
        blockSize = 21
    )
    """

    disp = stereo.compute(img01, img02)
    disp = disp.astype(np.float32) / 16.0
    disp = (disp - minDisp) / numDisp
    
    cv2.imshow("img01", img01)
    cv2.imshow("img02", img02)
    cv2.imshow("disp", disp)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    """
    plt.imshow(disp)
    plt.axis("off")
    plt.show()
    """

    print("--- End of Program ---")


if (__name__ == "__main__"):
    main()
