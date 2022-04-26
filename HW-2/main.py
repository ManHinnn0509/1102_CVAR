import os

import numpy as np
import cv2
from matplotlib import pyplot as plt

FLAG = cv2.IMREAD_GRAYSCALE

def main():

    l = ["cones", "tsukuba", "venus"]
    for name in l:
        p = f"./databases/{name}"
        # print(os.listdir(p)

        mainCalc(p)

    print("--- End of Program ---")

def mainCalc(p):

    # Dataset name
    name = p.split("/")[-1]

    imgs = [cv2.imread(f"{p}/{i}", FLAG) for i in os.listdir(p)]
    # print(imgs)

    # Downscale
    # imgs = [cv2.pyrDown(i) for i in imgs]
    
    imgL = imgs[0]
    imgR = imgs[1]

    """
    cv2.StereoBM_create() the disparity is computed by comparing the sum of absolute differences (SAD) of each 'block' of pixels.
    In semi-global block matching or cv2.StereoSGBM_create() forces similar disparity on neighbouring blocks.
    
    This creates a more complete disparity map but is more computationally expensive.

    calcDisp_2() <- StereoSGBM_create
    calcDisp_1() <- StereoBM_create
    """

    disp = calcDisp_2(imgL, imgR)

    """
    depth = (baseline * focal length) / disp
    
    From: https://github.com/pablospe/tsukuba_db/blob/master/README.Tsukuba
        baseline = 10
        focal length = 615
    """
    depth = (10 * 615) / disp
    print(depth)

    # plt.imshow(disp)
    # plt.imshow(depth)
    # plt.axis("off")
    # plt.show()

    """
    for img in [disp, depth]:
        pltDisplay(img)
    """
    
    # OverflowError: cannot convert float infinity to integer??
    saveMap(f"./{name}_depth.txt", depth)
    # saveMap(f"./{name}_disp.txt", disp)

    # Save img
    # https://stackoverflow.com/questions/19239381/pyplot-imsave-saves-image-correctly-but-cv2-imwrite-saved-the-same-image-as
    cv2.imwrite(f"./{name}_disp.png", disp * 255)

    print(f"Done! ({p})")

def saveMap(p, map):

    # https://stackoverflow.com/questions/2844922/how-to-fix-this-python-error-overflowerror-cannot-convert-float-infinity-to-in
    def fixINF(row):
        INF = 1e6
        return [max(min(x, INF), -INF) for x in row]

    with open(p, "w+") as f:
        for line in map:
            line = fixINF(line)
            np.savetxt(f, line, fmt="%d")

def pltDisplay(img):
    plt.figure()
    plt.imshow(img)
    plt.show()

def calcDisp_2(imgL, imgR, winSize=3, minDisp=16):
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

    disp = stereo.compute(imgL, imgR)
    disp = disp.astype(np.float32) / 16.0
    disp = (disp - minDisp) / numDisp

    return disp

def calcDisp_1(imgL, imgR, numDisparities=16, blockSize=21):
    stereo = cv2.StereoBM_create(
        numDisparities=numDisparities,
        blockSize=blockSize
    )

    disp = stereo.compute(imgL, imgR)
    return disp

def displayImages(l: list):
    for img in l:
        cv2.imshow(img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if (__name__ == "__main__"):
    main()
