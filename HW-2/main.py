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
    
    # 01 <- L ?
    # 02 <- R ?
    img01 = imgs[0]
    img02 = imgs[1]

    stereo = cv2.StereoBM_create(
            numDisparities=0,
            blockSize=21
    )

    depth = stereo.compute(img01, img02)
    
    """
    cv2.imshow("img01", img01)
    cv2.imshow("img02", img02)
    cv2.imshow("Depth", depth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    plt.imshow(depth, "gray")
    plt.axis("off")
    plt.show()

    print("--- End of Program ---")


if (__name__ == "__main__"):
    main()
