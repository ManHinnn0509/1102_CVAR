import os

import numpy as np
import cv2

def main():
    calc("bunny")
    calc("teapot")

    print("--- End of Program ---")

def calc(dir):

    fileNames = [f"./{dir}/{i}" for i in os.listdir(dir) if (i.endswith(".bmp"))]
    imgs = [cv2.imread(i) for i in fileNames]

    s = readFile(f"./{dir}/LightSource.txt")
    lightSrc = splitLightSource(s)

    # d = dict(zip(fileNames, [imgs, lightSrc]))

    # --- Build light source / vector
    src = np.array(lightSrc)
    # print(src)

    src = np.array([i / np.linalg.norm(i) for i in src])
    # print(src)

    # All the images' size should be the same
    
    # y, x
    height, width, ignored = imgs[0].shape

    shape = imgs[0].shape

    albedo = np.zeros(shape)
    normal = np.zeros(shape)

    h = np.zeros(shape)
    p = np.zeros(shape)
    q = np.zeros(shape)

    nx = np.zeros(shape)
    ny = np.zeros(shape)
    nz = np.zeros(shape)

    for y in range(height):
        for x in range(width):
            i = np.array([i[y][x] for i in imgs])

            srcT = src.T
            N = np.dot(
                np.dot(np.linalg.inv(np.dot(srcT, src)), srcT), i
            )

            G = N.T
            # print(G)
            
            G_grey = G[0] * 0.0722 + G[1] * 0.7152 + G[2] * 0.2126
            G_norm = np.linalg.norm(G_grey)

            # If norm != 0 then aldedo != 0 too
            if (G_norm != 0):
                # Normal n = b / |b|
                n = G_grey / G_norm
                normal[y][x] = n

                p[y][x] = -1 * (n[0] / n[2])
                q[y][x] = -1 * (n[1] / n[2])

                nx[y][x] = n[0]
                ny[y][x] = n[1]
                nz[y][x] = n[2]

                # Albedo = |b|
                albedo[y][x] = np.linalg.norm(G, axis=1)

    # --- Height / Depth
    h[0][0] = 0

    for y in range(1, height):
        h[y][0] = h[y - 1][0] + q[y][0]

    for y in range(height):
        for x in range(1, width):
            h[y][x] = h[y][x - 1] + p[y][x]

    # print(h)
    # print(h.shape)

    # 0 ~ 255 Only
    normal = (0.5 + (normal / 2)) * 255
    normal = normal.astype(np.uint8)
    # print(normal.shape)

    albedo = albedo / np.max(albedo) *  255
    albedo = albedo.astype(np.uint8)

    nx = (0.5 + (nx / 2)) * 255
    nx = nx.astype(np.uint8)
    
    ny = (0.5 + (ny / 2)) * 255
    ny = ny.astype(np.uint8)
    
    nz = (0.5 + (nz / 2)) * 255
    nz = nz.astype(np.uint8)
    

    # Show img, see result
    """
    cv2.imshow('Albedo', albedo)
    cv2.imshow('Normal', normal)
    cv2.imshow('Height / Depth', h)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    """
    # cv2.imshow('nx', nx)
    cv2.imshow('ny', ny)
    # cv2.imshow('nz', nz)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    
    # Save data
    cv2.imwrite(f"./{dir}/albedo.png", albedo)
    cv2.imwrite(f"./{dir}/normal.png", normal)

    cv2.imwrite(f"./{dir}/nx.png", nx)
    cv2.imwrite(f"./{dir}/ny.png", ny)
    cv2.imwrite(f"./{dir}/nz.png", nz)

    with open(f"./{dir}/depth.txt", "w+") as f:
        for line in h:
            np.savetxt(f, line, fmt="%d")

def splitLightSource(s):
    data = [i.split(":")[-1].split(",") for i in s.split("\n") if (i != "")]
    data = [[int(i) for i in l] for l in data]
    return data

def readFile(p, encoding="utf-8"):
    try:
        with open(p, "r", encoding=encoding) as f:
            return f.read()

    except Exception as e:
        print(e)
        return None

if (__name__ == "__main__"):
    main()
