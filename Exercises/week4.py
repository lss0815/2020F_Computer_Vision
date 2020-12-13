import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def SubLaplacian(Img):
    gN0 = cv.GaussianBlur(Img, (3,3), 1)
    gN = gN0[::2, ::2]
    gN0 = gN.repeat(2, axis=0).repeat(2, axis=1)
    print(gN0[0])
    print(Img[0])
    L = Img - gN0
    print(L[0])
    return L, gN

#Laplacian three times
def MyLaplacian(Img):
    L1, gN = SubLaplacian(Img)
    L2, gN = SubLaplacian(gN)
    L3, gN = SubLaplacian(gN)
    return L1, L2, L3, gN

def MyRecon(L1, L2, L3, gN):
    gN = gN.repeat(2, axis=0).repeat(2, axis=1) + L3
    gN = gN.repeat(2, axis=0).repeat(2, axis=1) + L2
    Img = gN.repeat(2, axis=0).repeat(2, axis=1) + L1
    return Img

if __name__ == '__main__':
    Img = cv.imread('island.jpg', cv.IMREAD_COLOR)
    L1, L2, L3, gN = MyLaplacian(Img)
    Img_Recon = MyRecon(L1, L2, L3, gN)
    result = Img - Img_Recon
    print("L1")
    print(L1)
    print("L2")
    print(L2)
    print("L3")
    print(L3)
    print("result")
    print(result)

"""
    plt.subplot(151), plt.imshow(cv.cvtColor(Img, cv.COLOR_BGR2RGB))
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(152), plt.imshow(cv.cvtColor(L1, cv.COLOR_BGR2RGB))
    plt.title('L1 Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(153), plt.imshow(cv.cvtColor(L2, cv.COLOR_BGR2RGB))
    plt.title('L2 Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(154), plt.imshow(cv.cvtColor(L3, cv.COLOR_BGR2RGB))
    plt.title('L3 Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(155), plt.imshow(cv.cvtColor(gN, cv.COLOR_BGR2RGB))
    plt.title('Generated Image'), plt.xticks([]), plt.yticks([])
    
    plt.show()
"""