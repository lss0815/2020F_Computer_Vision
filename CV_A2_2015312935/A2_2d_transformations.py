import numpy as np
import cv2 as cv
import A2_2015312935_module as My


def init_image(img, size):
    size_ir, size_ic = img.shape
    mid_ir, mid_ic = int(size_ir/2), int(size_ic/2)
    mid_r = int(size/2)
    ret_img = np.zeros((size, size), dtype=np.uint8)
    ret_img.fill(255)
    for i in range(size_ir):
        for j in range(size_ic):
            ret_img[mid_r-mid_ir+i][mid_r-mid_ic+j] = img[i][j]
    return ret_img


def image_with_arrows(img):
    size_ir, size_ic = img.shape
    mid_ir, mid_ic = int(size_ir/2), int(size_ic/2)
    ret_img = img.copy()
    cv.arrowedLine(ret_img, (0, mid_ir), (size_ic-1, mid_ir), 0, 4, tipLength=0.02)
    cv.arrowedLine(ret_img, (mid_ic, size_ir-1), (mid_ic, 0), 0, 4, tipLength=0.02)
    return ret_img


def get_transformed_image(img, M):
    return image_with_arrows(My.get_transformed_image(img, M))


def get_transform_matrix(code):
    ret = np.array([[1,0,0], [0,1,0], [0,0,1]], dtype=np.float64)
    val_sin = np.sin(5.*np.pi/180.)
    val_cos = np.cos(5.*np.pi/180.)
    if code == 'a':
        ret[0][2] = -5
    elif code == 'd':
        ret[0][2] = 5
    elif code == 'w':
        ret[1][2] = -5
    elif code == 's':
        ret[1][2] = 5
    elif code == 'r':
        ret[0][0] = val_cos
        ret[0][1] = val_sin
        ret[1][0] = (-1) * val_sin
        ret[1][1] = val_cos
    elif code == 'R':
        ret[0][0] = val_cos
        ret[0][1] = (-1) * val_sin
        ret[1][0] = val_sin
        ret[1][1] = val_cos
    elif code == 'f':
        ret[0][0] = -1
    elif code == 'F':
        ret[1][1] = -1
    elif code == 'x':
        ret[0][0] = 0.95
    elif code == 'X':
        ret[0][0] = 1.05
    elif code == 'y':
        ret[1][1] = 0.95
    elif code == 'Y':
        ret[1][1] = 1.05
    return ret


if __name__ == '__main__':
    img_smile = cv.imread('smile.png', cv.IMREAD_GRAYSCALE)

    img_origin = init_image(img_smile, 801)
    img_tmp = image_with_arrows(img_origin)

    M = np.array([[1,0,0], [0,1,0], [0,0,1]], dtype=np.float64)

    while True:
        img_tmp = get_transformed_image(img_origin, M)
        cv.imshow('Press any key to continue next image', img_tmp)
        key = cv.waitKey(0)
        if key == ord('Q'):
            cv.destroyAllWindows()
            break
        elif key == ord('H'):
            M = np.array([[1,0,0], [0,1,0], [0,0,1]], dtype=np.float64)
        elif ord('A') <= key <= ord('z'):
            M = np.matmul(get_transform_matrix(chr(key)), M)
