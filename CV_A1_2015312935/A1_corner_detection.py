import numpy as np
import cv2 as cv
import time
import A1_2015312935_module as My


def cross_correlation_1d(img, kernel):
    return My.cross_correlation_1d(img, kernel)


def cross_correlation_2d(img, kernel):
    return My.cross_correlation_2d(img, kernel)


def get_gaussian_filter_1d(size, sigma):
    return My.get_gaussian_filter_1d(size, sigma)


def get_gaussian_filter_2d(size, sigma):
    return My.get_gaussian_filter_2d(size, sigma)


def compute_image_gradient(img):
    return My.compute_image_gradient(img)


def non_maximum_suppression_dir(mag, dir):
    return My.non_maximum_suppression_dir(mag, dir)


def compute_corner_response(img):
    return My.compute_corner_response(img)


def non_maximum_suppression_win(R, winSize):
    return My.non_maximum_suppression_win(R, winSize)


def apply_gaussian_filter_img(img, size, sigma):
    kernel = get_gaussian_filter_1d(size, sigma)
    result = cross_correlation_1d(img, kernel)
    result = cross_correlation_1d(result, np.array([kernel]).T)
    return result


def apply_thresholding_green(img, img_res):
    rnum, cnum = img.shape
    ret_img = np.stack((img, )*3, axis=-1)
    for i in range(rnum):
        for j in range(cnum):
            if img_res[i][j] > 0.1:
                ret_img[i][j] = [0, 255, 0]
    return ret_img


def apply_thresholding_green_circle(img, img_res):
    rnum, cnum = img.shape
    ret_img = np.stack((img, )*3, axis=-1)
    for i in range(rnum):
        for j in range(cnum):
            if img_res[i][j] != 0:
                ret_img = cv.circle(ret_img, (j, i), 5, (0, 255, 0), 2)
    return ret_img


if __name__ == '__main__':
    # images read and gaussian 7, 1.5
    img_shapes = cv.imread('shapes.png', cv.IMREAD_GRAYSCALE)
    img_shapes_g = apply_gaussian_filter_img(img_shapes, 7, 1.5)

    img_lenna = cv.imread('lenna.png', cv.IMREAD_GRAYSCALE)
    img_lenna_g = apply_gaussian_filter_img(img_lenna, 7, 1.5)

    time_stamp1 = time.time()
    img_shapes_res = compute_corner_response(img_shapes_g)
    time_stamp2 = time.time()
    img_lenna_res = compute_corner_response(img_lenna_g)
    time_stamp3 = time.time()

    cv.imwrite('./result/part_3_corner_raw_shapes.png', (img_shapes_res*255).astype(np.uint8))
    cv.imwrite('./result/part_3_corner_raw_lenna.png', (img_lenna_res*255).astype(np.uint8))

    print('shapes.png compute_corner_response computational time :', time_stamp2 - time_stamp1)
    print('lenna.png compute_corner_response computational time :', time_stamp3 - time_stamp2)

    cv.imshow('Press any key to continue next image', img_shapes_res)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imshow('Press any key to continue next image', img_lenna_res)
    cv.waitKey(0)
    cv.destroyAllWindows()

    img_shapes_bin = apply_thresholding_green(img_shapes, img_shapes_res)
    img_lenna_bin = apply_thresholding_green(img_lenna, img_lenna_res)

    cv.imwrite('./result/part_3_corner_bin_shapes.png', img_shapes_bin)
    cv.imwrite('./result/part_3_corner_bin_lenna.png', img_lenna_bin)

    cv.imshow('Press any key to continue next image', img_shapes_bin)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imshow('Press any key to continue next image', img_lenna_bin)
    cv.waitKey(0)
    cv.destroyAllWindows()

    time_stamp1 = time.time()
    img_shapes_res_sup = non_maximum_suppression_win(img_shapes_res, 11)
    time_stamp2 = time.time()
    img_lenna_res_sup = non_maximum_suppression_win(img_lenna_res, 11)
    time_stamp3 = time.time()

    img_shapes_sup = apply_thresholding_green_circle(img_shapes, img_shapes_res_sup)
    img_lenna_sup = apply_thresholding_green_circle(img_lenna, img_lenna_res_sup)

    print('shapes.png non_maximum_suppression_win computational time :', time_stamp2 - time_stamp1)
    print('lenna.png non_maximum_suppression_win computational time :', time_stamp3 - time_stamp2)

    cv.imwrite('./result/part_3_corner_sup_shapes.png', img_shapes_sup)
    cv.imwrite('./result/part_3_corner_sup_lenna.png', img_lenna_sup)

    cv.imshow('Press any key to continue next image', img_shapes_sup)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imshow('Press any key to continue next image', img_lenna_sup)
    cv.waitKey(0)
    cv.destroyAllWindows()
