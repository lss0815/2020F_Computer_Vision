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


def apply_gaussian_filter_img(img, size, sigma):
    kernel = get_gaussian_filter_1d(size, sigma)
    result = cross_correlation_1d(img, kernel)
    result = cross_correlation_1d(result, np.array([kernel]).T)
    return result


if __name__ == '__main__':
    # images read and gaussian 7, 1.5
    img_shapes = cv.imread('shapes.png', cv.IMREAD_GRAYSCALE)
    img_shapes = apply_gaussian_filter_img(img_shapes, 7, 1.5)

    img_lenna = cv.imread('lenna.png', cv.IMREAD_GRAYSCALE)
    img_lenna = apply_gaussian_filter_img(img_lenna, 7, 1.5)

    time_stamp1 = time.time()
    img_shapes_mag, img_shapes_dir = compute_image_gradient(img_shapes)
    time_stamp2 = time.time()
    img_lenna_mag, img_lenna_dir = compute_image_gradient(img_lenna)
    time_stamp3 = time.time()

    print('shapes.png compute_image_gradient computational time :', time_stamp2 - time_stamp1)
    print('lenna.png compute_image_gradient computational time :', time_stamp3 - time_stamp2)

    cv.imwrite('./result/part_2_edge_raw_shapes.png', img_shapes_mag.astype(np.uint8))
    cv.imwrite('./result/part_2_edge_raw_lenna.png', img_lenna_mag.astype(np.uint8))

    cv.imshow('Press any key to continue next image', img_shapes_mag.astype(np.uint8))
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imshow('Press any key to continue next image', img_lenna_mag.astype(np.uint8))
    cv.waitKey(0)
    cv.destroyAllWindows()

    time_stamp1 = time.time()
    img_shapes_sup = non_maximum_suppression_dir(img_shapes_mag, img_shapes_dir)
    time_stamp2 = time.time()
    img_lenna_sup = non_maximum_suppression_dir(img_lenna_mag, img_lenna_dir)
    time_stamp3 = time.time()

    cv.imwrite('./result/part_2_edge_sup_shapes.png', img_shapes_sup.astype(np.uint8))
    cv.imwrite('./result/part_2_edge_sup_lenna.png', img_lenna_sup.astype(np.uint8))

    print('shapes.png non_maximum_suppression_dir computational time :', time_stamp2 - time_stamp1)
    print('lenna.png non_maximum_suppression_dir computational time :', time_stamp3 - time_stamp2)

    cv.imshow('Press any key to continue next image', img_shapes_sup.astype(np.uint8))
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imshow('Press any key to continue next image', img_lenna_sup.astype(np.uint8))
    cv.waitKey(0)
    cv.destroyAllWindows()
