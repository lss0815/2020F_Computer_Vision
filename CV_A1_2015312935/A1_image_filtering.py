import numpy as np
import cv2 as cv
import time
import A1_2015312935_module as My


def cross_correlation_1d(img, kernel):
    return My.cross_correlation_1d(img, kernel).astype(np.uint8)


def cross_correlation_2d(img, kernel):
    return My.cross_correlation_2d(img, kernel).astype(np.uint8)


def get_gaussian_filter_1d(size, sigma):
    return My.get_gaussian_filter_1d(size, sigma)


def get_gaussian_filter_2d(size, sigma):
    return My.get_gaussian_filter_2d(size, sigma)


def put_caption_to_img(img, size, sigma):
    result = img.copy()
    cv.putText(result, str(size)+"x"+str(size)+" s="+str(sigma), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, 0)
    return result


def make_gaussian_result_img(img, size, sigma):
    kernel = get_gaussian_filter_1d(size, sigma)
    result = cross_correlation_1d(img, kernel)
    result = cross_correlation_1d(result, np.array([kernel]).T)
    return put_caption_to_img(result, size, sigma)


def abs_difference_images(img1, img2):
    if img1.ndim != img2.ndim or img1.ndim != 2 or img1.shape[0] != img2.shape[0] or img1.shape[1] != img2.shape[1]:
        print("Image format not suitable")
        ret_img = img1
    else:
        r, c = img1.shape
        ret_img = np.zeros((r, c), dtype=np.uint8)
        for i in range(r):
            for j in range(c):
                temp = int(img1[i][j]) - int(img2[i][j])
                if temp < 0:
                    temp *= -1
                ret_img[i][j] = np.uint8(temp)
    return ret_img


if __name__ == '__main__':
    # my gaussian kernels 5, 1
    my_gauss_1d = get_gaussian_filter_1d(5, 1)
    my_gauss_2d = get_gaussian_filter_2d(5, 1)
    print("1D Gaussian Kernel (5, 1)")
    print(my_gauss_1d)
    print("2D Gaussian Kernel (5, 1)")
    print(my_gauss_2d)

    # shared kernels for check 1D, 2D difference
    my_gauss_1d = get_gaussian_filter_1d(17, 6)
    my_gauss_1d_vert = np.array([my_gauss_1d]).T
    my_gauss_2d = get_gaussian_filter_2d(17, 6)

    # shapes.png
    img_shapes = cv.imread('shapes.png', cv.IMREAD_GRAYSCALE)

    img_shapes_5_1 = make_gaussian_result_img(img_shapes, 5, 1)
    img_shapes_5_6 = make_gaussian_result_img(img_shapes, 5, 6)
    img_shapes_5_11 = make_gaussian_result_img(img_shapes, 5, 11)
    img_shapes_11_1 = make_gaussian_result_img(img_shapes, 11, 1)
    img_shapes_11_6 = make_gaussian_result_img(img_shapes, 11, 6)
    img_shapes_11_11 = make_gaussian_result_img(img_shapes, 11, 11)
    img_shapes_17_1 = make_gaussian_result_img(img_shapes, 17, 1)
    img_shapes_17_6 = make_gaussian_result_img(img_shapes, 17, 6)
    img_shapes_17_11 = make_gaussian_result_img(img_shapes, 17, 11)

    img_shapes_result1 = np.concatenate((img_shapes_5_1, img_shapes_5_6, img_shapes_5_11), axis=1)
    img_shapes_result2 = np.concatenate((img_shapes_11_1, img_shapes_11_6, img_shapes_11_11), axis=1)
    img_shapes_result3 = np.concatenate((img_shapes_17_1, img_shapes_17_6, img_shapes_17_11), axis=1)

    img_shapes_result = np.concatenate((img_shapes_result1, img_shapes_result2, img_shapes_result3), axis=0)

    cv.imwrite('./result/part_1_gaussian_filtered_shapes.png', img_shapes_result)
    cv.imshow('Press any key to continue next image', img_shapes_result)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # shapes.png 1D, 2D filtering difference
    stime_1d = time.time()
    img_shapes_1d = cross_correlation_1d(img_shapes, my_gauss_1d)
    img_shapes_1d = cross_correlation_1d(img_shapes_1d, my_gauss_1d_vert)
    etime_1d = time.time()

    stime_2d = time.time()
    img_shapes_2d = cross_correlation_2d(img_shapes, my_gauss_2d)
    etime_2d = time.time()

    img_shapes_dif = abs_difference_images(img_shapes_1d, img_shapes_2d)

    print("shapes.png 17x17 s=6 1D, 2D difference map, sum of intensity :", np.sum(img_shapes_dif))
    print("shapes.png 17x17 s=6 1D filtering computational time :", etime_1d - stime_1d)
    print("shapes.png 17x17 s=6 2D filtering computational time :", etime_2d - stime_2d)

    cv.imshow('Press any key to continue next image', img_shapes_dif)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # lenna.png
    img_lenna = cv.imread('lenna.png', cv.IMREAD_GRAYSCALE)

    img_lenna_5_1 = make_gaussian_result_img(img_lenna, 5, 1)
    img_lenna_5_6 = make_gaussian_result_img(img_lenna, 5, 6)
    img_lenna_5_11 = make_gaussian_result_img(img_lenna, 5, 11)
    img_lenna_11_1 = make_gaussian_result_img(img_lenna, 11, 1)
    img_lenna_11_6 = make_gaussian_result_img(img_lenna, 11, 6)
    img_lenna_11_11 = make_gaussian_result_img(img_lenna, 11, 11)
    img_lenna_17_1 = make_gaussian_result_img(img_lenna, 17, 1)
    img_lenna_17_6 = make_gaussian_result_img(img_lenna, 17, 6)
    img_lenna_17_11 = make_gaussian_result_img(img_lenna, 17, 11)

    img_lenna_result1 = np.concatenate((img_lenna_5_1, img_lenna_5_6, img_lenna_5_11), axis=1)
    img_lenna_result2 = np.concatenate((img_lenna_11_1, img_lenna_11_6, img_lenna_11_11), axis=1)
    img_lenna_result3 = np.concatenate((img_lenna_17_1, img_lenna_17_6, img_lenna_17_11), axis=1)

    img_lenna_result = np.concatenate((img_lenna_result1, img_lenna_result2, img_lenna_result3), axis=0)

    cv.imwrite('./result/part_1_gaussian_filtered_lenna.png', img_lenna_result)
    cv.imshow('Press any key to continue next image', img_lenna_result)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # lenna.png 1D, 2D filtering difference
    stime_1d = time.time()
    img_lenna_1d = cross_correlation_1d(img_lenna, my_gauss_1d)
    img_lenna_1d = cross_correlation_1d(img_lenna_1d, my_gauss_1d_vert)
    etime_1d = time.time()

    stime_2d = time.time()
    img_lenna_2d = cross_correlation_2d(img_lenna, my_gauss_2d)
    etime_2d = time.time()

    img_lenna_dif = abs_difference_images(img_lenna_1d, img_lenna_2d)

    print("lenna.png 17x17 s=6 1D, 2D difference map, sum of intensity :", np.sum(img_lenna_dif))
    print("lenna.png 17x17 s=6 1D filtering computational time :", etime_1d - stime_1d)
    print("lenna.png 17x17 s=6 2D filtering computational time :", etime_2d - stime_2d)

    cv.imshow('Press any key to continue next image', img_lenna_dif)
    cv.waitKey(0)
    cv.destroyAllWindows()
