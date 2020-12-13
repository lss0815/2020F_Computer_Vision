import numpy as np
import math


def vertical_cross_correlation_1d(img, kernel):
    krnum = kernel.shape[0]
    irnum, icnum = img.shape
    temp_img = img.copy().astype(np.float64)
    ret_img = np.zeros((irnum, icnum), dtype=np.float64)
    kmid = int(krnum / 2)
    temp_img = np.pad(temp_img, ((kmid, kmid), (0, 0)), mode='edge')

    for i in range(krnum):
        ret_img += temp_img[i:i+irnum] * kernel[i][0]

    return ret_img


def horizontal_cross_correlation_1d(img, kernel):
    kcnum = kernel.shape[0]
    irnum, icnum = img.shape
    temp_img = img.copy().astype(np.float64)
    ret_img = np.zeros((irnum, icnum), dtype=np.float64)
    kmid = int(kcnum / 2)
    temp_img = np.pad(temp_img, ((0, 0), (kmid, kmid)), mode='edge')

    for i in range(kcnum):
        ret_img += temp_img[:, i:i+icnum] * kernel[i]

    return ret_img


def cross_correlation_1d(img, kernel):
    check_2d = kernel.ndim
    if check_2d == 1:
        ret_img = horizontal_cross_correlation_1d(img, kernel)
    else:
        rnum, cnum = kernel.shape
        if cnum == 1:
            ret_img = vertical_cross_correlation_1d(img, kernel)
        else:
            ret_img = img

    return ret_img


def cross_correlation_2d(img, kernel):
    krnum, kcnum = kernel.shape
    irnum, icnum = img.shape
    temp_img = img.copy().astype(np.float64)
    ret_img = np.zeros((irnum, icnum), dtype=np.float64)
    krmid = int(krnum/2)
    kcmid = int(kcnum/2)
    temp_img = np.pad(temp_img, ((krmid, krmid), (kcmid, kcmid)), mode='edge')

    for i in range(krnum):
        for j in range(kcnum):
            ret_img += temp_img[i:i+irnum, j:j+icnum] * kernel[i][j]

    return ret_img


def get_gaussian_filter_1d(size, sigma):
    mid = int(size/2)
    result = np.zeros(size, dtype=np.float64)
    result[mid] = div_sum = 1.
    for i in range(1, mid+1):
        result[mid-i] = math.exp((-1)*(i*i)/(2*sigma*sigma))
        result[mid+i] = math.exp((-1)*(i*i)/(2*sigma*sigma))
        div_sum += 2*result[mid-i]
    for i in range(size):
        result[i] /= div_sum
    return result


def get_gaussian_filter_2d(size, sigma):
    gaussian_1d = get_gaussian_filter_1d(size, sigma)
    return np.outer(gaussian_1d, np.array([gaussian_1d]).T)


def compute_image_gradient(img):
    filter_1 = np.array([1, 2, 1])
    filter_1_tp = np.array([filter_1]).T
    filter_2 = np.array([1, 0, -1])
    filter_2_tp = np.array([filter_2]).T

    img_grad_x = cross_correlation_1d(img.astype(np.float64), filter_2)
    img_grad_x = cross_correlation_1d(img_grad_x, filter_1_tp)
    img_grad_y = cross_correlation_1d(img.astype(np.float64), filter_1)
    img_grad_y = cross_correlation_1d(img_grad_y, filter_2_tp)

    img_dir = np.arctan2(img_grad_y, img_grad_x)

    img_mag = np.power(img_grad_x, 2) + np.power(img_grad_y, 2)
    img_mag = np.sqrt(img_mag)

    if np.max(img_mag) > 255.0:
        img_mag *= np.float64(255.0)/np.max(img_mag)

    return img_mag, img_dir


def non_maximum_suppression_dir(mag, dir):
    rnum, cnum = mag.shape
    dir_t = dir.copy()
    dir_t *= 180. / np.pi
    ret_img = np.zeros((rnum, cnum), dtype=np.uint8)
    for i in range(rnum):
        for j in range(cnum):
            if dir_t[i][j] < 0:
                dir_t[i][j] += 180.
            posval = 255
            negval = 255

            if dir_t[i][j] < 22.5 or dir_t[i][j] >= 157.5:
                if j != cnum-1:
                    posval = mag[i][j+1]
                if j != 0:
                    negval = mag[i][j-1]
            elif 22.5 <= dir_t[i][j] < 67.5:
                if i != rnum-1 and j != cnum-1:
                    posval = mag[i+1][j+1]
                if i != 0 and j != 0:
                    negval = mag[i-1][j-1]
            elif 67.5 <= dir_t[i][j] < 112.5:
                if i != rnum-1:
                    posval = mag[i+1][j]
                if i != 0:
                    negval = mag[i-1][j]
            elif 112.5 <= dir_t[i][j] < 157.5:
                if i != rnum-1 and j != 0:
                    posval = mag[i+1][j-1]
                if i != 0 and j != cnum-1:
                    negval = mag[i-1][j+1]

            if mag[i][j] > posval and mag[i][j] >= negval:
                ret_img[i][j] = mag[i][j]

    return ret_img


def compute_corner_response(img):
    rnum, cnum = img.shape
    kappa = 0.04
    wsize = 5

    filter_1 = np.array([1, 2, 1])
    filter_1_tp = np.array([filter_1]).T
    filter_2 = np.array([1, 0, -1])
    filter_2_tp = np.array([filter_2]).T

    img_grad_x = cross_correlation_1d(img.astype(np.float64), filter_2)
    img_grad_x = cross_correlation_1d(img_grad_x, filter_1_tp)
    img_grad_y = cross_correlation_1d(img.astype(np.float64), filter_1)
    img_grad_y = cross_correlation_1d(img_grad_y, filter_2_tp)

    wmid = int(wsize/2)

    img_res = np.zeros((rnum, cnum), dtype=np.float64)

    for i in range(wmid, rnum-wmid):
        for j in range(wmid, cnum-wmid):
            img_grad_x_win = img_grad_x[i-wmid:i-wmid+wsize, j-wmid:j-wmid+wsize]
            img_grad_y_win = img_grad_y[i-wmid:i-wmid+wsize, j-wmid:j-wmid+wsize]

            sum_xx = np.sum(img_grad_x_win*img_grad_x_win)
            sum_xy = np.sum(img_grad_x_win*img_grad_y_win)
            sum_yy = np.sum(img_grad_y_win*img_grad_y_win)

            temp = (sum_xx*sum_yy - sum_xy*sum_xy) - kappa * (sum_xx + sum_yy) * (sum_xx + sum_yy)
            if temp > 0:
                img_res[i][j] = temp

    res_min = np.min(img_res)
    res_max = np.max(img_res)
    img_res -= res_min
    img_res /= (res_max - res_min)

    return img_res


def non_maximum_suppression_win(R, winSize):
    winMid = int(winSize/2)
    rnum, cnum = R.shape
    R_ret = R.copy()
    R_pad = np.pad(R_ret, ((winMid, winMid), (winMid, winMid)), mode='edge')

    for i in range(rnum):
        for j in range(cnum):
            maxval = np.max(R_pad[i:i+winSize, j:j+winSize])
            if R_ret[i][j] < maxval or R_ret[i][j] <= 0.1:
                R_ret[i][j] = 0

    return R_ret
