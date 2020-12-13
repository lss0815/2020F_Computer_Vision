import numpy as np
import cv2 as cv
import random
import time
import A2_2015312935_module as My


def get_hamming_distance(des1, des2):
    r = (1 << np.arange(8))[:, None]
    return np.count_nonzero((des1 & r) != (des2 & r))


def get_BF(kp1, des1, kp2, des2):
    len1 = len(kp1)
    len2 = len(kp2)
    matches = []
    for i in range(len1):
        for j in range(len2):
            matches.append(cv.DMatch(i, j, get_hamming_distance(des1[i], des2[j])))
    return matches


def compute_homography(srcP, destP):
    pnum = len(srcP)
    matrix_id = np.zeros((3,3), dtype=np.float64)
    matrix_id[0][0] = 1.
    matrix_id[1][1] = 1.
    matrix_id[2][2] = 1.

    xsum, ysum = 0., 0.
    for i in range(pnum):
        xsum += srcP[i][0]
        ysum += srcP[i][1]
    max_dis = 0.
    for i in range(pnum):
        srcP[i][0] -= xsum/pnum
        srcP[i][1] -= ysum/pnum
        if (srcP[i][0])*(srcP[i][0]) + (srcP[i][1])*(srcP[i][1]) > max_dis * max_dis:
            max_dis = np.sqrt((srcP[i][0])*(srcP[i][0]) + (srcP[i][1])*(srcP[i][1]))
    for i in range(pnum):
        srcP[i][0] *= np.sqrt(2.)/max_dis
        srcP[i][1] *= np.sqrt(2.)/max_dis
    matrix_temp1 = matrix_id.copy()
    matrix_temp1[0][2] = (-1)*xsum/pnum
    matrix_temp1[1][2] = (-1)*ysum/pnum
    matrix_temp2 = matrix_id.copy()
    matrix_temp2[0][0] = np.sqrt(2.)/max_dis
    matrix_temp2[1][1] = np.sqrt(2.)/max_dis
    matrix_ts = np.matmul(matrix_temp2, matrix_temp1)

    xsum, ysum = 0., 0.
    for i in range(pnum):
        xsum += destP[i][0]
        ysum += destP[i][1]
    max_dis = 0.
    for i in range(pnum):
        destP[i][0] -= xsum / pnum
        destP[i][1] -= ysum / pnum
        if (destP[i][0]) * (destP[i][0]) + (destP[i][1]) * (destP[i][1]) > max_dis * max_dis:
            max_dis = np.sqrt((destP[i][0]) * (destP[i][0]) + (destP[i][1]) * (destP[i][1]))
    for i in range(pnum):
        destP[i][0] *= np.sqrt(2.) / max_dis
        destP[i][1] *= np.sqrt(2.) / max_dis
    matrix_temp1 = matrix_id.copy()
    matrix_temp1[0][2] = (-1)*xsum/pnum
    matrix_temp1[1][2] = (-1)*ysum/pnum
    matrix_temp2 = matrix_id.copy()
    matrix_temp2[0][0] = np.sqrt(2.)/max_dis
    matrix_temp2[1][1] = np.sqrt(2.)/max_dis
    matrix_td = np.matmul(matrix_temp2, matrix_temp1)

    matrix_A = np.zeros((pnum*2, 9), dtype=np.float64)
    for i in range(pnum):
        matrix_A[2*i] = np.array([(-1)*srcP[i][0], (-1)*srcP[i][1], -1, 0, 0, 0, srcP[i][0]*destP[i][0], srcP[i][1]*destP[i][0], destP[i][0]])
        matrix_A[2*i+1] = np.array([0, 0, 0, (-1)*srcP[i][0], (-1)*srcP[i][1], -1, srcP[i][0]*destP[i][1], srcP[i][1]*destP[i][1], destP[i][1]])

    U, D, Vt = np.linalg.svd(matrix_A)
    V = Vt.transpose(1, 0)
    matrix_htemp = V[:,8]
    matrix_H = np.reshape(matrix_htemp, (3,3))

    matrix_temp = np.matmul(np.linalg.inv(matrix_td), matrix_H)
    matrix_result = np.matmul(matrix_temp, matrix_ts)
    return matrix_result


def compute_homography_ransac(srcP, destP, th):
    iters = 10000
    tsP = np.zeros((4,2),dtype=np.float64)
    tdP = tsP.copy()
    max_cnt = 0
    max_matrix_H = np.zeros((3,3), dtype=np.float64)
    for i in range(iters):
        nums = random.sample(range(0, 500),20)
        for j in range(4):
            tsP[j] = srcP[nums[j]]
            tdP[j] = destP[nums[j]]
        matrix_H = compute_homography(tsP, tdP)
        cnt = 0
        homo_temp = np.zeros((3), dtype=np.float64)
        homo_temp[2] = 1
        for j in range(500):
            homo_temp[0] = srcP[j][0]
            homo_temp[1] = srcP[j][1]
            pos_temp = np.matmul(matrix_H, homo_temp)
            if(pos_temp[0] - destP[j][0])*(pos_temp[0] - destP[j][0]) + (pos_temp[1] - destP[j][1])*(pos_temp[1] - destP[j][1]) < th*th:
                cnt += 1
        if cnt > max_cnt:
            max_cnt = cnt
            max_matrix_H = matrix_H.copy()
    print(max_cnt)
    return max_matrix_H


if __name__ == '__main__':
    img_desk = cv.imread('cv_desk.png', cv.IMREAD_GRAYSCALE)
    img_cover = cv.imread('cv_cover.jpg', cv.IMREAD_GRAYSCALE)

    orb = cv.ORB_create()
    kp_cover = orb.detect(img_cover, None)
    kp_cover, des_cover = orb.compute(img_cover, kp_cover)

    kp_desk = orb.detect(img_desk, None)
    kp_desk, des_desk = orb.compute(img_desk, kp_desk)

    matches_my = get_BF(kp_desk, des_desk, kp_cover, des_cover )
    matches_my = sorted(matches_my, key=lambda x: x.distance)

    img_bfmatch_my = cv.drawMatches(img_desk, kp_desk, img_cover, kp_cover, matches_my[:10],None,flags=2)

    cv.imshow('Press any key to continue next image', img_bfmatch_my)
    cv.waitKey(0)
    cv.destroyAllWindows()

    matches_my = get_BF(kp_cover, des_cover, kp_desk, des_desk)
    matches_my = sorted(matches_my, key=lambda x: x.distance)

    srcP = np.zeros((15,2), dtype=np.float64)
    destP = np.zeros((15,2), dtype=np.float64)
    for i in range(15):
        srcP[i] = np.array(kp_cover[matches_my[i].queryIdx].pt)
        destP[i] = np.array(kp_desk[matches_my[i].trainIdx].pt)

    homo_norm = compute_homography(srcP, destP)
    img_norm = cv.warpPerspective(img_cover, homo_norm, (img_desk.shape[1], img_desk.shape[0]))

    cv.imshow('Press any key to continue next image', img_norm)
    cv.waitKey(0)
    cv.destroyAllWindows()

    img_desk_norm = img_desk.copy()

    for i in range(img_desk.shape[0]):
        for j in range(img_desk.shape[1]):
            if(img_norm[i][j] != 0):
                img_desk_norm[i][j] = img_norm[i][j]

    cv.imshow('Press any key to continue next image', img_desk_norm)
    cv.waitKey(0)
    cv.destroyAllWindows()

    srcP = np.zeros((500, 2), dtype=np.float64)
    destP = np.zeros((500, 2), dtype=np.float64)
    for i in range(500):
        srcP[i] = np.array(kp_cover[matches_my[i].queryIdx].pt)
        destP[i] = np.array(kp_desk[matches_my[i].trainIdx].pt)

    stime = time.time()
    homo_ransac = compute_homography_ransac(srcP, destP, 1)
    print(time.time() - stime)

    img_ransac = cv.warpPerspective(img_cover, homo_ransac, (img_desk.shape[1], img_desk.shape[0]))

    cv.imshow('Press any key to continue next image', img_ransac)
    cv.waitKey(0)
    cv.destroyAllWindows()