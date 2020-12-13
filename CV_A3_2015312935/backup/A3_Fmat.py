import numpy as np
import cv2 as cv
import random
import compute_avg_reproj_error as cmerror

size_x = 480
size_y = 640


def compute_F_raw(M):
    mnum = M.shape[0]
    A = np.hstack(
        [(M[:, 0] * M[:, 2]).reshape(mnum, -1), (M[:, 0] * M[:, 3]).reshape(mnum, -1), M[:, 0].reshape(mnum, -1),
         (M[:, 1] * M[:, 2]).reshape(mnum, -1), (M[:, 1] * M[:, 3]).reshape(mnum, -1), M[:, 1].reshape(mnum, -1),
         M[:, 2].reshape(mnum, -1), M[:, 3].reshape(mnum, -1), np.ones((mnum, 1))])
    U, D, Vt = np.linalg.svd(A)

    V = Vt.transpose(1, 0)
    Ftemp = V[:, 8]
    F = np.reshape(Ftemp, (3, 3))

    return F


def compute_F_norm(M):
    mnum = M.shape[0]
    X = M.T[0:2]
    Xp = M.T[2:4]

    Xxm, Xym, Xpxm, Xpym = size_x/2, size_y/2, size_x/2, size_y/2
    Xx = X[0] - Xxm
    Xy = X[1] - Xym

    Xpx = Xp[0] - Xpxm
    Xpy = Xp[1] - Xpym


    Mp = np.hstack([Xx.reshape(mnum,-1)*2/size_x, Xy.reshape(mnum,-1)*2/size_y, Xpx.reshape(mnum,-1)*2/size_x, Xpy.reshape(mnum,-1)*2/size_y])

    A = np.hstack(
        [(Mp[:, 0] * Mp[:, 2]).reshape(mnum, -1), (Mp[:, 0] * Mp[:, 3]).reshape(mnum, -1), Mp[:, 0].reshape(mnum, -1),
         (Mp[:, 1] * Mp[:, 2]).reshape(mnum, -1), (Mp[:, 1] * Mp[:, 3]).reshape(mnum, -1), Mp[:, 1].reshape(mnum, -1),
         Mp[:, 2].reshape(mnum, -1), Mp[:, 3].reshape(mnum, -1), np.ones((mnum, 1))])

    U, D, Vt = np.linalg.svd(A)
    V = Vt.transpose(1, 0)
    Ftemp = V[:, 8]
    F = np.reshape(Ftemp, (3, 3))

    U, D, Vt = np.linalg.svd(F)
    Dp = np.array([[D[0], 0, 0], [0, D[1], 0], [0, 0, 0]])
    F = np.dot(np.dot(U, Dp), Vt)


    T1 = np.array([[2/size_x, 0, -Xxm*2/size_x], [0, 2/size_y, -Xym*2/size_y], [0, 0, 1]])

    F = np.dot(np.dot(T1.T, F), T1)

    return F


def compute_F_mine(M):
    mnum = M.shape[0]

    X = M.T[0:2]
    X = np.vstack([X, np.ones(mnum)])
    Xp = M.T[2:4]
    Xp = np.vstack([Xp, np.ones(mnum)])

    iters = 10000
    tempM = np.ones((8, 4), dtype=np.float64)
    retF = np.zeros((3,3))
    minval = 640*640*640*640*2
    for i in range(iters):
        nums = random.sample(range(0, mnum),20)
        for j in range(8):
            tempM[j] = M[nums[j]]
        tempF = compute_F_norm(tempM)
        LX = np.matmul(tempF, X)
        tX = Xp * LX
        disX = tX.sum(axis=0)**2/(LX[0]**2+LX[1]**2)

        LXp = np.matmul(tempF.T, Xp)
        tXp = X * LXp
        disXp = tXp.sum(axis=0)**2/(LXp[0]**2+LXp[1]**2)

        tsum = np.sum(disX) + np.sum(disXp)
        if tsum < minval:
            retF = tempF.copy()
            minval = tsum

    return retF


def visualize_line(imgp, F, X, color, th):
    tempX = np.append(X, [1.])
    Lx = np.matmul(F, tempX)
    for y in range(size_y):
        for x in range(size_x):
            if np.abs((Lx[0]*x+Lx[1]*y+Lx[2])/np.sqrt(Lx[0]**2+Lx[1]**2)) < th:
                imgp[y][x] = color
    return


if __name__ == '__main__':
    M_temple = np.loadtxt('temple_matches.txt')
    size_x = 480
    size_y = 640

    F_temple_raw = compute_F_raw(M_temple)
    F_temple_norm = compute_F_norm(M_temple)
    F_temple_mine = compute_F_mine(M_temple)

    print("Average Reprojection Errors (temple1.png and temple2.png)")
    print("   Raw =", cmerror.compute_avg_reproj_error(M_temple, F_temple_raw))
    print("   Norm =", cmerror.compute_avg_reproj_error(M_temple, F_temple_norm))
    print("   Mine =", cmerror.compute_avg_reproj_error(M_temple, F_temple_mine))

    M_house = np.loadtxt('house_matches.txt')
    size_x = 384
    size_y = 288

    F_house_raw = compute_F_raw(M_house)
    F_house_norm = compute_F_norm(M_house)
    F_house_mine = compute_F_mine(M_house)

    print("Average Reprojection Errors (house1.jpg and house2.jpg)")
    print("   Raw =", cmerror.compute_avg_reproj_error(M_house, F_house_raw))
    print("   Norm =", cmerror.compute_avg_reproj_error(M_house, F_house_norm))
    print("   Mine =", cmerror.compute_avg_reproj_error(M_house, F_house_mine))

    M_library = np.loadtxt('library_matches.txt')
    size_x = 512
    size_y = 384

    F_library_raw = compute_F_raw(M_library)
    F_library_norm = compute_F_norm(M_library)
    F_library_mine = compute_F_mine(M_library)

    print("Average Reprojection Errors (library1.jpg and library2.jpg)")
    print("   Raw =", cmerror.compute_avg_reproj_error(M_library, F_library_raw))
    print("   Norm =", cmerror.compute_avg_reproj_error(M_library, F_library_norm))
    print("   Mine =", cmerror.compute_avg_reproj_error(M_library, F_library_mine))

    img_temple = cv.imread('temple1.png')
    imgp_temple = cv.imread('temple2.png')
    size_x, size_y = img_temple.shape[1], img_temple.shape[0]
    mnum = M_temple.shape[0]

    while True:
        img_tmp = img_temple.copy()
        imgp_tmp = imgp_temple.copy()
        nums = random.sample(range(0, mnum), 3)
        visualize_line(imgp_tmp, F_temple_mine, M_temple[nums[0]][0:2], [0, 0, 255], 1)
        visualize_line(img_tmp, F_temple_mine.T, M_temple[nums[0]][2:4], [0, 0, 255], 1)
        visualize_line(imgp_tmp, F_temple_mine, M_temple[nums[1]][0:2], [0, 255, 0], 1)
        visualize_line(img_tmp, F_temple_mine.T, M_temple[nums[1]][2:4], [0, 255, 0], 1)
        visualize_line(imgp_tmp, F_temple_mine, M_temple[nums[2]][0:2], [255, 0, 0], 1)
        visualize_line(img_tmp, F_temple_mine.T, M_temple[nums[2]][2:4], [255, 0, 0], 1)
        cv.imshow('Press q to terminate, other keys for refreshing', np.concatenate((img_tmp, imgp_tmp), axis=1))
        key = cv.waitKey(0)
        if key == ord('q'):
            cv.destroyAllWindows()
            break

    img_house = cv.imread('house1.jpg')
    imgp_house = cv.imread('house2.jpg')
    size_x, size_y = img_house.shape[1], img_house.shape[0]
    mnum = M_house.shape[0]

    while True:
        img_tmp = img_house.copy()
        imgp_tmp = imgp_house.copy()
        nums = random.sample(range(0, mnum), 3)
        visualize_line(imgp_tmp, F_house_mine, M_house[nums[0]][0:2], [0, 0, 255], 1)
        visualize_line(img_tmp, F_house_mine.T, M_house[nums[0]][2:4], [0, 0, 255], 1)
        visualize_line(imgp_tmp, F_house_mine, M_house[nums[1]][0:2], [0, 255, 0], 1)
        visualize_line(img_tmp, F_house_mine.T, M_house[nums[1]][2:4], [0, 255, 0], 1)
        visualize_line(imgp_tmp, F_house_mine, M_house[nums[2]][0:2], [255, 0, 0], 1)
        visualize_line(img_tmp, F_house_mine.T, M_house[nums[2]][2:4], [255, 0, 0], 1)
        cv.imshow('Press q to terminate, other keys for refreshing', np.concatenate((img_tmp, imgp_tmp), axis=1))
        key = cv.waitKey(0)
        if key == ord('q'):
            cv.destroyAllWindows()
            break

    img_library = cv.imread('library1.jpg')
    imgp_library = cv.imread('library2.jpg')
    size_x, size_y = img_library.shape[1], img_library.shape[0]
    mnum = M_library.shape[0]

    while True:
        img_tmp = img_library.copy()
        imgp_tmp = imgp_library.copy()
        nums = random.sample(range(0, mnum), 3)
        visualize_line(imgp_tmp, F_library_mine, M_library[nums[0]][0:2], [0, 0, 255], 1)
        visualize_line(img_tmp, F_library_mine.T, M_library[nums[0]][2:4], [0, 0, 255], 1)
        visualize_line(imgp_tmp, F_library_mine, M_library[nums[1]][0:2], [0, 255, 0], 1)
        visualize_line(img_tmp, F_library_mine.T, M_library[nums[1]][2:4], [0, 255, 0], 1)
        visualize_line(imgp_tmp, F_library_mine, M_library[nums[2]][0:2], [255, 0, 0], 1)
        visualize_line(img_tmp, F_library_mine.T, M_library[nums[2]][2:4], [255, 0, 0], 1)
        cv.imshow('Press q to terminate, other keys for refreshing', np.concatenate((img_tmp, imgp_tmp), axis=1))
        key = cv.waitKey(0)
        if key == ord('q'):
            cv.destroyAllWindows()
            break