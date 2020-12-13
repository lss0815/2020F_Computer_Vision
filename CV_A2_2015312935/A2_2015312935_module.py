import numpy as np


def get_inv_nearest_value(x, y, M_inv, img):
    size_ir, size_ic = img.shape
    mid_ir, mid_ic = int(size_ir/2), int(size_ic/2)

    rx = M_inv[0][0]*(x-mid_ic) + M_inv[0][1]*(y-mid_ir) + M_inv[0][2]*1
    ry = M_inv[1][0]*(x-mid_ic) + M_inv[1][1]*(y-mid_ir) + M_inv[1][2]*1

    result_x, result_y = 0, 0

    if np.floor(rx) == rx:
        result_x = int(rx)
    elif np.abs(np.floor(rx) - rx) < np.abs(np.ceil(rx) - rx):
        result_x = int(np.floor(rx))
    else:
        result_x = int(np.ceil(rx))
    result_x = max(result_x, (-1)*mid_ic)
    result_x = min(result_x, mid_ic)

    if np.floor(ry) == ry:
        result_y = int(ry)
    elif np.abs(np.floor(ry) - ry) < np.abs(np.ceil(ry) - ry):
        result_y = int(np.floor(ry))
    else:
        result_y = int(np.ceil(ry))
    result_y = max(result_y, (-1)*mid_ir)
    result_y = min(result_y, mid_ir)

    return img[result_y+mid_ir][result_x+mid_ic]


def get_transformed_image(img, M):
    size_ir, size_ic = img.shape
    mid_ir, mid_ic = int(size_ir/2), int(size_ic/2)
    ret_img = np.zeros((size_ir, size_ic), dtype=np.uint8)
    ret_img.fill(255)

    M_inv = np.linalg.inv(M)

    cx = int(np.floor(M[0][2] * 1))
    cy = int(np.floor(M[1][2] * 1))

    for i in range(max(0,mid_ir+cy-150), min(size_ir-1,mid_ir+cy+150)):
        for j in range(max(mid_ic+cx-150,0), min(mid_ic+cx+150, size_ic-1)):
            ret_img[i][j] = get_inv_nearest_value(j, i, M_inv, img)

    return ret_img
