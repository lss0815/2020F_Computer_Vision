
import cv2

img = cv2.imread('island.jpg', cv2.IMREAD_COLOR)
cv2.imshow('ISLAND', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
