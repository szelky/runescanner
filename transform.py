import cv2
import numpy as np

def order_points(pts):
    # Create 4 arr each has 2 arr in it which already 0
    # [[0, 0],     0. top-left
    #  [0, 0],     1. top-right
    #  [0, 0],     2. bottom-right
    #  [0, 0]]     3. bottom-left
    
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)] # top-left will have min sums
    rect[2] = pts[np.argmax(s)] # bottom-right will have max sums

    d = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(d)] # top-right will have min diff
    rect[3] = pts[np.argmax(d)] # bottom-left will have max diff

    return rect

def four_point_transform(image, pts):
    # get ordered rect to set each corner
    # 0. top-left       > tr
    # 1. top-right      > tl
    # 2. bottom-right   > br
    # 3. bottom-left    > bl
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # now we compute the width and height of image
    # by using Euclidean Distance which is
    # sqrt((x2-x1)**2 + (y2-y1)**2)
    # and np.linalg.norm is just doing this formula
    widthTop = np.linalg.norm(tr - tl)
    widthBottom = np.linalg.norm(br - bl)
    maxWidth = max(int(widthTop), int(widthBottom))

    heightLeft = np.linalg.norm(tl - bl)
    heightRight = np.linalg.norm(tr - br)
    maxHeight = max(int(heightLeft), int(heightRight))

    # now we can create our destination points
    dst = np.array(
        [[0, 0],
         [maxWidth - 1, 0],
         [maxWidth - 1, maxHeight - 1],
         [0, maxHeight - 1]],
         dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped
