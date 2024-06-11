import numpy as np

class Keystone:
    def __init__(self):
        pass

    def getPerspectiveTransform(self, imgPoints, objPoints):

        # Initialize matrices A and B
        A = np.zeros((8, 8), dtype=np.float32)
        B = np.zeros((8, 1), dtype=np.float32)

        # Fill matrices A and B based on points
        for i in range(4):
            A[i*2, 0] = imgPoints[i][0]
            A[i*2, 1] = imgPoints[i][1]
            A[i*2, 2] = 1
            A[i*2, 6] = -imgPoints[i][0] * objPoints[i][0]
            A[i*2, 7] = -imgPoints[i][1] * objPoints[i][0]

            A[i*2+1, 3] = imgPoints[i][0]
            A[i*2+1, 4] = imgPoints[i][1]
            A[i*2+1, 5] = 1
            A[i*2+1, 6] = -imgPoints[i][0] * objPoints[i][1]
            A[i*2+1, 7] = -imgPoints[i][1] * objPoints[i][1]

            # destination points
            B[i*2] = objPoints[i][0]
            B[i*2+1] = objPoints[i][1]

        # least squares
        X = np.linalg.lstsq(A, B, rcond=None)[0]

        # Construct the transformation matrix M
        M = np.zeros((3, 3), dtype=np.float32)
        M[0, 0] = X[0]
        M[0, 1] = X[1]
        M[0, 2] = X[2]
        M[1, 0] = X[3]
        M[1, 1] = X[4]
        M[1, 2] = X[5]
        M[2, 0] = X[6]
        M[2, 1] = X[7]
        M[2, 2] = 1

        return M

    def warpPerspective(self, src, M, dsize):
        dst = np.zeros((dsize[1], dsize[0], 3), dtype=np.uint8)
        M_inv = np.linalg.inv(M)

        for y in range(dsize[1]):
            for x in range(dsize[0]):
                # Apply transformation to get corresponding source image pixel
                pt = np.dot(M_inv, np.array([x, y, 1]))
                pt = pt / pt[2]

                # Check if transformed point is within source image boundaries
                if 0 <= pt[0] < src.shape[1] and 0 <= pt[1] < src.shape[0]:
                    # Interpolate pixel value from source image
                    src_x = int(pt[0])
                    src_y = int(pt[1])

                    # Bilinear interpolation
                    x_diff = pt[0] - src_x
                    y_diff = pt[1] - src_y

                    top_left = src[src_y, src_x] * (1 - x_diff) * (1 - y_diff)
                    top_right = src[src_y, src_x + 1] * x_diff * (1 - y_diff)
                    bottom_left = src[src_y + 1, src_x] * (1 - x_diff) * y_diff
                    bottom_right = src[src_y + 1, src_x + 1] * x_diff * y_diff

                    dst[y, x] = top_left + top_right + bottom_left + bottom_right

        return dst
