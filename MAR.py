# MAR.py

import numpy as np

def mouth_aspect_ratio(mouth):
    # Compute the Euclidean distances between the vertical mouth landmarks (x, y)-coordinates
    A = np.linalg.norm(mouth[2] - mouth[10])
    B = np.linalg.norm(mouth[4] - mouth[8])

    # Compute the Euclidean distance between the horizontal mouth landmarks (x, y)-coordinates
    C = np.linalg.norm(mouth[0] - mouth[6])

    # Compute the mouth aspect ratio
    mar = (A + B) / (2.0 * C)
    return mar
