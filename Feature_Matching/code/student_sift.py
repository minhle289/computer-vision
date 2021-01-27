import numpy as np
import cv2
import math


def get_features(image, x, y, feature_width, scales=None):
    """
    JR Writes: To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    maximal points you may need to implement a more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)


    Below for advanced implementation:

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    assert image.ndim == 2, 'Image must be grayscale'
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    # If you choose to implement rotation invariance, enabling it should not    #
    # decrease your matching accuracy.                                          #
    #############################################################################

    width = x.shape[0]
    #Get gradients using Sobel kernels
    gradient_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

    thetas = np.rad2deg(np.arctan2(gradient_y, gradient_x))
    magnitudes = np.sqrt(np.square(gradient_x) + np.square(gradient_y))


    assert feature_width % 4 == 0

    window_offset = int(feature_width/2)

    cell_size = int(feature_width/4)

    output =[]

    for k1 in range(int(width)):
        cur_x = int(x[k1])
        cur_y = int(y[k1])

        window_mag = magnitudes[cur_y-window_offset : cur_y+window_offset, cur_x-window_offset : cur_x+window_offset]
        window_theta = thetas[cur_y-window_offset : cur_y+window_offset, cur_x-window_offset : cur_x+window_offset]

        features = []

        for i in range(0, feature_width, cell_size):
            for j in range(0, feature_width, cell_size):
                mag_bin = window_mag[i : i+cell_size, j : j+cell_size]
                theta_bin = window_theta[i : i+cell_size, j : j+cell_size]

                hist, bin_edges = np.histogram(theta_bin, bins = 8, range = (-180,180), weights = mag_bin)
                features.extend(hist)

        output.append(np.array(features))

    fv = np.array(output)
    #fv = fv/np.linalg.norm(fv+0.0001)
    fv = fv**0.85

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return fv
