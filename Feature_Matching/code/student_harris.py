import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_interest_points(image, feature_width):
    """

    JR adds: to ensure compatability with project 4A, you simply need to use
    this function as a wrapper for your 4A code.  Guidelines below left
    for historical reference purposes.

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   feature_width: integer representing the local feature width in pixels.

    Returns:
    -   x: A numpy array of shape (N,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N,) containing y-coordinates of interest points
    -   confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point
    -   scales (optional): A numpy array of shape (N,) containing the scale at each
            interest point
    -   orientations (optional): A numpy array of shape (N,) containing the orientation
            at each interest point
    """
    confidences, scales, orientations = None, None, None
    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                                                      #
    #############################################################################
    Ix = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

    #2/ Compute Ixx, Iyy, Ixy
    Ixx,Iyy,Ixy = Ix**2, Iy**2, Ix * Iy


    #I found the GaussianBlur returns a better output
    Ixx2 = cv2.GaussianBlur(Ixx, (5,5), 5)
    Iyy2 = cv2.GaussianBlur(Iyy, (5,5), 5)
    Ixy2 = cv2.GaussianBlur(Ixy, (5,5), 5)

    #4 Calculate R
    det = (Ixx2*Iyy2) - (Ixy2**2)
    trace = Ixx2 + Iyy2
    Rvals = det - 0.04*(trace**2)

 
    threshold = 0.01 * Rvals.max()
    height, width = np.shape(Rvals)
    Rvals_filter = []

    for x in range(height):
        for y in range(width):
            if Rvals[x][y] > threshold:
                Rvals_filter.append((x,y,Rvals.item(x,y)))

    Rvals_filter = np.array((Rvals_filter))
    
    #sort by response strength (highest to lowest)
    Rvals_filter= Rvals_filter[np.argsort(Rvals_filter[:,2])]
    Rvals_filter= np.flipud(Rvals_filter)
    coord = Rvals_filter[:,0:2]
    
    #fill in values for the minRad array
    #compute min distance for each point
    minRad=[]
    for i in range(1, len(Rvals_filter)):
        dist = np.linalg.norm(coord[0:i] - coord[i], axis=1)
        min = np.argsort(dist)[0]
        minRad.append((Rvals_filter[min][0], Rvals_filter[min][1], dist.min()))

    #sort from largest distance
    minRad.sort(key=lambda minRad:minRad[2], reverse=True)

    #always pick the highest response
    numPts = 3250
    x = np.zeros((numPts,))
    y = np.zeros((numPts,))

    x[0]= Rvals_filter[0][1]
    y[0]= Rvals_filter[0][0]

    for i in range(1, numPts):
        x[i] = minRad[i-1][1]
        y[i] = minRad[i-1][0]
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return x,y, confidences, scales, orientations