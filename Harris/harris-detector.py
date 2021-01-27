import cv2
import numpy as np
import math

def HarrisDetector(img,k = 0.04):
    '''
    Args:
    
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale or color (your choice)
                (i recommmend greyscale)
    -   k: k value for Harris detector

    Returns:
    -   R: A numpy array of shape (m,n) containing R values of interest points
   '''
    #Color to Grayscale
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    #1/ Calculate derivatives using Sobel
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    Ix = cv2.filter2D(img, cv2.CV_32F, kernel_x)
    Iy = cv2.filter2D(img, cv2.CV_32F, kernel_y)

    #2/ Compute Ixx, Iyy, Ixy
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    #3/ Convolve each image with Gaussian kernel (window 5)
    #Gfilter = cv2.getGaussianKernel(ksize = 5, sigma = 10)
    #Ixx2 = cv2.filter2D(Ixx, -1, Gfilter)
    #Iyy2 = cv2.filter2D(Iyy, -1, Gfilter)
    #Ixy2 = cv2.filter2D(Ixy, -1, Gfilter)

    #I found the GaussianBlur returns a better output
    Ixx2 = cv2.GaussianBlur(Ixx, (5,5), 5)
    Iyy2 = cv2.GaussianBlur(Iyy, (5,5), 5)
    Ixy2 = cv2.GaussianBlur(Ixy, (5,5), 5)

    #4 Calculate R    
    det = (Ixx2*Iyy2) - (Ixy2**2)
    trace = Ixx2 + Iyy2
    R = det - k*(trace**2)

    #throw away reponses that less than 1% max
    R[R<0.01*R.max()] = 0

    return R
    
 
def SuppressNonMax(Rvals, numPts):
    '''
    Args:
    
    -   Rvals: A numpy array of shape (m,n,1), containing Harris response values
    -   numPts: the number of responses to return

    Returns:

     x: A numpy array of shape (N,) containing x-coordinates of interest points
     y: A numpy array of shape (N,) containing y-coordinates of interest points
     confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point
   '''
    #sort array in decreasing orders of respond
    threshold = 0.01*Rvals.max()
    height, width = np.shape(Rvals)
    array = []
    array_dist = []
    
    array =[[x, y, Rvals[y][x]] for y in range(height) for x in range(width)]

    #filter out those witihin 1% of max(R)
    array_filter = []
    for pixel in array:
        if (pixel[2]>threshold):
            array_filter.append(pixel)

    #sort by response strength
    array_filter.sort(key = lambda x: x[2], reverse = True)

    #compute min distance for each point
    for i in range(1, len(array_filter)):
        dist = [compute_dist(array_filter[i], array_filter[x]) for x in range(0, i)]
        dist.sort()
        array_dist.append([array_filter[i][0], array_filter[i][1], dist[0]])

    #sort from largest distance
    array_dist.sort(key = lambda x: x[2], reverse = True)

    x = np.zeros((numPts,))
    y = np.zeros((numPts,))

    #always pick the highest response
    x[0] = array_filter[0][0]
    y[0] = array_filter[0][1]
    
    for i in range(1, numPts):
        x[i] = array_dist[i-1][0]
        y[i] = array_dist[i-1][1]   
    return x, y
    

def compute_dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
