# Minh Le
# Project 3
import cv2
import numpy as np
import math

# Hough Transformation
def myHoughLines(img, rho, theta, threshold):
    height, width = img.shape

    #calculate the max number of posible thetas
    num_thetas = math.ceil(math.pi/theta) + 1
    thetaVals = np.linspace(0, math.pi, num_thetas)

    #match OpenCV output, round theta to 7 decimals
    thetaVals = np.ceil(thetaVals*(10**7))/10**7
        
    #longest distance
    diagonal = math.sqrt(width * width + height * height)
    num_rhos  = math.ceil((diagonal*2)/rho) + 1
    rhoVals = np.linspace(-diagonal,diagonal, num_rhos)
    
    line_votes = np.zeros((num_rhos, num_thetas), 'uint8')
    
    for i in range(height):
        for j in range(width):

            #if non-zero pixel
            if img[i][j] != 0:
                for t_index in range(num_thetas):
                    t = thetaVals[t_index]
                    rhoVal = j*math.cos(t) + i*math.sin(t)

                    #get the closest rho available, use index in rohVals instead to accomodate negative values
                    #also with index no need to round off
                    r_index = np.searchsorted(rhoVals, rhoVal)
                      
                    #add 1 vote
                    line_votes[r_index][t_index] += 1

    output = []
    
    for i in range(num_rhos):
        for j in range(num_thetas):
            if (line_votes[i][j] >= threshold):
                output.append([rhoVals[i], thetaVals[j]])
    output = np.array(output).reshape((len(output),1,2))
    return output

# Sobel Filter
def sobel(img):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    gradient_x = cv2.filter2D(img, cv2.CV_32F, kernel_x)
    gradient_y = cv2.filter2D(img, cv2.CV_32F, kernel_y)
    
    magnitudes = np.sqrt(np.square(gradient_x) + np.square(gradient_y))

    #calculate angles, intital result is radian => convert to degree
    thetas = np.rad2deg(np.arctan2(gradient_y, gradient_x))
    
    return magnitudes, thetas


#Non Maximum Suppression
def non_max_suppression(magnitudes, thetas):
    height, width = magnitudes.shape
    max_edge = np.zeros((height, width), 'uint8')
        
    for i in range(1, height-1):
        for j in range(1, width-1):

            #1/ angle =  0
            if (thetas[i][j] <= 22.5) or (157.5 < thetas[i][j] <= 180):
                if (magnitudes[i][j] >=  magnitudes[i][j-1]) and (magnitudes[i][j] >= magnitudes[i][j+1]):
                    max_edge[i][j] = magnitudes[i][j]
                    
            #2/ angle = 45
            elif (22.5 < thetas[i][j] <= 67.5):
                if (magnitudes[i][j] >= magnitudes[i-1][j+1]) and (magnitudes[i][j] >=  magnitudes[i+1][j-1]):
                    max_edge[i][j] = magnitudes[i][j]
            
            #3/ angle = 90
            elif (67.5 < thetas[i][j] <= 112.5):
                if (magnitudes[i][j] >=  magnitudes[i-1][j]) and (magnitudes[i][j] >=  magnitudes[i+1][j]):
                    max_edge[i][j] = magnitudes[i][j]
            
            #angle = 135
            elif (112.5 < thetas[i][j] <= 157.5):
                if (magnitudes[i][j] >= magnitudes[i-1][j-1]) and (magnitudes[i][j] >= magnitudes[i+1][j+1]):
                    max_edge[i][j] = magnitudes[i][j]

                         
    return max_edge

#Double Threshold
def double_threshold(img, low_threshold, high_threshold):
    threshold_map = np.zeros_like(img)
    
    strong = 255
    weak = 10

    #no need to check for img < threshold case since the value in the map is already 0    
    threshold_map[np.where(img > high_threshold)] = strong
    threshold_map[np.where((img > low_threshold) & (img < high_threshold))] = weak
    
    return threshold_map

#Edge Tracking
def edge_tracking(img, weak=10, strong=255):    
    height, width = img.shape

    #check all pixels except those at borders
    for i in range(1, height-1):
        for j in range(1, width-1):
            if img[i][j] == weak:
                
                #check if any of the 8 surrounding neighbors is strong edge 
                if (img[i][j-1] == strong) or (img[i][j+1] == strong) or (img[i-1][j] == strong) or (img[i-1][j-1] == strong) or (img[i-1][j+1] == strong) or (img[i+1][j] == strong) or (img[i+1][j-1] == strong) or (img[i+1][j+1] == strong):
                    img[i][j] = 255
                else:
                    img[i][j] = 0
    
    return img    

#Canny
def Canny(image, threshold1, threshold2):
    filter = cv2.getGaussianKernel(ksize = 5, sigma = 5)
    image = cv2.filter2D(image, -1, filter)
    magnitudes, thetas = sobel(image)
    non_max = non_max_suppression(magnitudes, thetas)
    checkThres = double_threshold(non_max, threshold1, threshold2)
    return edge_tracking(checkThres)
