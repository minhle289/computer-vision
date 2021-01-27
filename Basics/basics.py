#project1.py
import numpy as np
import matplotlib.pyplot as plt
import cv2


def loadppm(filename):
    '''Given a filename, return a numpy array containing the ppm image
    input: a filename to a valid ascii ppm file 
    output: a properly formatted 3d numpy array containing a separate 2d array 
            for each color
    notes: be sure you test for the correct P3 header and use the dimensions and depth 
            data from the header
            your code should also discard comment lines that begin with #
    '''
    image = open(filename, "r")
    lines = image.readlines()
    data = []
    for line in lines:
        if line[0]!='#':
            data += line.split()        
    if (data[0] == 'P3'):
            width  = int(data[1])
            height = int(data[2])
            data = [int(i) for i in data[4:]]

            r = np.zeros((height, width), 'uint8')
            g = np.zeros((height, width), 'uint8')
            b = np.zeros((height, width), 'uint8')

            index = 0
            col_index = 0
            row_index = 0

            while row_index < height:
                while col_index < width:
                    r[row_index][col_index] = data[index]
                    g[row_index][col_index] = data[index+1]
                    b[row_index][col_index] = data[index+2]
                    index = index + 3
                    col_index = col_index+1
                row_index = row_index + 1
                col_index = 0
    # return the numpy 3d array
    return np.dstack((r,g,b))

def GetGreenPixels(img):
    '''given a numpy 3d array containing an image, return the green channel'''
    return img[:,:,1]

def GetBluePixels(img):
    '''given a numpy 3d array containing an image, return the blue channel'''
    return img[:,:,2]

def GetRedPixels(img):
    '''given a numpy 3d array containing an image, return the red channel'''
    return img[:,:,0]


#Convert image to greyscale
def GreyScale(img):
    '''given a numpy 3d array containing an image, return a 2D greyscale image'''
    shape = np.shape(img)
    height = shape[0]
    width = shape[1]
    grey = np.zeros((height, width), 'uint8')
    r = GetRedPixels(img)
    g = GetGreenPixels(img)
    b = GetBluePixels(img)
    for i in range(height):
        for j in range(width):
            grey[i][j] = (float(r[i][j]) + float(b[i][j])+ float(g[i][j]))/3
    return grey

#code to create black/white monochrome image
def monochrome(img):
    '''given a numpy 2d array containing an image, return a black/white image'''
    height, width = img.shape
    output = np.zeros_like(img)
    for i in range(height):
        for j in range(width):
            if img[i][j] < 120:
                output[i][j] = 0
            else:
                output[i][j] = 255
    return output

#Histogram Equalization
# calculate the histogram of the greyscale image
def histogram(img):
    '''given a numpy 2d array containing an image, return a histogram'''
    height, width = img.shape
    total = height*width
    
    hist = [0.0 for i in range(256)]

    #count appearance of each value in img
    for i in range(height):
        for j in range(width):
            hist[img[i][j]] += 1

    return hist

# calculate the cumulative distribution c(I) of the histogram values
def find_cdf(hist):
    cdf = []
    for i in range(len(hist)):
        cdf.append(sum(hist[:i+1]))     
    return cdf

# rescale the greyscale values accordingly
import math
def hist_equalization(img):
    height, width = img.shape
    hist = histogram(img)
    cdf = find_cdf(hist)
    cdf_min = min(cdf)
    length = len(cdf)
    new_vals = [math.floor(((i - cdf_min)/(height*width-cdf_min))*(length-1)) for i in cdf]

    #create new image to store new vals after equalization
    modified = np.zeros_like(img)

    for i in range(0, height):
        for j in range(0, width):
            modified[i][j] = new_vals[img[i][j]]
    return modified


if __name__ == "__main__":
    #put any command-line testing code you want here.
    #note this code in this block will only run if you run the module from the command line
    # (i.e. type "python3 project1.py" at the command prompt)
    # or within your IDE
    # it will NOT run if you simply import your code into the python shell.
    
    
    rgb = loadppm("../images/simple.ascii.ppm")
    #plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis
    #plt.imshow(rgb)
    #plt.show()
    
    #rgb = loadppm("../images/zebra.ascii.ppm")
    #plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis
    
    #rgb = loadppm("../images/simple.ascii.ppm")
    #green = GetGreenPixels(rgb)
    
    
    #plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis
    #plt.imshow(green,cmap='gray', vmin=0, vmax=255)
    #plt.show()
    
    #red = GetRedPixels(rgb)
    #plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis
    #plt.imshow(red,cmap='gray', vmin=0, vmax=255)
    #plt.show()
    
    #blue = GetBluePixels(rgb)
    #plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis
    #plt.imshow(blue,cmap='gray', vmin=0, vmax=255)
    #plt.show()
    
    ##code to test greyscale conversions of the colored boxes and the zebra
    #rgb_zebra = loadppm("../images/zebra.ascii.ppm")
    #rgb_block = loadppm("../images/simple.ascii.ppm")

    #grey_zebra = GreyScale(rgb_zebra)
    #grey_block = GreyScale(rgb_block)
    #plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis

    #plt.imshow(grey_zebra,cmap='gray', vmin=0, vmax=255)
    #plt.show()
    
    #plt.imshow(grey_block,cmap='gray', vmin=0, vmax=255)
    #plt.show()


    ##code to test black/white monochrome image
    #monochrome = monochrome(rgb_zebra)

    #plt.xticks([]), plt.yticks([])   
    #plt.imshow(monochrome,cmap='gray', vmin=0, vmax=255)
    #plt.show()

    
    ##code to test histogram equalization
    #modified = hist_equalization(rgb_zebra)

    #plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis
    #plt.imshow(modified,cmap='gray', vmin=0, vmax=255)
    #plt.show()
