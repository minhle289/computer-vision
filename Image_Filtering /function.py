import numpy as np

def my_imfilter(image, filter):
  """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (k, k)
  Returns
  - filtered_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using opencv or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that I can finish grading 
   before the heat death of hte universe. 
   your code works.
  - Remember these are RGB images, accounting for the final image dimension.
  """

  ############################
  ### TODO: YOUR CODE HERE ###
  h_fil = filter.shape[0]
  w_fil = filter.shape[1]
  h_pad = (h_fil-1)//2
  w_pad = (w_fil-1)//2
  
  filtered_image = np.zeros_like(image)
  h_img = image.shape[0]
  w_img = image.shape[1]
  
  #Grayscale image (2d numpy)
  if len(image.shape) == 2:
    padded = np.pad(image, ((h_pad,),(w_pad,)), 'constant')
    for i in range(h_img):
      for j in range(w_img):
        sum_area = padded[i:i+h_fil, j:j+w_fil]
        sum_area = np.multiply(sum_area, filter)
        filtered_image[i][j] = np.sum(sum_area)

  #3d RGB image (3d numpy)
  else:
    #reshape filter to allow multiplication of 2 nparray
    padded = np.pad(image, ((h_pad,),(w_pad,),(0,)), 'constant')
    filter = filter.reshape((h_fil, w_fil, 1))
    for i in range(h_img):
      for j in range(w_img):
        sum_area = padded[i:i+h_fil, j:j+w_fil, :]
        sum_area = np.multiply(sum_area, filter)
        filtered_image[i][j][:] = np.sum(sum_area, axis=(0,1))
	
  ### END OF STUDENT CODE ####
  ############################

  return filtered_image

def create_hybrid_image(image1, image2, filter):
  """
  Takes two images and creates a hybrid image. Returns the low
  frequency content of image1, the high frequency content of
  image 2, and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  Returns
  - low_frequencies: numpy nd-array of dim (m, n, c)
  - high_frequencies: numpy nd-array of dim (m, n, c)
  - hybrid_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
    as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
    in the notebook code.
  """

  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]

  ############################
  ### TODO: YOUR CODE HERE ###

  low_frequencies = my_imfilter(image1, filter)
  high_frequencies = image2 - my_imfilter(image2, filter)

  hybrid_image = low_frequencies + high_frequencies

  #clipping
  hybrid_image = np.clip(hybrid_image, 0.0, 1.0)

  ### END OF STUDENT CODE ####
  ############################

  return low_frequencies, high_frequencies, hybrid_image
