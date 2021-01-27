I found the numpy boolean indexing especially useful in this project.

Hough part:
I struggled quite a lot to match my output with OpenCV's HoughLines. Two things I found out that help:

The number of rhos and thetas: add 1 after my calculation (I forgot to count value 0 at first and that altered the output a lot)

Round up theta to 7 decimals (seem to be the output of OpenCV)

A major decision that I made is to use index of rho and theta array instead of the rho and theta value themselves (for index in line votes count). Two drawbacks of using rho and theta value as indexes: a lot of rounding + negative index (which confused me from time to time). I came across np.searchsorted which makes it super fast to find the index of the current rhoVal in the rho array.

Overall, my output is a bit different from OpenCV but not by a lot. However, with higher threshold, OpenCV sometimes perform much better.

Canny part:
At first I tried to keep the original value of the strong edge. However, it made the code more complicated (since I need to pass both the threshold map and the image around). After I found out the OpenCV's Canny just make all values in the returned output either 0 or 255, I decided to do the same thing for my code.

My edge detector performs decently well though it does not exactly match OpenCV's Canny.
