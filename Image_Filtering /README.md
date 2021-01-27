## Some reflections 

I had to search a lot online + try by myself to get how numpy functions work. The examples on numpy website are only basic cases and it took me quite a bit to know I can do something like giving pad_width (in np.pad) a more complicated tuple to get what I want. 

I also had a pretty tough time figure out how to do multiplication and sum for the 3d nparray. I was only able to arrive at something that works by trying to fix what the errors said... Definitely need to do a bit more research on numpy. At least I feel a lot more comfortable with 3d nparray, especially slicing, after this project.

In my_imfilter, I divide into 2 cases: gray scale and rgb image, so I can do padding and multiply/sum with filter array accordingly. I feel like it may not be the most efficient way since the codes for each differ only slightly. I tried to write something that can deal with both but had no success, found that dealing with each case separately is way easier.

Have no problem at all writing create_hybrid_image, pretty straightforward in what I need to do.