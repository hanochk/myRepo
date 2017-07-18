import scipy
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = sio.loadmat('1199.mat')

type(img)

imgplot = plt.imshow(img)
plt.axis('equal')
img=mpimg.imread('stinkbug.png')


help(sio.loadmat)


from scipy import misc
l = misc.face();
plt.imshow(l)
plt.show()

img=mpimg.imread('1199.bmp')
plt.imshow(img)
plt.show()

import skimage
from skimage import segmentation
imgseg=skimage.segmentation.felzenszwalb(img, scale=1, sigma=0.8, min_size=20)
plt.imshow(imgseg)
plt.show()
imgseg=skimage.segmentation.felzenszwalb(img, scale=1, sigma=0.000001, min_size=20)