import numpy as np
import pykitti
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

a = np.array([[1,2], [3, 4], [5, 6]])
print(a)
print(a[[0, 1, 2], [0, 1, 0]])
print(a.dtype)


x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)
print('x=',x,'\n')
print(np.add(x,y),'\n')
t = x+y
print(x,y,t,'\n')
print('x dot y ',x.dot(y))

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])




basedir = r'\z:\Datasets\KITTI\Formatted'

date = '2011_09_26'
drive = '0001'

# The range argument is optional - default is None, which loads the whole dataset
data = pykitti.raw(basedir, date, drive, range(0, 50, 5))

# Data are loaded only if requested
data.load_calib()
point_cam0 = data.calib.T_cam0_velo.dot(point_velo)

data.load_oxts()
point_w = data.oxts[0].T_w_imu.dot(point_imu)

data.load_rgb()
cam2_image = data.rgb[0].left
plt.imshow(cam2_image)
plt.axis('equal')
