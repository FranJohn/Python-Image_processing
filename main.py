#Image processing

import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.transform import rescale
import numpy as np
from scipy.signal import convolve2d
from scipy import fftpack
import math


def convolve(image, kernel):
    r_conv = convolve2d(in1=image[:, :, 0], in2=kernel, mode='valid', boundary='fill')
    g_conv = convolve2d(in1=image[:, :, 1], in2=kernel, mode='valid', boundary='fill')
    b_conv = convolve2d(in1=image[:, :, 2], in2=kernel, mode='valid', boundary='fill')
    return np.stack([r_conv, g_conv, b_conv], axis=2)


identity_kernel = np.array([[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]])
# Spatial Lowpass
spatial_lowpass = np.array([[1/9, 1/9, 1/9],
                            [1/9, 1/9, 1/9],
                            [1/9, 1/9, 1/9]])
# Edge Detection2
spatial_highpass = np.array([[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]])
# Bottom Sobel Filter
bottom_sobel = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])
#Top Sobel
top_sobel = np.array([[1, 2, 1],
                      [0, 0, 0],
                  [-1, -2, -1]])


picture = imread('C:\\Users\\chesk\\Pictures\\Python\\Creech.jpg')
picture_gray = rgb2gray(picture)

fig1 = plt.figure()
#Original
axis1 = fig1.add_subplot(221)
axis1.title.set_text('Original')
axis1.imshow(picture, cmap='gray')
#Identity
axis2 = fig1.add_subplot(222)
identity_picture = convolve(picture, identity_kernel)
axis2.imshow(abs(identity_picture), cmap='gray')
axis2.title.set_text('Identity')
#Spatial Lowpass
axis3 = fig1.add_subplot(223)
spatial_picture = convolve2d(picture_gray, spatial_lowpass)
axis3.imshow(abs(spatial_picture), cmap='gray')
axis3.title.set_text('Spatial Lowpass')
#Spatial Highpass
axis4 = fig1.add_subplot(224)
spatial_hp_picture = convolve2d(picture_gray, spatial_highpass)
axis4.imshow(abs(spatial_hp_picture), cmap='gray')
axis4.title.set_text('Spatial Highpass')

fig2 = plt.figure()
#Fourier --- not sure about this boyo
axis5 = fig2.add_subplot(221)
fourier = fftpack.fft2(picture).real
axis5.imshow(abs(fourier),  cmap='gray')
axis5.title.set_text('Fourier')
#Bottom Sobel
axis6 = fig2.add_subplot(222)
bottom_sobel_picture = convolve2d(picture_gray, bottom_sobel)
axis6.imshow(abs(bottom_sobel_picture), cmap='gray')
axis6.title.set_text('Bottom Sobel')
#Top Sobel
axis7 = fig2.add_subplot(223)
top_sobel_picture = convolve2d(picture_gray, top_sobel)
axis7.imshow(abs(top_sobel_picture), cmap='gray')
axis7.title.set_text('Top Sobel')

plt.show()