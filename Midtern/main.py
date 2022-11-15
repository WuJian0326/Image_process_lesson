# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os

import numpy as np
import cv2
import argparse
from matplotlib import pyplot as plt

def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        # print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output

def constract(img,constract1,bri):
    output = img * (constract1/127 +1) - constract1 +bri
    return output


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath",type=str, help="image file path",)
    parser.add_argument("--out", type=str, help="output folder name, plz have '/' at the end", default= "./save/")
    args = parser.parse_args()

    path = "./HW_Image/"
    pathsave = "./save/"


    img = cv2.imread(args.filepath,0)

    # "==================================="
    # x, y = np.mgrid[-3:4, -3:4]
    # gaussian_kernel = np.exp(-(x ** 2 + y ** 2))
    kernel1 = 1/300 * np.ones((2,150))

    kernel = np.array([[4, 8, 4],
                       [0, 0, 0],
                       [-4, -8, -4]])
    # Normalization
    # gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()


    img = (img*255).astype(np.uint8)

    grad1 = constract(img, constract1=150, bri=0)
    grad1 = convolve2D(img, kernel1)

    grad1 = constract(grad1, constract1 = 150, bri = 0)

    grad2 = convolve2D(grad1, kernel)


    count = 0
    i = 500
    start = 500
    end = 500

    slice1 = grad2[:, 0:1]
    # print(slice1.shape)
    slice2 = grad2[:,1:2]
    slice3 = grad2[:,2:3]
    while i < (len(slice1) - 3):
        if( slice1[i] - slice1[i+3] > 60 and slice1[i] >= 0 and slice1[i+3] < 0):
            if(start != 500):
                start = i
            count += 1
            img[i] = 0
            img[i+1] = 0
            img[i+2] = 0
            img[i+3] = 0

            i += 10
            end = i
        if(i - end > 100 and start != end):
            break
        i += 1
    filename = args.filepath.split("/")[-1]
    if not os.path.exists(args.out):
        os.mkdir(args.out)
        print(f"Create output folder {args.out}")
    cv2.imwrite(args.out + filename,img[start:end])

    plt.imshow(img[start:end], cmap=plt.get_cmap('gray'))
    plt.title(f"number of slides: {count}")
    plt.show()
    print("number of slice: ",count)

