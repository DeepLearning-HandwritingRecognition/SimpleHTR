import numpy
import numpy as np
import cv2
import skimage as sk
import os
from pathlib import Path

import random

#for this program just

outputDirName = ""
parentDirectory = os.getcwd()
numTransformations = 1


random.seed()


def random_noise(image_array):
    # add random noise to the image
    noiseImg=sk.util.random_noise(image_array, mode='gaussian', clip=True)
    return np.array(255 * noiseImg, dtype='uint8')


def blur_filter(img_array):
    # blur the image
    return cv2.blur(img_array, (8, 8))


# for IAM dataset

def reduce_line_thickness(image):
    kernel = np.ones((4, 4), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


def random_stretch(img):
    stretch = (random.random() - 0.5)  # -0.5 .. +0.5
    wStretched = max(int(img.shape[1] * (1 + stretch)), 1)  # random width, but at least 1
    img = cv2.resize(img, (wStretched, img.shape[0]))  # stretch horizontally by factor 0.5 .. 1.5
    return img




def main():
    # runs through and gets every single image from the collection
    os.chdir(Path(os.getcwd()).parent)
    os.chdir(os.path.join(os.getcwd(),"data"))
    for parentFolder in os.listdir(os.chdir(os.getcwd()+'\IAM_dataset')):
        for childFolder in os.listdir(os.chdir(os.getcwd()+"\\"""+parentFolder)):
            for image in os.listdir(os.chdir(os.getcwd()+"\\"""+childFolder)):
                loadAndTransform(image)
            os.chdir(Path(os.getcwd()).parent)
        os.chdir(Path(os.getcwd()).parent)


def loadAndTransform(imageFile):
    """this if statement is added because there is a corrupted file in the IAM database
        and deleting that file everytime is very tedious"""
    if imageFile == "a01-117-05-02.png" or imageFile == "r06-022-03-05.png":
        return

    imageArray = cv2.imread(imageFile)
    writeImage(imageArray, os.getcwd(), imageFile)

    #imageArray = randomSelection(imageArray)
    imageArray = random_stretch(imageArray)

    imageFile = imageFile[:-4] + "T" + imageFile[-4:]
    writeImage(imageArray, os.getcwd(), imageFile)


def writeImage(imageArray, directory, fileName):
    outputDir = directory.replace("IAM_dataset", outputDirName)
    if not os.path.isdir(outputDir):
        os.makedirs(os.path.join(outputDir))

    os.chdir(outputDir)
    cv2.imwrite(fileName, imageArray)
    os.chdir(directory)


def randomSelection(imageArray):
    # noise=0, blur=1, line=2, stretch=3
    image=imageArray
    for x in range(numTransformations):
        rng = random.randrange(0, 3)
        if rng==0:
            image=random_noise(imageArray)
        elif rng==1:
            image=blur_filter(imageArray)
        elif rng==2:
            image=reduce_line_thickness(imageArray)
        elif rng==3:
            image=random_noise(imageArray)

    return image


if __name__ == "__main__":
    main()
