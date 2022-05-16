import cv2
from pandas import wide_to_long
from phe import paillier
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lena256.bmp',0)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

height,width=img.shape[:2]
imgCopy=img.copy()
# public_key, private_key = paillier.generate_paillier_keypair()
# image_array=np.asarray(img)
# print(image_array)
# data_encrypted = [[public_key.encrypt(int(x)) for x in row] for row in image_array]
# print(data_encrypted)



def control_brightness(myimg,brightness=0):#aici primeste array, schimba
    for i in range(0, height):
        for j in range(0,width):
            grey= myimg[i,j]
            grey_new = np.where((255 - grey) < brightness,255,grey+brightness)
            myimg[i,j]=grey_new
    return myimg

def image_negation(myimg):
    for i in range(0, height):
        for j in range(0,width):
            pixelColorVals = myimg[i,j]
            redPixel    = 255 - pixelColorVals[0]; # Negate red pixel

            greenPixel  = 255 - pixelColorVals[1]; # Negate green pixel

            bluePixel   = 255 - pixelColorVals[2]; # Negate blue pixel

            myimg[i,j]=(redPixel, greenPixel, bluePixel);  
    return myimg

# img2=control_brightness(np.array(imgCopy),10)

img_negation=control_brightness(imgCopy,100)

plt.imshow(img_negation)
plt.show()