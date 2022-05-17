from dataclasses import dataclass
from xml.etree.ElementTree import tostring
import PIL
import cv2
from pandas import array, wide_to_long
from phe import paillier
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


public_key, private_key = paillier.generate_paillier_keypair()





img=Image.open("lena256.png")
height,width=img.size
print(height,width)

my_rows, my_cols = (height, width)
# img_d=np.asarray(img_n)
# print(img_d)
# img2=Image.fromarray(img_d)
# img2.save('my.png')



def encryptImage(img_n):

    my_array = [[0]*my_cols]*my_rows


    img_data=np.asarray(img_n)
    print(img_data)
    for i in range(0, height):
        my_array[i]=[public_key.encrypt(int(x),None,None) for x in img_data[i]]
    return my_array
#print(my_array)
def decryptImage(my_array):
    my_array_dec = [[0]*my_cols]*my_rows
    for i in range(0, height):
        my_array_dec[i]=[private_key.decrypt(x) for x in my_array[i]]
#print(my_array_dec)
    my_array_nolist=np.array(my_array_dec)
    print("\n\n")
    print(my_array_nolist)
    data=Image.fromarray(my_array_nolist)
    data.save("reslena.png") #nu salveaza bine, face poza neagra, incerc rezolv

img_encrypted=encryptImage(img)
decryptImage(img_encrypted)

#trebe testatae cu imaginea citita cu Image.open, nu cu cv2.open
def control_brightness(myimg,brightness=0):#aici primeste imaginea grey
    for i in range(0, height):
        for j in range(0,width):
            grey= myimg[i,j]
            grey_new = np.where((255 - grey) < brightness,255,grey+brightness)
            myimg[i,j]=grey_new
    return myimg

def image_negation(myimg): #aici e testata cu imaginea color, trebe pt imaginea grey
    for i in range(0, height):
        for j in range(0,width):
            pixelColorVals = myimg[i,j]#negate just grey pixel, deci nu va exista [0],[1],[2] 
            redPixel    = 255 - pixelColorVals[0]; # Negate red pixel

            greenPixel  = 255 - pixelColorVals[1]; # Negate green pixel

            bluePixel   = 255 - pixelColorVals[2]; # Negate blue pixel

            myimg[i,j]=(redPixel, greenPixel, bluePixel);  
    return myimg

# img2=control_brightness(np.array(imgCopy),10)

#img_negation=control_brightness(imgCopy,100)



plt.show()