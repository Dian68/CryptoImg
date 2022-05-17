from copy import copy
from dataclasses import dataclass
from xml.etree.ElementTree import tostring
import PIL
import cv2
from pandas import array, wide_to_long
import phe
from phe import paillier
from PIL import Image,ImageOps
import numpy as np
import matplotlib.pyplot as plt


public_key, private_key = paillier.generate_paillier_keypair()





img=Image.open("lenna128_noised.png")
img=ImageOps.grayscale(img)
height,width=img.size
print(height,width)

imgcv=cv2.imread("lenna128_noised.png",0)
 
def encryptImage(img_n):
    my_array = [[0]*height]*width
    img_data=np.asarray(img_n)
    for i in range(0, height):
        my_array[i]=[public_key.encrypt(int(x),None,None) for x in img_data[i]]
    return my_array


def decryptImage(my_array):
    my_array_dec = [[0]*height]*width
    for i in range(0, height):
        my_array_dec[i]=[private_key.decrypt(x) for x in my_array[i]]
    my_array_nolist=np.array(my_array_dec)
    im=Image.fromarray((255-my_array_nolist * 255).astype('uint8'), mode='L')
 #nu salveaza bine, face poza neagra, incerc rezolv
    return im



#trebe testatae cu imaginea citita cu Image.open, nu cu cv2.open
def control_brightness_image(myimg,brightness=0):
    myimg=myimg+brightness
    return myimg

def control_brightness_crypted(myimg_Array, brightness=0):
    for i in range(0, height):
        for j in range(0,width):
            myimg_Array[i][j]=myimg_Array[i][j]+ brightness
    return myimg_Array

def image_negation_image(myimg): 
    myimg=255-myimg
    return myimg
def image_negation_crypted(myimg_Array):
    for i in range(0, height):
        for j in range(0,width):
            myimg_Array[i][j]=255-myimg_Array[i][j]
    return myimg_Array


def encrypt_image_second(img, public_key):  
    shape = img.shape
    img = img.flatten()
    img_enc = []
    for i in range(len(img)):
        tmp = public_key.encrypt(int(img[i]))
        img_enc.append(tmp)
        x = private_key.decrypt(tmp)
    return np.reshape(img_enc, shape)

def decrypt_image_second(img, private_key):
  shape = img.shape
  img = img.flatten()
  img_dec = []
  for i in range(len(img)):
    tmp = private_key.decrypt(img[i])
    img_dec.append(tmp)  
  return np.reshape(np.array(img_dec).astype(np.uint8), shape)

def secure_noise_reduction_image(myimg):
    imgnou=myimg.copy()
    cv2.fastNlMeansDenoising(myimg,imgnou,h=10)
    return imgnou


def secure_noise_reduction_crypted(myimg,kernel_size):
    myimg = np.array(myimg)
    img_filtered = myimg.copy()
    (n, m) = myimg.shape
    offset = kernel_size // 2
    for i in range(offset, n-offset, 1):
        for j in range(offset, m-offset, 1):
            window = myimg[i-offset:i+offset+1, j-offset:j+offset+1]
            img_filtered[i][j] = np.mean(window)
    return img_filtered
    

def edge_detection_image(myimg):
    edges = cv2.Canny(myimg,50,150)
    return edges

def edge_detection_encrypted(myimage):
    nonoiseimg=secure_noise_reduction_crypted(myimage,3)
    edges=myimage-nonoiseimg
    return edges


  
img2=imgcv.copy()
#img_encrypted=encryptImage(img)


# img_encrypted_plus_brig=control_brightness_crypted(img_encrypted,100)
#img_plus_brig=control_brightness_image(img2,100)

#img_encrypted_negate=image_negation_crypted(img_encrypted)
#img_negate=image_negation_image(img2)


# img_enc=encrypt_image_second(imgcv,public_key)
# img_noise_reduction=secure_noise_reduction_image(img2)
# img_noise_encrypted=secure_noise_reduction_crypted(img_enc,3)
# img_dec=decrypt_image_second(img_noise_encrypted,private_key)

img_enc=encrypt_image_second(imgcv,public_key)
img_edge=edge_detection_image(img2)
img_edge_encrypted=edge_detection_encrypted(img_enc)
img_dec=decrypt_image_second(img_edge_encrypted,private_key)


#img_decrypted=decryptImage(img_noise_encrypted)

fig=plt.figure()
fig.add_subplot(1,3,1)
plt.imshow(imgcv, cmap='gray')
fig.add_subplot(1,3,2)
plt.imshow(img_edge,cmap='gray')
fig.add_subplot(1,3,3)
plt.imshow(img_dec,cmap='gray')

plt.show()