import numpy as np
import cupy as cp
from patchify import patchify, unpatchify
import tensorflow as tf


fmax = cp.finfo(cp.float64).max #maximum float64 numbwe
fmin = cp.finfo(cp.float64).min #minimum float64 numbwe



def create_fields(img,patch_shape,step):
    """
    Form the receptive fields.

    Keyword arguments:
    img -- input image
    patch_shape -- receptive field dims (touple)
    step -- patch window step
       
    """
    img = np.array(img.tolist()) #convert to numpy
    img=patchify(img,patch_shape,step)
    img=img.reshape(img.shape[0]*img.shape[1],img.shape[2],img.shape[3])
    return tocp(np.expand_dims(np.expand_dims(img,1),1).tolist())

def reconstruct(imp_shape,img_shape,patches):
    """
    Retrieve the original image from the receptive fields.

    Keyword arguments:
    imp_shape -- original shape of the patch table
    img_shape--  image shape 
    patches -- the patch table
       
    """
    patches = np.array(patches.tolist())
    patches = patches.reshape(*imp_shape)
    patches=unpatchify(patches,img_shape)
    return tocp(patches)
    

def tocp(arr):
    """
    Converts a list to CuPy array.

    Keyword arguments:
    arr-- list to be converted
       
    """
    return cp.array(arr)

def rgb2gray(x):
    """
    Converts an array of rgb images to grayscale.

    Keyword arguments:
    x-- images to be converted
       
    """
    if len(x.shape)==3:
        return 0.299*x[:,:,0]+0.587*x[:,:,1]+0.114*x[:,:,2]
    return 0.299*x[:,:,:,0]+0.587*x[:,:,:,1]+0.114*x[:,:,:,2]
        

def import_data(dname='mnist'):
    """
    imports and preprocesses the dataset.

    Keyword arguments:
    dname-- tf dataset name to be imported
       
    """
    if dname== 'cifar10':
        (x1,y1),(x2,y2)= tf.keras.datasets.cifar10.load_data() 
        x1=rgb2gray(x1)
        x2=rgb2gray(x2)
        y1=y1.squeeze()
        y2=y2.squeeze()
        scaling_factor = np.max(x1).tolist()
        x1 =x1/ scaling_factor
        x2 =x2/ scaling_factor
        return (x1,y1),(x2,y2)
    else:
        if dname=='mnist':
            (x1,y1),(x2,y2) =  tf.keras.datasets.mnist.load_data()
        elif dname== 'fmnist':
            (x1,y1),(x2,y2)= tf.keras.datasets.fashion_mnist.load_data()
        else:
            return None
        scaling_factor = np.max(x1).tolist()
        x1 =x1/ scaling_factor
        x2 =x2/ scaling_factor
        return (x1,y1),(x2,y2)

     
        