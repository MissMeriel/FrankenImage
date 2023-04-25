import numpy as np
import torch
import cv2
import os
import sys
import random
import string
# saliency/SmoothGrad Boilerplate imports. (pytorch implementation)
import saliency.core as saliency
import numpy as np
import PIL.Image
from matplotlib import pylab as P
import torch
from torchvision import models, transforms
import tensorflow as tf
sys.path.append('models')

device = torch.device("cpu") #torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# saliency/SmoothGrad utils
def ShowImage(im, title='', ax=None, save=False, fid=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im)
    P.title(title)
    P.pause(3)
    if save:
        randstr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
        if fid is not None:
            P.savefig(f"test-xrai-{fid}-{randstr}.jpg")
        else:
            P.savefig(f"test-xrai-{randstr}.jpg")


def ShowGrayscaleImage(im, title='', ax=None, save=False):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
    P.title(title)
    P.pause(3)
    if save:
        randstr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
        P.savefig(f"test-xrai-{randstr}.jpg")

def ShowHeatMap(im, title, ax=None, save=False):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im, cmap='inferno')
    P.title(title)
    if save:
        randstr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
        P.savefig(f"test-xrai-{randstr}.jpg")

def LoadImage(file_path, imsize=(240, 135)):
    im = PIL.Image.open(file_path)
    im = im.resize(imsize)
    im = np.asarray(im)
    return im

transformer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
def PreprocessImages(images):
    # assumes input is 4-D, with range [0,255]
    #
    # torchvision have color channel as first dimension
    # with normalization relative to mean/std of ImageNet:
    #    https://pytorch.org/vision/stable/models.html
    images = np.array(images)
    images = images/255
    images = np.transpose(images, (0,3,1,2))
    images = torch.tensor(images, dtype=torch.float32, requires_grad=True)
    images = transformer.forward(images)
    return images

# import model
from models.DAVE2pytorch import *
ptfilename = "models/model-DAVE2v1-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt"
model = torch.load(ptfilename).eval().to(device)

# import VQVAE
from models.VQVAE import Encoder, Decoder, VQVAEModel
from models import VQVAE
vqvaemodel = VQVAE.instantiate_VQVAE(checkpoint_root="checkpoints136x240-origdataset")

# Load validation image
# img_fqp = "/home/meriel/datasets/sampledir/arrows0.png"
img_fqp = "/home/meriel/datasets/hirochi_raceway-9205-lap-test2/sample-00001.jpg"
img_fqp = "/home/meriel/datasets/training_images_industrial-racetrackstartinggate0/hopper_industrial_0.jpg"
img_fqp = "/home/meriel/datasets/sampledir/arrows1.png"
fname = img_fqp[0:-4].split("/")[-1]
img_fqp = "/home/meriel/datasets/training_images_industrial-racetrackstartinggate0/hopper_industrial_500.jpg"
image_dims = (135, 240)
batch_size=32
im_orig = LoadImage(img_fqp, imsize=image_dims[::-1])
im_tensor = PreprocessImages([im_orig])
ShowImage(im_orig)
print(f"{im_tensor.shape=}")

# Register hooks for Grad-CAM, which uses the last convolution layer
print(model)
conv_layer = model.conv5
conv_layer_outputs = {}
def conv_layer_forward(m, i, o):
    # move the RGB dimension to the last dimension
    conv_layer_outputs[saliency.base.CONVOLUTION_LAYER_VALUES] = torch.movedim(o, 1, 3).detach().numpy()
def conv_layer_backward(m, i, o):
    # move the RGB dimension to the last dimension
    conv_layer_outputs[saliency.base.CONVOLUTION_OUTPUT_GRADIENTS] = torch.movedim(o[0], 1, 3).detach().numpy()
conv_layer.register_forward_hook(conv_layer_forward)
conv_layer.register_full_backward_hook(conv_layer_backward)

# use SmoothGrad to find salient regions
class_idx_str = 'class_idx_str'
def call_model_function(images, call_model_args=None, expected_keys=None):
    images = PreprocessImages(images)
    target_class_idx = call_model_args[class_idx_str]
    output = model(images)
    if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
        grads = torch.autograd.grad(output, images, grad_outputs=torch.ones_like(output))
        grads = torch.movedim(grads[0], 1, 3)
        gradients = grads.detach().numpy()
        return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
    else:
        one_hot = torch.zeros_like(output)
        one_hot[:,target_class_idx] = 1
        model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)
        return conv_layer_outputs

# TEST VAE
print(vqvaemodel, "\n")
# temp = np.random.normal(loc=0.5, scale=.5, size=(1, 136, 240, 3))
# temp = LoadImage(img_fqp, imsize=(240, 136))[:,:,::-1][None]
temp = PIL.Image.open(img_fqp).resize((240, 136))
temp = np.array(temp)[None]

im_tftensor = tf.convert_to_tensor(temp, dtype_hint=tf.float32) #, dtype=None, dtype_hint=None, name=None)
im_tftensor = (tf.cast(im_tftensor, tf.float32) / 255.0) - 0.5
print(f"{im_tftensor.shape=}")
temp_reconst = vqvaemodel(im_tftensor, is_training=False)['x_recon'].numpy() + 0.5
print(temp_reconst.shape, type(temp_reconst))
ShowImage(temp_reconst[0])

predictions = model(im_tensor)
print(f"{predictions=}")
print("Prediction: " + str(predictions))
ShowImage(im_orig, title=f"Orig. img pred={predictions.detach().numpy().item():.3f}", save=True)
# predictions = predictions.detach().numpy().item()
prediction_class = predictions
call_model_args = {class_idx_str: prediction_class}

use_xrai = True
use_vanilla_smoothgrad = False
if use_vanilla_smoothgrad:
    # Construct the saliency object. This alone doesn't do anthing.
    gradient_saliency = saliency.GradientSaliency()

    # Compute the vanilla mask and the smoothed mask.
    vanilla_mask_3d = gradient_saliency.GetMask(im_orig, call_model_function, call_model_args)
    smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(im_orig, call_model_function, call_model_args)

    # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_mask_3d)
    smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)

    # Set up matplot lib figures.
    ROWS = 1
    COLS = 2
    UPSCALE_FACTOR = 10
    P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))

    # Render the saliency masks.
    ShowGrayscaleImage(vanilla_mask_grayscale, title='Vanilla Gradient', ax=P.subplot(ROWS, COLS, 1))
    ShowGrayscaleImage(smoothgrad_mask_grayscale, title='SmoothGrad', ax=P.subplot(ROWS, COLS, 2))

    print(f"{np.mean(smoothgrad_mask_grayscale)=} \n{np.std(smoothgrad_mask_grayscale)=} \n{np.min(smoothgrad_mask_grayscale)=} \t {np.max(smoothgrad_mask_grayscale)=} \t {type(smoothgrad_mask_grayscale)}")
elif use_xrai:
    # Construct the saliency object. This alone doesn't do anthing.
    xrai_object = saliency.XRAI()

    # Compute XRAI attributions with default parameters
    xrai_attributions = xrai_object.GetMask(im_orig, call_model_function, call_model_args, batch_size=20)

    # Set up matplot lib figures.
    ROWS = 1
    COLS = 3
    UPSCALE_FACTOR = 20
    P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))

    # Show original image
    ShowImage(im_orig, title='Original Image', ax=P.subplot(ROWS, COLS, 1))

    # Show XRAI heatmap attributions
    ShowHeatMap(xrai_attributions, title='XRAI Heatmap', ax=P.subplot(ROWS, COLS, 2))

    # Show most salient 30% of the image
    mask = xrai_attributions >= np.percentile(xrai_attributions, 50)
    im_mask = np.array(im_orig)
    im_mask[~mask] = 0
    ShowImage(im_mask, title='Top 50%', ax=P.subplot(ROWS, COLS, 3), save=True)

# noise-out salient regions
print(im_mask.shape)
print(f"{np.mean(im_mask)=} \n{np.std(im_mask)=} \n{np.min(im_mask)=} \t {np.max(im_mask)=} \t {type(im_mask)}")
im_unpert = LoadImage("/home/meriel/datasets/training_images_industrial-racetrackstartinggate0/hopper_industrial_374.jpg", imsize=image_dims[::-1])
print(f"{im_unpert[:,:,0].shape=}")
mean0 = np.mean(im_unpert[:,:,0])
std0 = np.std(im_unpert[:,:,0])
mean1 = np.mean(im_unpert[:,:,1])
std1 = np.std(im_unpert[:,:,1])
mean2 = np.mean(im_unpert[:,:,2])
std2 = np.std(im_unpert[:,:,2])
im_out = np.copy(im_orig)
# for i in range(im_mask.shape[0]):
#     for j in range(im_mask.shape[1]):
#         if (im_mask[i][j] == 0).all():
#             noisy_arr = np.array([np.random.choice((-1, 0, 1)) * np.random.normal(loc=mean0,scale=std0), 
#                                     np.random.choice((-1, 0, 1)) * np.random.normal(loc=mean1,scale=std1),
#                                     np.random.choice((-1, 0, 1)) * np.random.normal(loc=mean2,scale=std2)])
#             im_out[i][j][:] = np.clip(im_out[i][j][:] + 0.5 * np.squeeze(noisy_arr), 0, 255)

print(f"{np.unique(xrai_attributions)=}")
# use xrai heatmap to determine masking
xrai_attributions_index = -1
xrai_max_val = np.unique(xrai_attributions)[xrai_attributions_index]
for i in range(xrai_attributions.shape[0]):
    for j in range(xrai_attributions.shape[1]):
        # print(im_mask[i][j].shape)
        if (xrai_attributions[i][j] == xrai_max_val):
        #    im_mask[i][j][:] = [random.randrange(0,255), random.randrange(0,255) , random.randrange(0,255)]
            noisy_arr = np.array([np.random.choice((-1, 0, 1)) * np.random.normal(loc=mean0,scale=std0), 
                                    np.random.choice((-1, 0, 1)) * np.random.normal(loc=mean1,scale=std1),
                                    np.random.choice((-1, 0, 1)) * np.random.normal(loc=mean2,scale=std2)])
            im_out[i][j][:] = np.clip(im_out[i][j][:] + 0.33 * np.squeeze(noisy_arr), 0, 255)


ShowImage(im_out, title='noise-filled mask', save=True)
im_out_tensor = PreprocessImages([im_out])
predictions = model(im_out_tensor)
ShowImage(im_out, title=f'Noise-filled mask pred={predictions.detach().numpy().item():.3f}', save=True)

# run through VQVAE
im_out = cv2.resize(im_out, (240, 136), interpolation = cv2.INTER_AREA)
im_tftensor = tf.convert_to_tensor(im_out[None], dtype_hint=tf.float32)
im_tftensor = (tf.cast(im_tftensor, tf.float32) / 255.0) - 0.5
print(f"{im_tftensor.shape=}")
temp_reconst = vqvaemodel(im_tftensor, is_training=False)['x_recon'].numpy() + 0.5
print(temp_reconst.shape, type(temp_reconst))
ShowImage(temp_reconst[0], title='vae refill', save=True)

# patch VQVAE'd region back into original image
im_patched = np.copy(im_orig)
temp_reconst = cv2.resize(temp_reconst[0], (240, 135), interpolation = cv2.INTER_AREA)
ShowImage(temp_reconst, title='temp_reconst resized for patching', save=True)
# xrai_max_val = np.unique(xrai_attributions)[xrai_attributions_index]
for i in range(xrai_attributions.shape[0]):
    for j in range(xrai_attributions.shape[1]):
        if (xrai_attributions[i][j] == xrai_max_val):
            im_patched[i][j][:] = (temp_reconst[i][j][:]) * 255

ShowImage(im_patched, title='im_patched', save=True)

# test frankenimage against original image
frankenimg = cv2.resize(im_patched, (240, 135), interpolation = cv2.INTER_AREA)
# frankenimg = np.transpose(frankenimg, (2,0,1))
frankenimg_tensor = PreprocessImages([frankenimg])
print(f"{frankenimg_tensor.shape=} \t {type(frankenimg_tensor)}")
print(f"{frankenimg_tensor.shape=} \t {type(frankenimg_tensor)}")
predictions = model(frankenimg_tensor)
print(f"{predictions=}")
ShowImage(frankenimg, title=f'FrankenImage pred={predictions.detach().numpy().item():.3f}', save=True)

# test frankenimage against original image with noised-out regions