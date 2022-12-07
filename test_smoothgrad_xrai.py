import numpy as np
import torch
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

sys.path.append('models')

device = torch.device("cpu") #torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# saliency/SmoothGrad utils
def ShowImage(im, title='', ax=None, save=False):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im)
    P.title(title)
    P.pause(1)
    if save:
        randstr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
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
    # print(f"before normalization: {torch.max(images)=}")
    images = transformer.forward(images)
    # print(f"after normalization: {torch.max(images)=}")
    # return images.requires_grad_(True)
    return images


# import model
from models.DAVE2pytorch import * #DAVE2pytorch, DAVE2v1, DAVE2v2, DAVE2v3, Epoch
ptfilename = "models/model-DAVE2v1-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt"
model = torch.load(ptfilename).eval().to(device)

# import validation images
img_fqp = "/home/meriel/datasets/sampledir/arrows59.png"
# img_fqp = "/home/meriel/datasets/training_images_industrial-racetrackstartinggate0/hopper_industrial_374.jpg"

# img_fqp = "/home/meriel/datasets/hirochi_raceway-9205-lap-test2/sample-00001.jpg"
# img_fqp = "/home/meriel/datasets/training_images_industrial-racetrackstartinggate0/hopper_industrial_0.jpg"
image_dims = (135, 240)
batch_size=32
im_orig = LoadImage(img_fqp)

# Register hooks for Grad-CAM, which uses the last convolution layer
# vggmodel = models.inception_v3(pretrained=True, init_weights=False)
# print(vggmodel)
print(model)
# conv_layer = vggmodel.Mixed_7c
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

# Load the image
# im_orig = X_valid[0][0] #LoadImage('./doberman.png')
# im_tensor = torch.tensor(np.transpose(im_orig, (2,0,1))[None], dtype=torch.float32) #
im_tensor = PreprocessImages([im_orig])
print(f"{im_tensor.shape=}")
# Show the image
ShowImage(im_orig)

predictions = model(im_tensor)
print(f"{predictions=}")
# predictions = predictions.detach().numpy().item()
prediction_class = predictions
call_model_args = {class_idx_str: prediction_class}

print("Prediction class: " + str(prediction_class))
# im = im_orig.astype(np.float32)
# im = np.transpose(im, (2,0,1))[None]
# print(f"{im.shape=}")
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