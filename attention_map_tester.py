import numpy as np
import torch
import os
import sys
from models import VQVAE
from models.VQVAE import Encoder, Decoder, VQVAEModel
# try:
#   import sonnet.v2 as snt
#   tf.enable_v2_behavior()
# except ImportError:
#   import sonnet as snt

# import tensorflow.compat.v2 as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

sys.path.append('models')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# import datset
fqp = "/home/meriel/datasets/automation_test_track-8185-lap-test"
image_dims = (136, 240)
batch_size=32

# from models.VQVAEDatasetGenerator import DatasetGenerator
# generator = DatasetGenerator([0,1,2], batch_size=10000, dim=(32,32,32), n_channels=1, feature="steering", shuffle=False)
# X_valid = generator.process_training_dir(fqp, imgsize=image_dims)
# print(type(X_valid[0][0]))
# print(f"{X_valid[0][0].shape=}")

from tensorflow.keras.preprocessing.image import img_to_array
import scipy.misc
import os
from imutils import paths
import cv2
from tensorflow.keras.models import load_model
import time
ptfilename = "../Self-Driving-Car-Keras/model.h5"
model = load_model(ptfilename)

imgs = os.listdir(fqp)
imgs.remove("data.csv")

# test images on pretrained dave2 
# for img in X_valid[0]:
#   img = cv2.resize(img, dsize=(200, 66), interpolation=cv2.INTER_CUBIC)
#   steer = model(img[None])
#   print(steer[0][0])

# save as keras 2.0 model
# from pytorch2keras import pytorch_to_keras
from torchvision.transforms import Compose, ToTensor #, Resize, Lambda, Normalize
# transform=Compose([ToTensor()])
# input_img = transform(X_valid[0][0]).float()
# print(f"{input_img.shape=}")
# k_model = pytorch_to_keras(dave2model.to(device), input_img[None].to(device), [(10, 32, 32,)], verbose=True) 

# from pt2keras import Pt2Keras
# import tensorflow as tf
# input_shape = (1, 3, 135, 240)
# converter = Pt2Keras()
# keras_model: tf.keras.Model = converter.convert(dave2model.to(torch.device("cpu")), input_shape)
# keras_model.save('output_model.h5')


# determine faulty/OOD images



## DEEPLIFT IS MISSING SUPPORT FOR ELU ACTIVATION FUNCTION
## OTHER OPTIONS SEE HERE: https://github.com/kundajelab/deeplift#my-model-architecture-is-not-supported-by-this-deeplift-implementation-what-should-i-do
# # use DeepLift to find salient regions
# import deeplift
# from deeplift.conversion import kerasapi_conversion as kc
# deeplift_model = kc.convert_model_from_saved_files(
#         ptfilename,
#         nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.DeepLIFT_GenomicsDefault
#         ) 
# find_scores_layer_idx = 0
# deeplift_contribs_func = deeplift_model.get_target_contribs_func(
#                             find_scores_layer_idx=find_scores_layer_idx,
#                             target_layer_idx=-1)
# scores = np.array(deeplift_contribs_func(task_idx=0,
#                                          input_data_list=[img],
#                                          batch_size=10,
#                                          progress_update=1000))
# print(deeplift_model.get_name_to_layer().keys())


# Use CAPTUM to find salient regions
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)
model = torch.load("models/model-DAVE2v1-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt").eval().to(device)
img = cv2.imread(fqp + "/sample-00100.jpg")
input = np.array(img)
print(f"{input.shape=}")
transform=Compose([ToTensor()])
input = transform(input)[None].float()
baseline = torch.zeros(input.shape).float()
print(f"{input.shape=}")
print(f"{baseline.shape=}")
ig = IntegratedGradients(model)
attributions, delta = ig.attribute(input.to(device), baseline.to(device), target=0, return_convergence_delta=True)
print('IG Attributions:', attributions)
print('Convergence Delta:', delta)
print(f"{type(attributions)=}\n{attributions.shape=}\n{type(delta)=}\n{delta.shape=}")
saveimg = attributions.cpu()[0].permute(1,2,0).numpy()
saveimg = (saveimg-np.min(saveimg))/(np.max(saveimg)-np.min(saveimg))
print(f"{saveimg.shape=}\n{np.min(saveimg)=}\n{np.max(saveimg)=}")

saveimg = saveimg
cv2.imwrite("temp.jpg", saveimg*255)
print(f"{input.shape=}")
print(f"{model(input.to(device))}")
# # use SmoothGrad to find salient regions
# import saliency.core as saliency
# import tensorflow as tf
# ...
# # call_model_function construction here.
# def call_model_function(x_value_batched, call_model_args, expected_keys):
# 	tape = tf.GradientTape()
# 	grads = np.array(tape.gradient(output_layer, images))
# 	return {saliency.INPUT_OUTPUT_GRADIENTS: grads}
# ...
# # Load data.
# image = GetImagePNG(...)
# # Compute IG+SmoothGrad.
# ig_saliency = saliency.IntegratedGradients()
# smoothgrad_ig = ig_saliency.GetSmoothedMask(image, 
# 											call_model_function, call_model_args=None)
# # Compute a 2D tensor for visualization.
# grayscale_visualization = saliency.VisualizeImageGrayscale(
#     smoothgrad_ig)

# noise-out salient regions

# run through VQVAE

# patch VQVAE'd region back into original image

# test frankenimage against original image

# test frankenimage against original image with noised-out regions
