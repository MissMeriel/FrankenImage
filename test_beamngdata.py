import numpy as np
import torch
import os
import sys
from models import VQVAE
from models.VQVAE import Encoder, Decoder, VQVAEModel
try:
  import sonnet.v2 as snt
  tf.enable_v2_behavior()
except ImportError:
  import sonnet as snt

import tensorflow.compat.v2 as tf
sys.path.append('models')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# import datset
fqp = "/home/meriel/datasets/automation_test_track-8185-lap-test"
image_dims = (136, 240)
batch_size=32

from models.VQVAEDatasetGenerator import DatasetGenerator
generator = DatasetGenerator([0,1,2], batch_size=10000, dim=(32,32,32), n_channels=1, feature="steering", shuffle=False)
X_valid = generator.process_training_dir(fqp, imgsize=image_dims)
print(type(X_valid[0][0]))
print(f"{X_valid[0][0].shape=}")
# import VQVAE
# vqvaemodel = VQVAE.instantiate_VQVAE(checkpoint_root="merielsvae-checkpoints136x240")

# import base DAVE2 model
from models.DAVE2pytorch import * #DAVE2pytorch, DAVE2v1, DAVE2v2, DAVE2v3, Epoch
ptfilename = "models/model-DAVE2v1-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt"
dave2model = torch.load(ptfilename).eval().to(device)

# convert pytorch to onnx
x = torch.randn(batch_size, 3, 135, 240, requires_grad=True).to(device)
torch_out = dave2model(x)
onnxfilename = "testconverter.onnx"
torch.onnx.export(dave2model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  onnxfilename,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=8,          # the ONNX version to export the model to
                  do_constant_folding=False,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                #   dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                #                 'output' : {0 : 'batch_size'}}
                                )

# convert onnx to keras 2.0
import onnx, onnx2keras
onnx_model = onnx.load(onnxfilename)
k_model = onnx2keras.onnx_to_keras(onnx_model, ['input'])

# save as keras 2.0 model
# from pytorch2keras import pytorch_to_keras
# from torchvision.transforms import Compose, ToTensor #, Resize, Lambda, Normalize
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

# use SmoothGrad/DeepLift to find salient regions
# import deeplift
# from deeplift.conversion import kerasapi_conversion as kc
# deeplift_model =\
#     kc.convert_model_from_saved_files(
#         ptfilename,
#         nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.DeepLIFT_GenomicsDefault) 
# find_scores_layer_idx = 0
# deeplift_contribs_func = deeplift_model.get_target_contribs_func(
#                             find_scores_layer_idx=find_scores_layer_idx,
#                             target_layer_idx=-1)
# scores = np.array(deeplift_contribs_func(task_idx=0,
#                                          input_data_list=[X],
#                                          batch_size=10,
#                                          progress_update=1000))
# print(deeplift_model.get_name_to_layer().keys())

# noise-out salient regions

# run through VQVAE

# patch VQVAE'd region back into original image

# test frankenimage against original image

# test frankenimage against original image with noised-out regions
