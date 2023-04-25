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

# import model

# import validation images

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