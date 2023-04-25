
# Saliency methods being tested:

## DeepLift
[paper]() | [repo]() | [docs]()

Short Description: 

Used previously by: 

Limitations:
- 

## SmoothGrad
[paper](https://arxiv.org/abs/1706.03825) | [repo](https://github.com/PAIR-code/saliency) | [docs](https://pair-code.github.io/saliency/#home) | [lecture](https://www.crcv.ucf.edu/wp-content/uploads/2018/11/lecture-17-SmoothGrad.pdf)

Short Description: "SmoothGrad creates noisy copies of an input image then averages gradients (or another saliency method) with respect to these copies. This often sharpens the resulting saliency map, and removes irrelevant noisy regions."
Google PAIR develops framework-agnostic implementation for state-of-the-art saliency methods (XRAI, BlurIG, SmoothGrad, and more).

Used previously by: ThirdEye

Limitations:
- Has implementations for PyTorch, TensorFlow 2, TensorFlow 1
- Actually an augmentation of other gradient analysis techniques

## Captum


https://captum.ai/api/saliency.html

https://arxiv.org/abs/1312.6034


Short Description: 

Used previously by: 

Limitations: