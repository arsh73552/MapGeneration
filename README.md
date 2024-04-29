---
title: RoadMapGAN: Generating Road Maps from Satellite Images using GANs
---

# RoadMapGAN: Generating Road Maps from Satellite Images using GANs

Welcome to RoadMapGAN, a project aimed at generating road maps from satellite images using Generative Adversarial Networks (GANs) with a U-Net architecture. This README provides an overview of the project, its objectives, architecture, usage, and testing procedures.

## Overview
RoadMapGAN utilizes deep learning techniques to generate accurate road maps from satellite images. The project leverages the power of GANs, particularly the U-Net architecture, to generate realistic road maps. The discriminator network is designed to distinguish between real road maps and generated ones, with patch-based discrimination to focus on specific regions of the image.

## Architecture
The architecture of RoadMapGAN consists of two main components:

1. **Generator (U-Net)**: The generator takes satellite images as input and generates corresponding road maps. The U-Net architecture is employed due to its effectiveness in image-to-image translation tasks.

2. **Discriminator**: The discriminator network is responsible for discerning between real road maps and generated ones. It utilizes patch-based discrimination to focus on specific regions of the images, enhancing the model's ability to produce high-quality outputs.

## Requirements
To run the RoadMapGAN project, ensure you have the following dependencies installed:
- Python 3.x
- PyTorch
- NumPy
- pytest (for running unit tests)

## Usage
1. **Clone the Repository**:


The project utilizes Generative Adversarial Networks and the U-Net network architecture in order to convert satellite images to Road Views. This project has a variety of use cases as it is dynamic in nature. That is, environmental changes can be detected using this model. For instance the model can be used for Flood detection, Vegetation cover check, etc.

### Required Modules
    1. Pytorch = v1.11.0+cu113
    2. Matplotlib = v3.5.2
    3. tqdm = v4.64.1
    4. torchvision = v0.12.0+cu113
    5. PIL = v9.1.1

### Example

Prediction               || Satellite Image                || Real Output

![Can't Load Image](https://github.com/arsh73552/MapGeneration/blob/main/exampleOut.jpg)


## Citation

If you find this project useful, please cite as the inspiration for the project has been taken from the paper mentioned below:

> Olaf Ronneberger 
> Philipp Fischer 
> Thomas Brox
> U-Net: Convolutional Networks for Biomedical Image Segmentation
> https://doi.org/10.48550/arXiv.1505.04597

### License

<a href = 'MIT-LICENSE.txt'>License</a>


