# MPHY0041 Segmentation Project Technical Report

This study explores prostate cancer segmentation in magnetic resonance imaging (MRI). 
Using data from the [PROMISE 12 Challenge](https://promise12.grand-challenge.org/),
the study seeks to answer the following research question:

> Segmenting prostate glands from axial MR images can be performed either on 2D or 3D
volumetric data. Which one is better?

We created two convolutional neural networks in 
[2D](https://github.com/zcemctt/MPHY0041_Segmentation/tree/master/UNet_2D_Cy) and 
[3D](https://github.com/zcemctt/MPHY0041_Segmentation/tree/master/UNet_3D_Cy). 
The performance of each model was compared against each other using a common clinical metric.  


## Steps to Reproduce Results
The hyperparameters used in both models are included in `.yaml` files 
([2D](https://github.com/zcemctt/MPHY0041_Segmentation/blob/master/UNet_2D_Cy/hyper_parameters.yaml) 
and [3D](https://github.com/zcemctt/MPHY0041_Segmentation/blob/master/UNet_3D_Cy/hyper_parameters.yaml) hyperparameters).
The final hyperparameters used in both models are summarized in the table below:

| Hyperparameters      | [2D Model](https://github.com/zcemctt/MPHY0041_Segmentation/blob/master/UNet_2D_Cy/hyper_parameters.yaml) | [3D Model](https://github.com/zcemctt/MPHY0041_Segmentation/blob/master/UNet_3D_Cy/hyper_parameters.yaml) |
| ----------- | ----------- | ----------- |
| N           | 50          | 50          |
| N_test      | 30          | 30          |

## Roles and Contributions 
