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
The hyperparameters used in both models are included in `.yaml` files. The final hyperparameters used in both models are summarized in the table below:

| Hyperparameters      | [2D Model](https://github.com/zcemctt/MPHY0041_Segmentation/blob/master/UNet_2D_Cy/hyper_parameters.yaml) | [3D Model](https://github.com/zcemctt/MPHY0041_Segmentation/blob/master/UNet_3D_Cy/hyper_parameters.yaml) |
| ----------- | ----------- | ----------- |
| learning_rate  | 1e-4        | 1e-4        |
| epochs         | 1000        | 1000        |
| val_size       | 0.1         | 0.1         |
| dropout        | 0.5         | 0.5         |
| batch_size     | 16          | 4           |
| patience       | 100         | 100         |

where `learning_rate` is the step size at each iteration while minimizing a loss function, `epochs` is the maximum number
of epochs used to train the model, `val_size` is the proportion of the training dataset (`N=50`) used for test validation,
`dropout` is the proportion of neurons that is randomly selected to be ignored during training (for regularization),
`batch_size` is the number of training examples used in one iteration, and `patience` is the number of epochs with no
model improvement for early-stopping.

To train the model, run the `train.py` script for each model. We encourage using TensorFlow-GPU to train the model in a
reasonable duration of time. The files to run TensorBoard are saved in the `logs` directory. Run the following code to view results on TensorBoard: `tensorboard --logdir=logs/ --host localhost --port 8088`. The predicted masks from the model are compared to the true masks in the training dataset for each sample and slice in the `plots_training` directory. The predicted masks are plotted alongside the MR image in the `plots_testing` directory. There is also a `plots_training_overlay` directory that overlays the predicted and true masks on the MR image for each slice.

## Roles and Contributions 

The roles and contributions of each team member are summarized in the table below:

| Name                      | Responsibilities          |
|---------------------------|---------------------------|
| Guglielmo Pellegrino      | Cleaning of code; commenting; importing code to [Colab](https://colab.research.google.com/drive/1QF9A59y2OcXYU6PlmKnDCMcDXJnNzo8A); dropout optimization |
| Aman Ganglani             | Architecture research and documentation; TensorFlow-GPU installation; model training |
| Cyrus Tanade              | Wrote original 2D and 3D segmentation code; wrote the [Technical Report](https://github.com/zcemctt/MPHY0041_Segmentation/blob/master/README.md); hyperparameter optimization; maintenance of [GitHub Repository](https://github.com/zcemctt/MPHY0041_Segmentation) |
| Jack Weeks                | Model optimization (data augmentation and regularization); hyperparameter optimization |
| Nikita Jesaibegjans       | Background research; computational metrics; hyperparameter optimization |
| Josephine Windsor-Lewis   | Wrote mask overlay code; clinical metrics; group project management |
