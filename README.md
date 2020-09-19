# scratch-detection
Tensorflow U-Net model for semantic segmentation of scratches from damaged films.

## Introduction

The goal of this project is to aid the restoration of damaged film frames from old movies by segmenting scratches in the image. Figure 1 shows an example of a damaged frame with several vertically oriented scratches. Deep learning is utilized for this task.

## Network Structure

The network architecture used is the U-Net model, which is commonly used in medical semantic segmentation tasks (Ronneberger, Fischer, & Brox, 2015). The model consists of two main portions; a contractive path, and an expansive path. The architecture of the model is illustrated in Figure 2.

The contractive path consists of repeating blocks, each containing two convolutional layers, a dropout layer, and a max pooling layer. The convolutional layers convolve the input image using a kernel to extract image features. The model learns the best kernels to use to extract the most relevant features through training. The dropout layer is a form of regularization that prevents the model from ‘memorizing’ the mapping between input and output. This is important because models that do this tend to generalize poorly when presented with input they have never seen before. Finally, the max pooling layers reduce the dimensionality of the input as it passes through each block, while preserving important information about the image. This has two purposes. The first is that this reduces the training time because it reduces the number of parameters the model must learn. Second, it essentially increases the size of the neighborhood that a convolutional layer operates on. The result is that the first blocks tend to extract low level features such as edges and texture, while the deeper layers extract high level features such as shapes. Normally, this structure is enough to classify the image using labels (ie, what class the image is). However, to label the image pixel-by-pixel, the expansive path is necessary.

The expansive path is essentially a reflection of the contractive path, except that the data is upsampled as it moves through the convolutional blocks. This upsampling is performed by a special layer called a transpose convolutional layer. Intuitively, the purpose of this layer is to reverse the convolutions that have occurred in the contractive path such that an image is reconstructed (ie., this tells the model where the class is, as opposed to just what it is). To improve the performance of this operation, the output of the transpose convolutional layer is concatenated with the output of the corresponding convolutional block. This aids in reconstructing the small details of the image. 

 
Figure 2 U-Net architecture.

## Training Data

62 scratched film stills of dimension 1080x1920 were provided. Scratches in the image were labeled manually by the class. The image pairs were split into a training set (80% of the data) and a validation set (20% of the data). 

Feeding the raw images into the model for training would result in a very long training time due to the large dimensions. To compensate for this, smaller random patches of the images were generated. The patch size used is 128x128. This also allows the data to be augmented because the number of possible patches outnumbers the 62 frames. Here, 20,000 patches were generated and split into training and testing sets. An example set of the patches is shown in Figure 3.
 
Figure 3 Sample of randomly generated 128x128 image patches and corresponding mask.

An important consideration is the sparsity of the scratches in the image. For any given image (or patch), the number of pixels representing the scratches is much smaller than the number of non-scratch pixels. This presents bias into the model. To mitigate this, the number of empty image masks (ie. no scratches in the patch) allowed was limited to a maximum of half the generated patches.

## Measuring Loss

Since this is a binary classification problem (ie., scratch vs not-scratch), binary cross-entropy was used as the loss function in the model. To further account for the sparsity of the scratches, a weighted version of binary cross-entropy was used. This penalizes misclassification of a scratch more heavily than that of the image background.  

## Training

The model was implemented using the TensorFlow Keras API. Early stopping was used to detect when further training would yield no improvement, as measured by the loss value for each epoch. The loss value eventually reaches a plateau as shown in Figure 4. Figure 5 shows the accuracy of the model after each epoch. 


 
Figure 4 Training loss (orange) and validation loss (blue) per epoch.
 
Figure 5 Model accuracy for the training set (orange) and validation set (blue).
Results

The model was successful in detecting most large scratches in the image with an accuracy of 96%. Figure 6 shows an example of the resulting mask. The accuracy of the model is limited by a few factors:

•	Since the frames were manually segmented by the class, each person likely had a different approach to finding and labeling the images. For example, some people ignored very small scratches and focused on very large ones, while others counted the small scratches. This could be improved upon by standardizing the manual labeling procedure.
•	The masks generated by hand are also subject to human error. Regardless of the procedure used by each person, each person will also segment with a certain amount of error. This can be avoided by training the labelers.

 
Figure 6 Sample result of predicted scratch mask.
 
 
Figure 9 Segmentation result from project 2

## References
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.

