# Diabetic_Retinopathy_Detection_Using_CNN

## Introduction

Diabetic retinopathy (DR) is a leading cause of blindness in individuals with diabetes. Early detection is crucial for preventing vision loss. In some areas, manual screening is challenging due to infrastructure limitations and a shortage of skilled professionals. This project aims to automate DR detection using Machine Learning and Deep Learning algorithms, particularly Convolutional Neural Networks (CNNs). The [APTOS 2019 Blindness Detection dataset](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) from Kaggle, comprising retina images, is utilized for multi-class image classification.

## Background and Related Work

### Ensemble Learning Approach
Research by Niloy et al. [2] employed Ensemble Learning for early blindness detection, achieving 91% accuracy using an ET classifier. This project incorporates some of their preprocessing techniques.

### CNN and Transfer Learning
Carson et al. [3] utilized CNNs and Transfer Learning (VGG16, GoogLeNet, AlexNet) achieving accuracies ranging from 57.2% to 74.5%. This project implements a custom CNN model alongside VGG16, ResNet50, and InceptionV3.

## Methods

### Preprocessing

#### Train/Validation/Test Split
Implemented a validation set approach, and images were organized into class-specific folders for training, validation, and testing.

#### Image Resizing
Ensured uniformity by resizing all images to 200 x 200 pixels.

#### Image Augmentation
Addressed imbalanced class distribution through rotation-based data augmentation.

### Custom Model - CNN Architecture

Developed a CNN model with convolution layers, max pooling, batch normalization, dropout, and dense layers with softmax activation.

### Transfer Learning

Implemented [VGG16](https://www.upgrad.com/blog/basic-cnn-architecture/), [ResNet50](https://www.geeksforgeeks.org/residual-networks-resnet-deep-learning/#), and [InceptionV3](https://iq.opengenus.org/inception-v3-model-architecture/) models for comparison.

## Results/Discussion

### Training Procedure

Mitigated overfitting in the custom CNN model through techniques such as l2 regularization, weight he initialization, and dropout layers. Evaluated models with and without data augmentation.

### Training Results

Observed minimal overfitting in models without augmentation. Data augmentation slightly improved generalization but led to decreased performance in some models.

### Evaluation Results

VGG-16 and ResNet50 (without data augmentation) demonstrated the best performance, achieving 66% accuracy on the validation set. The other two models still performed well. Data augmentation resulted in a performance decrease across all models.

### Labeling Test Data

Models were used to predict test data labels for further analysis.

## Conclusion

The custom CNN model performed competitively with Transfer Learning models. Data augmentation did not significantly improve performance, possibly due to insufficient preprocessing. Future work involves more in-depth domain research, exploring Decision Trees, and extending the project to predict the age of blindness onset.

## Team members:

1. Bhavesh Rajesh Talreja
2. Girish Rajani-Bathija
3. Shriya Prasanna

## References

1. Dataset: [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data)
2. [Research Paper by Niloy et al.](https://arxiv.org/ftp/arxiv/papers/2006/2006.07475.pdf)
3. [Lam C, Yi D, Guo M, Lindsey T. Automated Detection of Diabetic Retinopathy using Deep Learning. AMIA Jt Summits Transl Sci Proc. 2018 May 18;2017:147-155. PMID: 29888061; PMCID: PMC5961805.](https://pubmed.ncbi.nlm.nih.gov/29888061/)
4. [Ramachandran, N., Hong, S.C., Sime, M.J. and Wilson, G.A. (2018), Diabetic retinopathy screening using deep neural network. Clin. Experiment. Ophthalmol., 46: 412-416.](https://doi.org/10.1111/ceo.13056)
5. [Sreng, S.; Maneerat, N.; Hamamoto, K.; Panjaphongse, R. Automated Diabetic Retinopathy Screening System Using Hybrid Simulated Annealing and Ensemble Bagging Classifier. Appl. Sci. 2018, 8, 1198.](https://doi.org/10.3390/app8071198)
6. [GitHub Notebook - APTOS Blindness Detection](https://github.com/adityasurana/APTOS-Blindness-Detection-Kaggle/blob/master/APTOS_Blind_Detection.ipynb)
7. [InceptionV3 Model Architecture](https://iq.opengenus.org/inception-v3-model-architecture/)
8. [Residual Networks (ResNet) - Deep Learning](https://www.geeksforgeeks.org/residual-network)
