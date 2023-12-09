Pneumonia Detection Using Convolutional Neural Network

## Overview
Pneumonia is an inflammatory condition of the lung primariy affecting the small air sacs known as alveoli in one or both lungs. It can be caused by infection with viruses or bacteria; and identifying the pathogen responsible for Pneumonia could be highly challenging.

Diagnosis of Pneumonia often starts with medical history and self reported symptoms, followed by a physical exam that usually includes chest auscultation. A chest radiograph would then be recommended if the doctors think the person might have Pneumonia. In adults with normal vital signs and a normal lung examination, the diagnosis is unlikely.

## Business Problem
To develop a robust learning model capable of accurately classifying images into predefined catergories.For this project, we have developed and evaluated various Convolutional Neural Networks that can quickly classify Normal vs. Pneumonia frontal chest radiographs. The implementation of these models could help alert doctors and radiologists on classifying those images with pneumonia from those that are normal.

## Dataset
The dataset used was from Guangzhou Women and Children’s Medical Center of one to five years old downloaded from Kaggle
The diagnoses for the images were then graded by two expert physicians, and checked by a third expert before being cleared for training the AI system.

## Dataset Structure
 The dataset is organized into 3 folders (train, test, val)and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories(Pneumonia/Normal)

 ## Data Preview




![ALT TEXT](https://github.com/Yaqi-graphics/group-25-phase-4-project-moringa/blob/Mourine/Image%205.PNG)










 ## Modeling
 Convolutional Neural Networks (CNN)

 In CNN,there are layers called Convolutional layers which serves as the building block of CNN.Neurons in the first layer are only connected to pixels in their receptive fields.Then each neuron in the second convolutional layer is only connected to neurons within their receptive field in the first layer, and so on. This hierarchical structure resembles real-world images

 Pooling Layers they subsample the input image inorder to reduce computational load, memory usage, and number of parameters.

 #Model construction.
 for our modelling,Image Augmentation Using PNet was used which consisted of only five convolution blocks, each followed by a max‐pooling layer. Even though PNet has a smaller number of parameters, it outperforms both the AlexNet and VGG 16 in the pneumonia detection

 We can load the data using Pnet model that has been pertrained on imagenet as follows:
 image_gen = ImageDataGenerator(
        rotation_range = 45,
        rescale = 1./255)
Then the batch was joined to create a dataset
    X_train_aug, y_train_aug = next(train_generator)
    X_test, y_test = next(test_generator)
    X_val, y_val = next(val_generator)

After feature extraction, the learned convolutional feature weights are flattened and sent to two fully-connected layers for label prediction. Spatial pooling is carried out by five max-pooling layers, which follow each of the convolutional layers. Max-pooling is performed over a 2 × 2 pixel window to enhance the generalization ability of the network."

Start the beginning of training:
history = model.fit(X_train_aug,
                       y_train_aug,
                       epochs=20,
                       batch_size=20,
                       callbacks= [callbacks],
                       class_weight= weights,
                       validation_data=(X_val, y_val))

##Visualizing the results for y_test.
























##Transfer Learning
Using DenseNet as our base, we'll try using our CNN on top of a pretrained model to see if we get better results and this was used to achieve the intended results:
base_model = DenseNet121(input_shape=(224, 224, 3),
                         include_top=False,
                         pooling='avg'
## Model evaluation
Although all 4 models have very high overall accuracy, cnn_4 has higher Accuracy, Recall and Precision across the board.














## Conclusion
Based on our cm numbers,our PNet with image augmentation and weights was our best model so far, with 179 TP (actual normal) and 55 FP (predicted normal), and 42 FN(predicted pneumonia), and 348 TN(actual pneumonia). While we can be reasonably (about 80% sure) that our model will produce an accurate prediction, for this use case, we would want much better outcomes for our precision and recall before deploying our model at large.
Rather than purely pursuing better metric scores, it'd be best to take advantage of experts' domain knowledge, and have these outputs reviewed by clinicians and radiologists who can provide input on whether or not the model has identified correct/potential regions the chest area that might be indicators of Pneumonia.

## Recommendation
In order to employ our model at scale with confidence, we'd want to improve our models detection of pneumonia by incorporating Object Detection/Localization into the models so that the output would not only be whether or not the X-ray exhibit abnormal pulmonary patterns typically observed in Pneumonia, but also the location of the identified patterns.
