GestureFlow:
Real-time hand gesture Recognition using ResNet50
(A Deep Learning Approach)


Shivani Teotia                           Om Sisodia                       Sakshi Goyal                      Siddhant Singh
E21CSEU0588                          E21CSEU0459                   E21CSEU0544                    E21CSEU0614 
 
 
Abstract— Real-time gesture recognition has gained significant attention in recent years due to its diverse applications in human-computer interaction, virtual reality, gaming, and robotics. Deep learning techniques, particularly Convolutional Neural Networks (CNNs), have shown remarkable performance in recognizing hand gestures from image data. This paper presents a real-time gesture recognition system utilizing the powerful ResNet50 architecture. The proposed system achieves high accuracy and efficiency, making it suitable for real-world applications. Experimental results demonstrate the effectiveness of ResNet50 in accurately recognizing hand gestures in real-time scenarios.
    Keywords—Gesture Recognition, Real-time, ResNet50, Deep Learning, Image Classification
I.	INTRODUCTION 
Hand gesture recognition is a crucial component of human-computer interaction systems, allowing users to interact with devices and interfaces in a natural and intuitive manner. From controlling virtual environments to assisting individuals with disabilities, gesture recognition systems have found diverse applications in various domains. With the advent of deep learning, particularly Convolutional Neural Networks (CNNs), the field of gesture recognition has witnessed significant advancements in accuracy and efficiency.
Traditional methods of gesture recognition often relied on hand-crafted features and machine learning algorithms, which were limited in their ability to capture complex patterns and variations in hand movements. However, deep learning approaches, especially CNNs, have demonstrated superior performance by automatically learning hierarchical features from raw input data.
Among the various deep learning architectures, Residual Networks (ResNets) have emerged as a powerful class of models for image recognition tasks. ResNet50, a variant of ResNet, has shown exceptional performance in image classification, object detection, and semantic segmentation tasks. Its unique architecture with skip connections allows the network to effectively learn residual mappings, thereby mitigating the vanishing gradient problem and enabling the training of very deep networks.
In this paper, we propose a real-time gesture recognition system based on the ResNet50 architecture. The system 
aims to accurately classify hand gestures in real-time scenarios, allowing for seamless interaction between users and devices. By leveraging the capabilities of ResNet50, our system achieves high accuracy while maintaining real-time performance, making it suitable for applications such as sign language recognition, human-computer interaction interfaces, and gesture-controlled devices.
II.	RELATED WORK
A.	Gesture Recognition Using Deep Learning
In recent years, deep learning models, particularly Convolutional Neural Networks (CNNs), have shown remarkable success in various computer vision tasks, including gesture recognition. Deep learning models can automatically learn discriminative features from raw visual data, enabling highly accurate and robust recognition of gestures [1].
B.	CNNs for Gesture Recognition
CNNs have been extensively applied to gesture recognition tasks due to their ability to capture spatial features effectively. Initially, 2D CNNs were employed to process video frames as multi-channel inputs, achieving notable success in recognizing static gestures [1]. For instance, Temporal Segment Network (TSN) [1] divides videos into segments and utilizes 2D CNNs to extract features for action recognition.
C.	Introduction to 3D CNNs
Recognizing the importance of temporal information in gesture recognition, researchers introduced 3D CNNs, which utilize 3D convolutions and pooling to capture both spatial and temporal features directly from video sequences [1]. 3D CNNs take a sequence of video frames as inputs and have demonstrated superior performance compared to 2D CNNs in capturing motion patterns and temporal dynamics.
D.	Real-Time Gesture Recognition
Real-time gesture recognition systems require efficient processing of continuous video streams to detect and classify gestures promptly. Several approaches have been proposed to address this challenge. Some methods focus on detection and classification separately [1], while others aim to achieve both simultaneously [2]. However, ensuring single-time activations for performed gestures remains a critical issue, which has not been adequately addressed.
E.	ResNet50 for Gesture Recognition
To improve both accuracy and efficiency in gesture recognition, the Residual Network (ResNet50) architecture has gained attention. ResNet50 employs residual connections to address the vanishing gradient problem, enabling the training of deeper networks with improved performance [2]. Its deep architecture and ability to learn complex features make it suitable for gesture recognition tasks.
F.	Contributions of ResNet50 in Gesture Recognition
In this study, we leverage ResNet50 for real-time hand gesture recognition. Unlike previous approaches that focus solely on offline classification accuracy, our proposed method integrates offline working models into a hierarchical architecture to achieve real-time operation while maintaining high accuracy. We utilize a lightweight CNN for gesture detection and ResNet50 as a classifier, ensuring efficient and accurate recognition of dynamic gestures [2].

III.	METHODOLOGY
A.	Data Collection and Preprocessing
The dataset used in this study consists of binary black and white images of hand gestures obtained from a Kaggle dataset. Each image depicts a hand gesture against a black background, where the hand is represented in white. The dataset contains six types of hand gestures: 'blank', 'fist', 'five', 'ok', 'thumbsdown', and 'thumbsup'.
To preprocess the images:
The binary images are resized to 64x64 pixels to standardize the input size for model training.
As the images are already binary and contain only the hand gesture against a black background, no further preprocessing steps such as color normalization or background removal are required.
 
Fig. 1.	The above figure illustrates the 6 types of classes/gestures in the dataset
B.	Data Splitting
The dataset is split into training, testing, and evaluation sets using a stratified split. The training set comprises 80% of the data, while 10% is allocated for testing and 10% for evaluation. This ensures a balanced representation of each gesture class in all three sets.
C.	Model Architecture Selection
 
Fig. 2.	Resnet50Architecture

The ResNet50 architecture is chosen as the base model due to its effectiveness in feature extraction from images. Despite the binary nature of the images, ResNet50 is capable of learning meaningful features, which can aid in gesture recognition.
D.	Model Modification and Compilation
A custom classifier head is added on top of the ResNet50 base. This head consists of a Flatten layer followed by a Dropout layer with a dropout rate of 0.6 to reduce overfitting. Finally, a Dense layer with softmax activation is used to output the classification probabilities for the six gesture classes.
E.	Model Training
The compiled model is trained on the training set for 50 epochs with a batch size of 64. During training, the model's performance is monitored using both the training and evaluation datasets to detect overfitting.
TABLE I. 	DATASET SPLITS
Dataset	Number of images
Training	7430
Evaluation	1858
Test	1033

F.	Model Evaluation
The trained model's performance is evaluated on both the testing and evaluation datasets to assess its generalization capability. The accuracy metric is used to measure the model's performance on both datasets.
             
where 𝑇𝑃, 𝐹𝑃, and 𝐹𝑁 denote the number of true positives, false positives, and false negatives, respectively.
The following results are observed.      

 
Fig. 3.	Accuray of train and test data 
 
Fig. 4.	Model loss 

The model achieved an accuracy of 99.892% on the evaluation dataset and 99.806% on the test dataset.
TABLE II. 	ACCURACY OF TEST AND EVALUATION DATA
Dataset	Accuracy
Test Data	99.806%
Evaluation Data	99.892%

Here are some observed results the following images are extracted from a real-time video used during the testing phase.
 
Fig. 5.	The above figure illustrates the images extracted from the real-time video and the corresponding results
The model was able to predict all the above correctly during testing, results might vary on the quality of the camera, lighting and other external factors.
IV.	CONCLUSION
In this study, we developed a real-time gesture recognition system using the powerful ResNet50 architecture. The system has been trained and tested on a dataset consisting of binary black and white images, where the hand gestures are represented by white pixels on a black background. By leveraging Convolutional Neural Networks (CNNs), particularly ResNet50, we achieved high accuracy and efficiency in recognizing hand gestures, making our system suitable for a wide range of real-world applications.
Our experimental results demonstrated the effectiveness of ResNet50 in accurately recognizing hand gestures in real-time scenarios. The model achieved an accuracy of 99.806% on the test set and 99.892% on the evaluation set. This high level of accuracy indicates the robustness of our system in distinguishing between different hand gestures.
Overall, the developed gesture recognition system shows promising results and can be further optimized and deployed in various domains such as human-computer interaction, virtual reality, gaming, and robotics to enhance user experience and interaction.


REFERENCES
[1]	Real-time Hand Gesture Detection and Classification Using Convolutional Neural Networks Okan Kop¨ ukl ¨ u¨ 1 , Ahmet Gunduz1 , Neslihan Kose2 , Gerhard Rigoll1 1 Institute for Human-Machine Communication, TU Munich, Germany 
[2]	 Practice of Gesture Recognition Based on Resnet50 Zhiming Li 2020 J. Phys.: Conf. Ser. 1574 01 

