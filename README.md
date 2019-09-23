# Realtime_facedetection
This project is the integration of three real-time face detection functions, face recognition, age and gender detection.

# Dependence
1. OpenCV-3.4.3
2. OpenCV-3.4.3-contrib
3. C++11
4. Tensoflow-1.14
5. Ubuntu 16.04

# Description
There are four main pretrained models below; 
face_detector from Tensorflow, age_net and gender_net from Caffe, face_net from Torch.
This project attempts to integrate these pretrained models to build up the multi-functional camera product,
in the near future it can be deployed on APP and Intel architecture platform with Qt and OpenVINO.

# Demo
In demo_image folder there are four demo images for face detection on different conditions.
In is clear that this face detection is not influenced by light and hat.

# TODO
1. Resolve the bug for the ROI when running four models at the same time
2. Improve the backend computaiton by OpenVINO instead of OpenCV
3. Deploy this poject on APP by Qt development
4. The use of transfer Learning for fine-tuning on different objective detection

# Reference
1. FaceNet: A Unified Embedding for Face Recognition and Clustering, https://arxiv.org/abs/1503.03832
2. Age and Gender Classification Using Convolutional Neural Networks
