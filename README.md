# YoloTeeth-X-Ray-Instance-Segmentation-Object-Detection-with-YOLOv8

<!-- Repository Overview Badges -->
<div align="center">
    <img src="https://img.shields.io/github/stars/arpsn123/YoloTeeth-X-Ray-Instance-Segmentation-Object-Detection-with-YOLOv8?style=for-the-badge&logo=github&logoColor=white&color=ffca28" alt="GitHub Repo Stars">
    <img src="https://img.shields.io/github/forks/arpsn123/YoloTeeth-X-Ray-Instance-Segmentation-Object-Detection-with-YOLOv8?style=for-the-badge&logo=github&logoColor=white&color=00aaff" alt="GitHub Forks">
    <img src="https://img.shields.io/github/watchers/arpsn123/YoloTeeth-X-Ray-Instance-Segmentation-Object-Detection-with-YOLOv8?style=for-the-badge&logo=github&logoColor=white&color=00e676" alt="GitHub Watchers">
</div>

<!-- Issue & Pull Request Badges -->
<div align="center">
    <img src="https://img.shields.io/github/issues/arpsn123/YoloTeeth-X-Ray-Instance-Segmentation-Object-Detection-with-YOLOv8?style=for-the-badge&logo=github&logoColor=white&color=ea4335" alt="GitHub Issues">
    <img src="https://img.shields.io/github/issues-pr/arpsn123/YoloTeeth-X-Ray-Instance-Segmentation-Object-Detection-with-YOLOv8?style=for-the-badge&logo=github&logoColor=white&color=ff9100" alt="GitHub Pull Requests">
</div>

<!-- Repository Activity & Stats Badges -->
<div align="center">
    <img src="https://img.shields.io/github/last-commit/arpsn123/YoloTeeth-X-Ray-Instance-Segmentation-Object-Detection-with-YOLOv8?style=for-the-badge&logo=github&logoColor=white&color=673ab7" alt="GitHub Last Commit">
    <img src="https://img.shields.io/github/contributors/arpsn123/YoloTeeth-X-Ray-Instance-Segmentation-Object-Detection-with-YOLOv8?style=for-the-badge&logo=github&logoColor=white&color=388e3c" alt="GitHub Contributors">
    <img src="https://img.shields.io/github/repo-size/arpsn123/YoloTeeth-X-Ray-Instance-Segmentation-Object-Detection-with-YOLOv8?style=for-the-badge&logo=github&logoColor=white&color=303f9f" alt="GitHub Repo Size">
</div>

<!-- Language & Code Style Badges -->
<div align="center">
    <img src="https://img.shields.io/github/languages/count/arpsn123/YoloTeeth-X-Ray-Instance-Segmentation-Object-Detection-with-YOLOv8?style=for-the-badge&logo=github&logoColor=white&color=607d8b" alt="GitHub Language Count">
    <img src="https://img.shields.io/github/languages/top/arpsn123/YoloTeeth-X-Ray-Instance-Segmentation-Object-Detection-with-YOLOv8?style=for-the-badge&logo=github&logoColor=white&color=4caf50" alt="GitHub Top Language">
</div>

<!-- Maintenance Status Badge -->
<div align="center">
    <img src="https://img.shields.io/badge/Maintenance-%20Active-brightgreen?style=for-the-badge&logo=github&logoColor=white" alt="Maintenance Status">
</div>



YoloTeeth represents a significant advancement in the realm of dental image analysis, leveraging the state-of-the-art **YOLOv8** architecture for instance segmentation and object detection of teeth in X-ray images. This repository sets a new benchmark in dental radiography, facilitating improved diagnostic capabilities and supporting rigorous research initiatives by accurately identifying and delineating individual teeth. The project aims to provide dental professionals with enhanced tools for diagnosing and understanding oral health through detailed image analysis.

## 🚀 Technologies Used

- ![YOLOv8](https://img.shields.io/badge/yolov8-0.1.0-orange.svg) **YOLOv8**: The primary model utilized for real-time object detection and segmentation, featuring a single-pass architecture designed to optimize both performance and speed. YOLOv8 is known for its accuracy and efficiency, making it suitable for complex tasks in medical imaging.
  


- ![PyTorch](https://img.shields.io/badge/pytorch-1.9.0-red.svg) **PyTorch**: The deep learning framework employed for model implementation and training. PyTorch's flexibility and ease of use allow for quick iterations and modifications, making it an excellent choice for developing machine learning models.


- ![OpenCV](https://img.shields.io/badge/opencv-4.5.1-brightgreen.svg) **OpenCV**: An essential image processing library utilized for managing and processing dental images. OpenCV provides numerous functionalities, such as image filtering, transformations, and augmentations, crucial for preparing data for model training.


- ![TensorFlow](https://img.shields.io/badge/tensorflow-2.6.0-lightgrey.svg) **TensorFlow**: While primarily focused on YOLOv8, TensorFlow may be employed for additional model training or evaluation tasks, providing versatility in handling various machine learning tasks.



## 📊 Dataset

The dataset employed in this project consists of **269 dental X-ray images**, along with their corresponding annotations, derived from my previous project: [Dental-X-RAY-Image-Detection-and-Instance-Segmentation](https://github.com/arpsn123/Dental-X-RAY-Image-Detection-and-Instance-Segmentation.git).

- **Training Set**: 
  - A comprehensive collection of 269 images, each converted into individual **YOLOv8 PyTorch TXT** files, resulting in 269 .txt files that provide the necessary data for training the model. This format ensures that the model can efficiently read and interpret the training data during the learning process.

- **Validation Set**: 
  - The validation dataset consists of 5 images, with their annotations also converted into YOLOv8 .txt files, resulting in 5 .txt files. This validation set is crucial for assessing the model's performance and ensuring that it generalizes well to unseen data.

## 📈 Model Overview

**YOLOv8** distinguishes itself through its efficient architecture, predicting bounding boxes and class probabilities in a single inference pass. This model is designed to optimize the trade-off between speed and accuracy, incorporating advanced techniques such as a robust backbone network and a Feature Pyramid Network (FPN) for multi-scale feature extraction. The FPN enhances the model's ability to detect objects at various scales, making it particularly effective for analyzing dental images, where tooth sizes and positions can vary significantly.

![Training Batch Example](https://github.com/arpsn123/YoloTeeth-X-Ray-Instance-Segmentation-Object-Detection-with-YOLOv8/assets/112195431/279db196-66e0-4da9-a69d-84c7a4498873)

## 📊 Performance Evaluation

The evaluation of YOLOv8's performance is conducted through meticulous monitoring of loss and accuracy curves throughout the training process.

- **Loss Curve**: This curve tracks the model's ability to minimize the error between predicted bounding boxes and the ground truth annotations. Metrics such as mean squared error or binary cross-entropy loss are typically employed to gauge this performance. A decreasing loss indicates that the model is learning effectively, while plateaus or increases in loss may signal the need for further adjustments in training parameters.

- **Accuracy Curve**: Although accuracy is less commonly emphasized in object detection, metrics such as Intersection over Union (IoU) and mean Average Precision (mAP) are crucial for assessing the model’s detection capabilities. Continuous monitoring of these metrics is essential for guiding model optimizations and training adjustments, ultimately enhancing object detection performance and addressing potential overfitting.

Monitoring these metrics allows for strategic adjustments to the training process, helping to ensure that the model can robustly identify and classify teeth in X-ray images, which is vital for effective dental diagnosis.

![Performance Results](https://github.com/arpsn123/YoloTeeth-X-Ray-Instance-Segmentation-Object-Detection-with-YOLOv8/assets/112195431/24ecb4c0-724a-4ee8-bb94-95896471b61d)

## 📁 Integration of `data.yaml` in YOLOv8 Training

The `data.yaml` file is integral to the training process of YOLOv8, encapsulating critical metadata and configuration parameters associated with the dataset. This file facilitates the model's access to training and validation images and defines the number of classes and their respective labels, ensuring an efficient training configuration. The proper configuration of this file is essential for enabling the model to effectively learn from the dataset and achieve high accuracy in detection tasks.

## 🛠️ Installation & Usage

### Prerequisites
- **Python 3.8 or higher**: Ensure that you have an appropriate version of Python installed on your system.
- **pip**: The package installer for Python, required for installing dependencies.

### Steps to Set Up the Project

1. **Clone the Repository:**
   To get started, clone this repository to your local machine using the following command:
   ```bash
   git clone https://github.com/arpsn123/YoloTeeth-X-Ray-Instance-Segmentation-Object-Detection-with-YOLOv8.git
   cd YoloTeeth-X-Ray-Instance-Segmentation-Object-Detection-with-YOLOv8
   ```

2. **Install Required Dependencies:**
   Install all necessary dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model:**
   Use the provided training scripts to initiate the training process with the dataset. Modify the training parameters as needed to optimize model performance.

4. **Run Inference:**
   Apply the trained model on new dental X-ray images to evaluate performance and visualize results. The inference scripts provided will guide you through this process.

## 🌟 Acknowledgements

- Special thanks to the open-source community and contributors for providing the tools and resources that made this project possible.
- This work is inspired by ongoing advancements in deep learning and medical imaging research. Collaboration and shared knowledge in the community have been invaluable in shaping this project.


