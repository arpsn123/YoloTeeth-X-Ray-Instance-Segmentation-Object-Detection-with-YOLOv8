## YoloTeeth-X-Ray-Instance-Segmentation-Object-Detection-with-YOLOv8
YoloTeeth merges cutting-edge technology with the intricacies of dental image analysis. This repository pioneers the use of YOLOv8 for teeth X-ray instance segmentation and object detection. Crafted with precision and efficiency, the project addresses the challenges of dental radiography. By accurately identifying and delineating individual teeth, YoloTeeth sets a new standard in dental healthcare, empowering professionals with enhanced diagnostic capabilities and facilitating impactful research endeavors

### Dataset
The _same dataset_, which was used into my [Dental-X-RAY-Image-Detection-and-Instance-Segmentation](https://github.com/arpsn123/Dental-X-RAY-Image-Detection-and-Instance-Segmentation.git) also used here.

ALL 269 images and their combined single _.json annotation file_ used for training and 5 validation images their combined single _.json annotation file_ used for validation are converted into individual _**YOLOv8 PyTorch TXT**_, means 269 _.txt file_ for training images and 5 _.txt file_ for validation images.

### Model
_**YOLOv8**_ is a real-time object detection and segmentation model with a single-pass architecture, predicting bounding boxes and class probabilities for objects. It balances performance and speed, utilizing advanced features like a powerful backbone network and possibly a Feature Pyramid Network for multi-scale feature extraction.

![train_batch2011](https://github.com/arpsn123/YoloTeeth-X-Ray-Instance-Segmentation-Object-Detection-with-YOLOv8/assets/112195431/279db196-66e0-4da9-a69d-84c7a4498873)

### Performance Evaluation
In YOLOv8, loss and accuracy curves provide essential insights into the model's performance during training. The loss curve measures how effectively the model minimizes the difference between predicted bounding boxes and ground truth annotations over training epochs, typically using metrics like mean squared error or binary cross-entropy loss. Meanwhile, accuracy curves, although less common in object detection tasks, may track metrics like Intersection over Union (IoU) or mean Average Precision (mAP) to gauge detection accuracy. Continuous monitoring of these curves guides model adjustments and training strategies to enhance object detection performance while mitigating issues such as overfitting.

![results](https://github.com/arpsn123/YoloTeeth-X-Ray-Instance-Segmentation-Object-Detection-with-YOLOv8/assets/112195431/24ecb4c0-724a-4ee8-bb94-95896471b61d)

### Integration of data.yaml in YOLOv8 Training
In YOLOv8, the data.yaml file is typically used to store metadata and configuration information related to the dataset being used for training the model. This data.yaml file helps YOLOv8 locate the training and validation images, as well as define the number of classes and their corresponding names. It's an essential component for configuring the dataset used in training the YOLOv8 model.
