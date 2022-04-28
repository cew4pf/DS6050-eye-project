# **DS6050: Deep Learning Final Project**

## **Employing Neural Networks for Early Detection of Ocular Diseases**

#### Ana Daley (amd2yc), Evan Mitchell (etm8fs), Cecily Wolfe (cew4pf)

#### Spring 2022

### **Categories**
Multi-Class Classification, Medicine

### **Motivation**
As of 2021, the World Health Organization (WHO) estimates over two billion people suffer from visual impairment, and nearly half of these cases were preventable, or have not yet been treated (1). Neural networks can mitigate challenges inherent in diagnosis, using advanced classification algorithms to enable early detection and correction.

### **Dataset**
The Ocular Disease Intelligent Recognition (ODIR) dataset on Kaggle provided by Peking University and Shanggong Medical Technology Co. Ltd. includes 5000 color fundus images of retinal tissue taken with various camera brands, including Canon, Zeis, and Kowa. Pictures of the right and left eyes of patients are labeled with at one of eight diagnoses – normal (1140 cases), diabetes (1128), glaucoma (215), cataract (212), age-related macular degeneration (164), hypertension (103), myopia (174), and other diseases/abnormalities (979) – as well as patients’ age and sex (2,3).


### **Explanation of Files**
* DS_6050_Project_Proposal.pdf: initial project proposal submitted on March 23, 2022
* DS_6050_Project_Sources.pdf: online sources used for help with cleaning data and creating neural networks
* data_preprocessing.ipynb: Jupyter notebook to access the dataset from Kaggle and process it (assign correct class labels to the images, balance the classes, etc.) and visualize the images with class labels
  * Requires `TensorFlow`, Kaggle API access
* dl_proj_models.slurm: slurm file to submit python files using `TensorFlow` to the Slurm Workload Manager (i.e., via Rivanna)
 * Takes a user argument for a python file to submit (e.g., `sbatch dl_proj_models.slurm <filename.py>`)
* ensemble_and_custom_models.ipynb: evaluating individual and weighted ensemble models created using transfer learning (`MobileNetV2`, `InceptionNetV3`, `VGG16`, `ResNet50V2`, `EfficientNetV2`) and the custom CNN model

Models: folder containing python files for training models created using transfer learning and a custom Convolutional Neural Network (CNN)
 * dl_project_custom.py: custom CNN with multiple alternating convolution and max pooling layers, followed by fully connected, batch normalization, and dropout layers
 * [dl_project_efficient.py](https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet_v2/EfficientNetV2L): `EfficientNetV2` transfer learning model
 * dl_project_ensemble.py: weighted ensemble model using the five transfer learning models with different weights
 * [dl_project_inceptionnet.py](https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_v3/InceptionV3): `InceptionNetV3` transfer learning model
 * [dl_project_mobilenet.py](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet_v2/MobileNetV2): `MobileNetV2` transfer learning model
 * [dl_project_res.py](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet_v2/ResNet50V2): `ResNet50V2` transfer learning model
 * [dl_project_vgg.py](https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/VGG16): `VGG16` transfer learning model

Training_Output: folder containing the output from slurm jobs running
* dlproj_custom-37680066.txt: output from `sbatch dl_proj_models.slurm dl_project_custom.py`
* dlproj_eff-36851516.txt: output from `sbatch dl_proj_models.slurm dl_project_efficient.py`
* dlproj_ensemble-37443516.txt: output from `sbatch dl_proj_models.slurm dl_project_ensemble.py`
* dlproj_inc-36671798.txt: output from `sbatch dl_proj_models.slurm dl_project_inceptionnet.py`
* dlprog_mn-36630848.txt: : output from `sbatch dl_proj_models.slurm dl_project_mobilenet.py`
* dlproj_res-36813958.txt: output from `sbatch dl_proj_models.slurm dl_project_res.py`
* dlproj_vgg-36710683.txt: output from `sbatch dl_proj_models.slurm dl_project_vgg.py`


### **Sources**
1. “Vision Impairment and Blindness.” World Health Organization. World Health Organization, 2021. https://www.who.int/news-room/fact-sheets/detail/blindness-and-visual-impairment.

2. Larxel. “Ocular Disease Recognition.” Kaggle, September 24, 2020. https://www.kaggle.com/andrewmvd/ocular-disease-recognition-odir5k?select=preprocessed_images.

3. Li, Cheng, Jin Ye, Junjun He, Shanshan Wang, Yu Qiao, and Lixu Gu. “Dense Correlation Network for Automated Multi-Label Ocular Disease Detection with Paired Color Fundus Photographs.” 2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI), 2020. https://doi.org/10.1109/isbi45749.2020.9098340.
