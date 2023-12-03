# KaggleDataset_BIRDS_525_SPECIES CNN Classifier

This is a non-profit side project, created purely personal interest.  
The main task of this project is to classify bird species.

Here is some example in this dataset：
| IVORY GULL (象牙鷗) | LIMPKIN (秧鶴) | MYNA (八哥) |
| ------------- | ------------- | ------------- |
| ![image](https://github.com/SHENG-KAI-HUANG/KaggleDataset_BIRDS_525_SPECIES/blob/main/sampleImage/IVORY%20GULL_002.jpg)  | ![image](https://github.com/SHENG-KAI-HUANG/KaggleDataset_BIRDS_525_SPECIES/blob/main/sampleImage/LIMPKIN_003.jpg)   | ![image](https://github.com/SHENG-KAI-HUANG/KaggleDataset_BIRDS_525_SPECIES/blob/main/sampleImage/MYNA_010.jpg)  |

Birds are really cute right? If you're interested, please explore "BIRDS 525 SPECIES- IMAGE CLASSIFICATION" datasets to find more images.  
The dataset URL：https://www.kaggle.com/datasets/gpiosenka/100-bird-species/discussion/456917  
  
Using EfficientNet-B4 [1], test accuracy about 98.4%  
__Usage of trained model ：  
python ./code/tradition/classifier.py -n efficientnet-b4 -m /result/EfficientNet-B4/trainComplete.parm -i [single image path]__  
  
Special thanks to Mr. Gerald Piosenka, who providing a high-quality bird dataset.
> [1] Tan, M. and Le, Q.V. (2019) EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. Proceedings of the 36th International Conference on Machine Learning, ICML 2019, Long Beach, 9-15 June 2019, 6105-6114.http://proceedings.mlr.press/v97/tan19a.html
