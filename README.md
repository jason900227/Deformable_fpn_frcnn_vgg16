# Deformable FPN Faster R-CNN with VGG16 for Multi-class Chest X-ray Lesion Detection
![Python](https://img.shields.io/badge/Python-3.8.20-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red)
![CUDA](https://img.shields.io/badge/CUDA-11.8-orange)

This repository applies a Deformable Feature Pyramid Network (FPN) Faster R-CNN with a VGG16 backbone for multi-class chest X-ray lesion detection.

It is configured to detect and classify 10 types of thoracic abnormalities: Atelectasis, Calcification, Consolidation, Effusion, Emphysema, Fibrosis, Fracture, Mass, Nodule, and Pneumothorax.

The code has been adapted from existing repositories to facilitate training and evaluation on this dataset,
for details and original implementations, please refer to the repositories listed in the References section below.

## 1. Environment
* **OS**: Windows 10/11  
* **Python**: 3.8.20  
* **PyTorch**: 2.0.1
* **CUDA**: 11.8

## 2. Installation
### 2.1 Clone this project
  ```
  cd ~
  git clone https://github.com/jason900227/Deformable_fpn_frcnn_vgg16.git
  cd Deformable_fpn_frcnn_vgg16
  ```
### 2.2 Create New Conda Environment
  ```
  # Create environment with Python 3.8.20
  conda create --name Deformable_fpn_frcnn_vgg16 python=3.8.20 -y
  
  # Activate environment
  conda activate Deformable_fpn_frcnn_vgg16
  
  # Install PyTorch 2.0.1 with CUDA 11.8
  conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
  
  # Install other dependencies
  pip install -r requirements.txt
  ```

  ## 3. Dataset
  ### 3.1 Download
  Please download the **ChestX-Det10** dataset from [Deepwise-AILab/ChestX-Det10-Dataset](https://github.com/Deepwise-AILab/ChestX-Det10-Dataset).

  You will need the following four files:
  * [`test.json`](https://github.com/Deepwise-AILab/ChestX-Det10-Dataset)  
  * [`train.json`](https://github.com/Deepwise-AILab/ChestX-Det10-Dataset)    
  * [`test_data.zip`](http://resource.deepwise.com/xraychallenge/test_data.zip)
  * [`train_data.zip`](http://resource.deepwise.com/xraychallenge/train_data.zip)

  After downloading, extract the `.zip` files and organize the directory structure as follows:
  ```
  ./raw_data/
    │
    ├── test.json
    ├── train.json
    ├── test_data/ # extracted from test_data.zip
    │   ├── 36199.png
    │   ├── 36212.png
    │   └── ...
    └── train_data/ # extracted from train_data.zip
        ├── 36200.png
        ├── 36201.png
        └── ...
  ```

  Ensure that all four files are placed under the `./raw_data` directory as shown above.
  ### 3.2 Preprocessing
  After downloading and organizing the dataset as described in **Section 3.1**, please run the preprocessing script to convert the data into the VOC format.

  Run the following command in the project root directory:
  ```
  python .\raw_data\preprocessing.py
  ```

  After execution, the following directory structure will be generated:
  ```
  ./data/
    │
    ├── test/
    │   └── VOCdevkit/
    │       └── VOC2007/
    │           ├── Annotations/
    │           │   ├── 36212.xml
    │           │   ├── 36266.xml
    │           │   └── ...
    │           ├── ImageSets/
    │           │   └── Main/
    │           │       └── test.txt
    │           └── JPEGImages/
    │               ├── 36212.jpg
    │               ├── 36266.jpg
    |               └── ...
    │
    └── train/
        └── VOCdevkit/
            └── VOC2007/
                ├── Annotations/
                │   ├── 36204.xml
                │   ├── 36205.xml
                |   └── ...
                ├── ImageSets/
                │   └── Main/
                │       └── train.txt
                └── JPEGImages/
                    ├── 36204.jpg
                    ├── 36205.jpg
                    └── ...
  ```

  ## 4. Usage Examples
  You can easily modify the training parameters in `./utils/config.py` and execute the following scripts to train and test the model.
  ### 4.1 Train
  Run the following command to perform training:
  ```
  python .\train.py
  ```
  ### 4.2 Test
  Run the following command to perform testing:
  ```
  python .\test.py
  ```
  > Note: You can visualize the testing results by setting `visualize=True` in the configuration file, and the output images will be saved under the `./save_dir/visuals` directory specified in the configuration.

  ## 5. Demo
  TO DO

  ## 6. References

  * [Deepwise-AILab/ChestX-Det10-Dataset](https://github.com/Deepwise-AILab/ChestX-Det10-Dataset)

  * Jingyu Liu, Jie Lian, and Yizhou Yu, "ChestX-Det10: Chest X-ray Dataset on Detection of Thoracic Abnormalities.", arXiv preprint arXiv:2006.10550v3, 2020, [https://arxiv.org/abs/2006.10550v3](https://arxiv.org/abs/2006.10550v3)

  * [Ziruiwang409/improved-faster-rcnn](https://github.com/Ziruiwang409/improved-faster-rcnn/tree/main)

  * [chenyuntc/simple-faster-rcnn-pytorch](https://github.com/chenyuntc/simple-faster-rcnn-pytorch)

  * [jwyang/fpn.pytorch](https://github.com/jwyang/fpn.pytorch)

  * [txytju/Faster-RCNN-FPN](https://github.com/txytju/Faster-RCNN-FPN)

  * [msracver/Deformable-ConvNets](https://github.com/msracver/Deformable-ConvNets)

  * [developer0hye/PyTorch-Deformable-Convolution-v2](https://github.com/developer0hye/PyTorch-Deformable-Convolution-v2)