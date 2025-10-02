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
  TO DO

  ## 4. Result
  TO DO

  ## 5. References
  TO DO