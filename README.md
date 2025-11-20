<div align="center">
<h1>CIGPose: Causal Intervention Graph Neural Network for Whole-Body Pose Estimation </h1> 
</div>

## 📊 Model Zoo & Results
#### Results on COCO-WholeBody v1.0 val

| Arch                                                   | Input Size | FLOPS (G) | Body AP | Foot AP | Face AP | Hand AP | Whole AP | ckpt                                       |
| ------------------------------------------------------ | ---------- | --------- | ------- | ------- | ------- | ------- | -------- | ------------------------------------------ |
| [CIGPose-m](mmpose/projects/cigpose/wholebody_2d_keypoint/cigpose-m_8xb64-420e_coco-wholebody-256x192.py)       | 256x192    | 2.3       | 69.0    | 64.3    | 82.1    | 49.7    | 59.9     | [pth](https://drive.google.com/file/d/1GvTAyJGUIVI-5KWKRumqx1YH4dEjmwBv/view?usp=sharing) |
| [CIGPose-l](mmpose/projects/cigpose/wholebody_2d_keypoint/cigpose-l_8xb64-420e_coco-wholebody-256x192.py)       | 256x192    | 4.6       | 71.2    | 69.0    | 83.3    | 54.0    | 62.6     | [pth](https://drive.google.com/file/d/1mC5GkU7v_M3tIi8NaDaKDrYvyde-ca2J/view?usp=sharing) |
| [CIGPose-l](mmpose/projects/cigpose/wholebody_2d_keypoint/cigpose-l_8xb32-420e_coco-wholebody-384x288.py)       | 384x288    | 10.7      | 73.0    | 72.0    | 88.3    | 59.8    | 66.3     | [pth](https://drive.google.com/file/d/1q_j2bA3A5UubBDEeyHo2z1MPrcDSOMSt/view?usp=sharing) |
| [CIGPose-x](mmpose/projects/cigpose/wholebody_2d_keypoint/cigpose-x_8xb32-420e_coco-wholebody-384x288.py)       | 384x288    | 18.7      | 73.5    | 72.3    | 88.1    | 60.2    | 67.0     | [pth](https://drive.google.com/file/d/1S08T9Mvpqjt9TGtQ1Tc-5X8HRDff157d/view?usp=sharing) |
| [CIGPose-l+UBody](mmpose/projects/cigpose/wholebody_2d_keypoint/cigpose-l_8xb32-420e_coco-ubody-256x192.py) | 256x192    | 4.6      | 71.3    | 66.2    | 83.4    | 55.5    | 63.1     | [pth](https://drive.google.com/file/d/1KtgCcbIPIC1EGyIKjXvVF6aag1ow_9xY/view?usp=sharing) |
| [CIGPose-l+UBody](mmpose/projects/cigpose/wholebody_2d_keypoint/cigpose-l_8xb32-420e_coco-ubody-384x288.py) | 384x288    | 10.7      | 73.1    | 72.3    | 88.0    | 61.2    | 66.9     | [pth](https://drive.google.com/file/d/1HAhF_tZNOxY4hNWTKooSnMGAguK1U8Pu/view?usp=sharing) |
| [CIGPose-x+UBody](mmpose/projects/cigpose/wholebody_2d_keypoint/cigpose-x_8xb32-420e_coco-ubody-384x288.py) | 384x288    | 18.7      | 73.5    | 70.3    | 88.4    | 62.6    | 67.5     | [pth](https://drive.google.com/file/d/13RmDZpWku876gsrrMso6DfnK7F4Z4vZ2/view?usp=sharing) |

#### Results on COCO val2017

| Arch                                                         | Input Size | FLOPS (G) | Params (M) | AP | AR | ckpt                                                         |
| ------------------------------------------------------------ | ---------- | --------- | ------- | ------- | ------- | ------------------------------------------------------------ |
| [CIGPose-m](mmpose/projects/cigpose/body_2d_keypoint/cigpose-m_1xb384-420e_coco-256x192.py) | 256x192    | 1.9     | 14   | 76.6 | 79.3 | [pth](https://drive.google.com/file/d/1tNXidCKVhXqxT8WIYPEdSfIEownbIl5E/view?usp=sharing) |
| [CIGPose-l](mmpose/projects/cigpose/body_2d_keypoint/cigpose-l_1xb256-420e_coco-256x192.py) | 256x192    | 4.2      | 28   | 77.6 | 80.3    | [pth](https://drive.google.com/file/d/1jsKXa4waJKkLFmF7IhWVA_xMBs5_61zr/view?usp=sharing) |
| [CIGPose-l](mmpose/projects/cigpose/body_2d_keypoint/cigpose-l_1xb64-420e_coco-384x288.py) | 384x288    | 9.4    | 29   | 78.5  | 81.1  | [pth](https://drive.google.com/file/d/14GUos6gnw3Rm78SPPIb48DnK5gzhzFAb/view?usp=sharing) |

#### Results on CrowdPose test set

| Arch                                                         | Input Size | Params (M) | AP   | AP easy | AP medium | AP hard | ckpt                                                         |
| ------------------------------------------------------------ | ---------- | ---------- | ---- | ------- | --------- | ------- | ------------------------------------------------------------ |
| [CIGPose-m](mmpose/projects/cigpose/body_2d_keypoint/cigpose-m_1xb64-210e_crowdpose-256x192.py) | 256x192    | 14.4       | 71.4 | 81.0    | 72.7      | 58.9    | [pth](https://drive.google.com/file/d/1nH_ONU-4CbaUs6xh20Jqhj_gHf_HFtmi/view?usp=sharing) |
| [CIGPose-l](mmpose/projects/cigpose/body_2d_keypoint/cigpose-l_1xb64-210e_crowdpose-256x192.py) | 256x192    | 28.4       | 73.7 | 82.8    | 75.1      | 61.2    | [pth](https://drive.google.com/file/d/1_q2GlrZxOvtK2aQLx79G-B7h3s3kPLgM/view?usp=sharing) |
| [CIGPose-l](mmpose/projects/cigpose/body_2d_keypoint/cigpose-l_1xb64-210e_crowdpose-384x288.py) | 384x288    | 28.8       | 74.2 | 82.9    | 75.6      | 62.5    | [pth](https://drive.google.com/file/d/10DYfrnQstJZ3WjWY3I9d9R7cxlptlrNL/view?usp=sharing) |
| [CIGPose-x](mmpose/projects/cigpose/body_2d_keypoint/cigpose-x_1xb64-210e_crowdpose-384x288.py) | 384x288    | 50.4       | 75.8 | 84.2    | 77.3      | 63.6    | [pth](https://drive.google.com/file/d/1VN_OhFUeKtwvm0Pdyj5mt5mLwlVCIXaO/view?usp=sharing) |
------
## 🛠️ Installation

Our code is based on [MMPose](https://github.com/open-mmlab/mmpose).

```
# 1. Create a conda environment
conda create -n cigpose python=3.8 -y
conda activate cigpose

# 2. Install PyTorch (adjust CUDA version as needed)
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html

# 3. Install MMCV and MMDetection
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"

# 4. Clone this repository
git clone https://github.com/53mins/CIGPose.git
cd CIGPose/mmpose
pip install -v -e .
```
------
## 🏃 Usage

#### Data Preparation
Please refer to MMPose guidelines for dataset preparation:

- [COCO-WholeBody](https://github.com/open-mmlab/mmpose/blob/main/docs/en/dataset_zoo/2d_wholebody_keypoint.md) 

- [COCO](https://github.com/open-mmlab/mmpose/blob/main/docs/en/dataset_zoo/2d_body_keypoint.md) 

- [CrowdPose](https://github.com/open-mmlab/mmpose/blob/main/docs/en/dataset_zoo/2d_body_keypoint.md%23crowdpose) 

## ⭐Train a model

```
cd mmpose
bash tools/dist_train.sh mmpose/projects/cigpose/wholebody_2d_keypoint/cigpose-l_8xb32-420e_coco-wholebody-384x288.py 8
```

## ⭐Test a model

```
bash tools/dist_test.sh mmpose/projects/cigpose/wholebody_2d_keypoint/cigpose-l_8xb32-420e_coco-wholebody-384x288.py path/to/checkpoint.pth
```
