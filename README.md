<h1 style="text-align:center;">Causal Intervention Graph Neural Network for Robust Whole-body Pose Estimation</h1>

### Results on COCO-WholeBody v1.0 val

| Arch                                                   | Input Size | FLOPS (G) | Body AP | Foot AP | Face AP | Hand AP | Whole AP | ckpt                                       |
| ------------------------------------------------------ | ---------- | --------- | ------- | ------- | ------- | ------- | -------- | ------------------------------------------ |
| [CIGPose-m](mmpose/projects/cigpose/wholebody_2d_keypoint/cigpose-m_8xb64-420e_coco-wholebody-256x192.py)       | 256x192    | 2.3       | 69.0    | 64.3    | 82.1    | 49.7    | 59.9     | [pth](https://drive.google.com/file/d/1GvTAyJGUIVI-5KWKRumqx1YH4dEjmwBv/view?usp=sharing) |
| [CIGPose-l](mmpose/projects/cigpose/wholebody_2d_keypoint/cigpose-l_8xb64-420e_coco-wholebody-256x192.py)       | 256x192    | 4.6       | 71.2    | 69.0    | 83.3    | 54.0    | 62.6     | [pth](https://drive.google.com/file/d/1mC5GkU7v_M3tIi8NaDaKDrYvyde-ca2J/view?usp=sharing) |
| [CIGPose-l](mmpose/projects/cigpose/wholebody_2d_keypoint/cigpose-l_8xb32-420e_coco-wholebody-384x288.py)       | 384x288    | 10.7      | 72.9    | 71.4    | 88.2    | 59.8    | 66.3     | [pth](https://www.google.com/search?q=%23) |
| [CIGPose-x](mmpose/projects/cigpose/wholebody_2d_keypoint/cigpose-x_8xb32-420e_coco-wholebody-384x288.py)       | 384x288    | 18.7      | 73.5    | 72.3    | 88.1    | 60.2    | 67.0     | [pth](https://drive.google.com/file/d/1S08T9Mvpqjt9TGtQ1Tc-5X8HRDff157d/view?usp=sharing) |
| [CIGPose-l+UBody](mmpose/projects/cigpose/wholebody_2d_keypoint/cigpose-l_8xb32-420e_coco-ubody-384x288.py) | 384x288    | 10.7      | 72.7    | 72.2    | 88.1    | 61.8    | 66.6     | [pth](https://drive.google.com/file/d/11vRoU6tO3otliS_clDTqaKGUMVn9PT8s/view?usp=sharing) |
| [CIGPose-x+UBody](mmpose/projects/cigpose/wholebody_2d_keypoint/cigpose-x_8xb32-420e_coco-ubody-384x288.py) | 384x288    | 18.7      | 73.5    | 70.3    | 88.4    | 62.6    | 67.5     | [pth](https://drive.google.com/file/d/13RmDZpWku876gsrrMso6DfnK7F4Z4vZ2/view?usp=sharing) |
