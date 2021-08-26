# DESDnet
PyTorch implementation of "Reverse erasure guided spatio-temporal autoencoder with compact feature representation for video anomaly detection"  by Yuanhong Zhong, Xia Chen, Jinyang Jiang, Fan Ren.


## Dependencies
* Python 3.6
* PyTorch 1.1.0
* Numpy
* Sklearn

## Datasets
* USCD Ped2 [[dataset](https://github.com/StevenLiuWen/ano_pred_cvpr2018)]
* CUHK Avenue [[dataset](https://github.com/StevenLiuWen/ano_pred_cvpr2018)]
* ShanghaiTech [[dataset](https://github.com/StevenLiuWen/ano_pred_cvpr2018)]

These datasets are from an official github of "Future Frame Prediction for Anomaly Detection - A New Baseline (CVPR 2018)".
Download the datasets into your_dataset_directory.

## Training
python Train.py # for training
You can freely define parameters with your own settings like
python Train.py --gpus 1 --dataset_path 'your_dataset_directory' --dataset_type Avenue --exp_dir 'your_log_directory' --t_length 5 # for training

## Evaluation
Test your own model
Check your dataset_type (ped2, Avenue or shanghai)
python Evaluate.py --dataset_type ped2 --model_dir your_model.pth

We also provide the pre-trained models and the labels of UCSD Ped2, Avenue and ShanghaiTech datasets at https://pan.baidu.com/s/1tbWJeJwIWVfTIJ0kcsj1aw (password：mnxg). To test these models, you need download and put them in exp\dataset_type\checkpoint folder.


