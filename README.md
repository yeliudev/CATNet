# Context Aggregation Network

[![arXiv](https://badgen.net/badge/arXiv/2111.11057/red?cache=300)](https://arxiv.org/abs/2111.11057)
[![License](https://badgen.net/github/license/yeliudev/CATNet?label=License&color=cyan&cache=300)](https://github.com/yeliudev/CATNet/blob/main/LICENSE)

This repository maintains the official implementation of the paper **Learning to Aggregate Multi-Scale Context for Instance Segmentation in Remote Sensing Images** by [Ye Liu](https://yeliu.dev/), [Huifang Li](https://faculty.whu.edu.cn/show.jsp?n=Huifang%20Li), [Chao Hu](https://orcid.org/0000-0001-6183-9051), [Shuang Luo](https://www.researchgate.net/profile/shuang-luo-6), [Yan Luo](https://www.researchgate.net/profile/Yan_Luo65), and [Chang Wen Chen](https://web.comp.polyu.edu.hk/chencw/), which has been accepted by [TNNLS 2023](https://cis.ieee.org/publications/t-neural-networks-and-learning-systems).

<p align="center"><img width="850" src="https://raw.githubusercontent.com/yeliudev/CATNet/main/.github/model.svg"></p>

## Installation

Please refer to the following environmental settings that we use. You may install these packages by yourself if you meet any problem during automatic installation.

- CUDA 11.8
- CUDNN 8.7.0.84
- Python 3.11.3
- PyTorch 2.0.1
- [MMEngine](https://github.com/open-mmlab/mmengine) 0.7.4
- [MMCV](https://github.com/open-mmlab/mmcv) 2.0.0
- [MMDetection](https://github.com/open-mmlab/mmdetection) 3.0.0
- [NNCore](https://github.com/yeliudev/nncore) 0.3.6

### Install from source

1. Clone the repository from GitHub.

```
git clone https://github.com/yeliudev/CATNet.git
cd CATNet
```

2. Install dependencies.

```
pip install -r requirements.txt
```

3. Set environment variable

```
export PYTHONPATH=$PWD:$PYTHONPATH
```

## Getting Started

### Download and prepare the datasets

1. Download the pre-processed datasets.

- [iSAID](https://huggingface.co/yeliudev/CATNet/resolve/main/datasets/isaid-9d62d4ad.zip)
- [DIOR](https://huggingface.co/yeliudev/CATNet/resolve/main/datasets/dior-b162132d.zip)
- [NWPU VHR-10](https://huggingface.co/yeliudev/CATNet/resolve/main/datasets/vhr-79ccc9f3.zip)
- [HRSID](https://huggingface.co/yeliudev/CATNet/resolve/main/datasets/hrsid-4e02052e.zip)

2. Prepare the files in the following structure.

```
CATNet
├── configs
├── datasets
├── models
├── tools
├── data
│   ├── dior
│   │   ├── Annotations
│   │   ├── ImageSets
│   │   └── JPEGImages
│   ├── hrsid
│   │   ├── annotations
│   │   └── images
│   ├── isaid
│   │   ├── train
│   │   ├── val
│   │   └── test
│   └── vhr
│       ├── annotations
│       └── images
├── README.md
├── setup.cfg
└── ···
```

### Train a model

Run the following command to train a model using a specified config.

```
mim train mmdet <path-to-config> --gpus 4 --launcher pytorch
```

> If an `out-of-memory` error occurs on iSAID dataset, please uncomment [L22-L24](https://github.com/yeliudev/CATNet/blob/main/datasets/isaid.py#L22:L24) in the dataset code and try again. This will filter out a few images with more than 1,000 objects, largely reducing the memory cost.

### Test a model and evaluate results

Run the following command to test a model and evaluate results.

```
mim test mmdet <path-to-config> --checkpoint <path-to-checkpoint> --gpus 4 --launcher pytorch
```

## Model Zoo

We provide multiple pre-trained models here. All the models are trained using 4 NVIDIA A100 GPUs and are evaluated using the default metrics of the datasets.

<table>
  <tr>
    <th rowspan="2">Dataset</th>
    <th rowspan="2">Model</th>
    <th rowspan="2">Backbone</th>
    <th rowspan="2">Schd</th>
    <th rowspan="2">Aug</th>
    <th colspan="2">Performance</th>
    <th rowspan="2">Download</th>
  </tr>
  <tr>
    <th>BBox AP</th>
    <th>Mask AP</th>
  </tr>
  <tr>
    <td align="center" rowspan="2">
      <a href="https://arxiv.org/abs/1905.12886">iSAID</a>
    </td>
    <td align="center">
      <a href="https://github.com/yeliudev/CATNet/blob/main/configs/isaid/cat_mask_rcnn_r50_3x_isaid.py">CAT Mask R-CNN</a>
    </td>
    <td align="center">ResNet-50</td>
    <td align="center">3x</td>
    <td align="center">&cross;</td>
    <td align="center">45.1</td>
    <td align="center">37.2</td>
    <td align="center">
      <a href="https://huggingface.co/yeliudev/CATNet/resolve/main/checkpoints/cat_mask_rcnn_r50_3x_isaid-384df911.pth">model</a> |
      <a href="https://huggingface.co/yeliudev/CATNet/resolve/main/checkpoints/cat_mask_rcnn_r50_3x_isaid.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/yeliudev/CATNet/blob/main/configs/isaid/cat_mask_rcnn_r50_aug_3x_isaid.py">CAT Mask R-CNN</a>
    </td>
    <td align="center">ResNet-50</td>
    <td align="center">3x</td>
    <td align="center">&check;</td>
    <td align="center">47.7</td>
    <td align="center">39.2</td>
    <td align="center">
      <a href="https://huggingface.co/yeliudev/CATNet/resolve/main/checkpoints/cat_mask_rcnn_r50_aug_3x_isaid-1e5351dd.pth">model</a> |
      <a href="https://huggingface.co/yeliudev/CATNet/resolve/main/checkpoints/cat_mask_rcnn_r50_aug_3x_isaid.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center" rowspan="4">
      <a href="https://arxiv.org/abs/1909.00133">DIOR</a>
    </td>
    <td align="center">
      <a href="https://github.com/yeliudev/CATNet/blob/main/configs/dior/catnet_r50_3x_dior.py">CATNet</a>
    </td>
    <td align="center">ResNet-50</td>
    <td align="center">3x</td>
    <td align="center">&cross;</td>
    <td align="center">74.0</td>
    <td align="center">—</td>
    <td align="center">
      <a href="https://huggingface.co/yeliudev/CATNet/resolve/main/checkpoints/catnet_r50_3x_dior-5cb86542.pth">model</a> |
      <a href="https://huggingface.co/yeliudev/CATNet/resolve/main/checkpoints/catnet_r50_3x_dior.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/yeliudev/CATNet/blob/main/configs/dior/catnet_r50_aug_3x_dior.py">CATNet</a>
    </td>
    <td align="center">ResNet-50</td>
    <td align="center">3x</td>
    <td align="center">&check;</td>
    <td align="center">78.2</td>
    <td align="center">—</td>
    <td align="center">
      <a href="https://huggingface.co/yeliudev/CATNet/resolve/main/checkpoints/catnet_r50_aug_3x_dior-6ec5fae1.pth">model</a> |
      <a href="https://huggingface.co/yeliudev/CATNet/resolve/main/checkpoints/catnet_r50_aug_3x_dior.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/yeliudev/CATNet/blob/main/configs/dior/cat_rcnn_r50_3x_dior.py">CAT R-CNN</a>
    </td>
    <td align="center">ResNet-50</td>
    <td align="center">3x</td>
    <td align="center">&cross;</td>
    <td align="center">75.8</td>
    <td align="center">—</td>
    <td align="center">
      <a href="https://huggingface.co/yeliudev/CATNet/resolve/main/checkpoints/cat_rcnn_r50_3x_dior-044be4c7.pth">model</a> |
      <a href="https://huggingface.co/yeliudev/CATNet/resolve/main/checkpoints/cat_rcnn_r50_3x_dior.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/yeliudev/CATNet/blob/main/configs/dior/cat_rcnn_r50_aug_3x_dior.py">CAT R-CNN</a>
    </td>
    <td align="center">ResNet-50</td>
    <td align="center">3x</td>
    <td align="center">&check;</td>
    <td align="center">80.6</td>
    <td align="center">—</td>
    <td align="center">
      <a href="https://huggingface.co/yeliudev/CATNet/resolve/main/checkpoints/cat_rcnn_r50_aug_3x_dior-89845304.pth">model</a> |
      <a href="https://huggingface.co/yeliudev/CATNet/resolve/main/checkpoints/cat_rcnn_r50_aug_3x_dior.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center" rowspan="2">
      <a href="https://doi.org/10.1016/j.isprsjprs.2014.10.002">NWPU<br>VHR-10</a>
    </td>
    <td align="center">
      <a href="https://github.com/yeliudev/CATNet/blob/main/configs/vhr/cat_mask_rcnn_r50_6x_vhr.py">CAT Mask R-CNN</a>
    </td>
    <td align="center">ResNet-50</td>
    <td align="center">6x</td>
    <td align="center">&cross;</td>
    <td align="center">71.0</td>
    <td align="center">69.3</td>
    <td align="center">
      <a href="https://huggingface.co/yeliudev/CATNet/resolve/main/checkpoints/cat_mask_rcnn_r50_6x_vhr-d38af93b.pth">model</a> |
      <a href="https://huggingface.co/yeliudev/CATNet/resolve/main/checkpoints/cat_mask_rcnn_r50_6x_vhr.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/yeliudev/CATNet/blob/main/configs/vhr/cat_mask_rcnn_r50_aug_6x_vhr.py">CAT Mask R-CNN</a>
    </td>
    <td align="center">ResNet-50</td>
    <td align="center">6x</td>
    <td align="center">&check;</td>
    <td align="center">72.4</td>
    <td align="center">70.7</td>
    <td align="center">
      <a href="https://huggingface.co/yeliudev/CATNet/resolve/main/checkpoints/cat_mask_rcnn_r50_aug_6x_vhr-599b2304.pth">model</a> |
      <a href="https://huggingface.co/yeliudev/CATNet/resolve/main/checkpoints/cat_mask_rcnn_r50_aug_6x_vhr.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center" rowspan="2">
      <a href="https://doi.org/10.1109/access.2020.3005861">HRSID</a>
    </td>
    <td align="center">
      <a href="https://github.com/yeliudev/CATNet/blob/main/configs/hrsid/cat_mask_rcnn_r50_6x_hrsid.py">CAT Mask R-CNN</a>
    </td>
    <td align="center">ResNet-50</td>
    <td align="center">6x</td>
    <td align="center">&cross;</td>
    <td align="center">70.9</td>
    <td align="center">57.6</td>
    <td align="center">
      <a href="https://huggingface.co/yeliudev/CATNet/resolve/main/checkpoints/cat_mask_rcnn_r50_6x_hrsid-198d5409.pth">model</a> |
      <a href="https://huggingface.co/yeliudev/CATNet/resolve/main/checkpoints/cat_mask_rcnn_r50_6x_hrsid.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/yeliudev/CATNet/blob/main/configs/hrsid/cat_mask_rcnn_r50_aug_6x_hrsid.py">CAT Mask R-CNN</a>
    </td>
    <td align="center">ResNet-50</td>
    <td align="center">6x</td>
    <td align="center">&check;</td>
    <td align="center">72.0</td>
    <td align="center">59.6</td>
    <td align="center">
      <a href="https://huggingface.co/yeliudev/CATNet/resolve/main/checkpoints/cat_mask_rcnn_r50_aug_6x_hrsid-0da9459c.pth">model</a> |
      <a href="https://huggingface.co/yeliudev/CATNet/resolve/main/checkpoints/cat_mask_rcnn_r50_aug_6x_hrsid.json">metrics</a>
    </td>
  </tr>
</table>

## Citation

If you find this project useful for your research, please kindly cite our paper.

```bibtex
@article{liu2023learning,
  title={Learning to Aggregate Multi-Scale Context for Instance Segmentation in Remote Sensing Images},
  author={Liu, Ye and Li, Huifang and Hu, Chao and Luo, Shuang and Luo, Yan and Chen, Chang Wen},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023}
}
```
