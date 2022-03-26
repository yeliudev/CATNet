# Context Aggregation Network

[![arXiv](https://badgen.net/badge/arXiv/2111.11057/red?cache=300)](https://arxiv.org/abs/2111.11057)
[![License](https://badgen.net/github/license/yeliudev/CATNet?label=License&color=cyan&cache=300)](https://github.com/yeliudev/CATNet/blob/main/LICENSE)

This repository maintains the official implementation of the paper **Learning to Aggregate Multi-Scale Context for Instance Segmentation in Remote Sensing Images** by [Ye Liu](https://yeliu.me/), [Huifang Li](http://faculty.whu.edu.cn/show.jsp?n=Huifang%20Li), [Chao Hu](https://orcid.org/0000-0001-6183-9051), [Shuang Luo](https://www.researchgate.net/profile/shuang-luo-6), [Yan Luo](https://orcid.org/0000-0002-9533-6070), and [Chang Wen Chen](https://www4.comp.polyu.edu.hk/~chencw/).

<p align="center"><img width="850" src="https://raw.githubusercontent.com/yeliudev/CATNet/main/.github/model.svg"></p>

## Installation

Please refer to the following environmental settings that we use. You may install these packages by yourself if you meet any problem during automatic installation.

- CUDA 10.2 Update 2
- CUDNN 8.0.5.39
- Python 3.9.7
- PyTorch 1.10.0
- [MMCV](https://github.com/open-mmlab/MMCV) 1.3.17
- [MMDetection](https://github.com/open-mmlab/mmdetection) 2.18.1
- [NNCore](https://github.com/yeliudev/nncore) 0.3.2

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

## Getting Started

### Download and prepare the datasets

1. Download and extract the datasets.

> Note that the images in iSAID dataset are splitted into patches with both sides no more than 512 pixels, as reported in our paper. We strongly recommend using this pre-processed version directly since the offical toolkit has known unknown bugs, leading to undesirable patch sizes (e.g. extreme aspect ratios).

- [iSAID](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21039533r_connect_polyu_hk/EXOi4UhzDI1KucvBQccQ2FgBOG__n3UpUvpAJDDhIhZ_rg?e=BxzYgy)
- [DIOR](https://drive.google.com/drive/folders/1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC)
- [NWPU VHR-10](https://1drv.ms/u/s!AmgKYzARBl5cczaUNysmiFRH4eE)
- [HRSID](https://drive.google.com/file/d/1xgXi8KC3MDWuu7Yp4n2J-LOPYLRszHAc)

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
│   │   ├── JPEGImages-test
│   │   └── JPEGImages-trainval
│   ├── hrsid
│   │   ├── annotations
│   │   └── images
│   ├── isaid
│   │   ├── annotations
│   │   ├── train
│   │   └── val
│   └── vhr
│       ├── ground truth
│       └── positive image set
├── README.md
├── setup.cfg
└── ···
```

3. Convert DIOR annotations to PASCAL VOC format.

```
python tools/convert_dior.py
```

4. Convert NWPU VHR-10 annotations to COCO format.

```
python tools/convert_vhr.py
```

### Train a model

Run the following command to train a model using a specified config.

```
torchrun --nproc_per_node=4 tools/train.py <path-to-config>
```

### Test a model and evaluate results

Run the following command to test a model and evaluate results.

```
torchrun --nproc_per_node=4 tools/test.py <path-to-config> <path-to-checkpoint>
```

## Model Zoo

We provide multiple pre-trained models here. All the models are trained using 4 NVIDIA Tesla V100-SXM2 GPUs and are evaluated using the default metrics of the datasets.

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
      <a href="https://github.com/yeliudev/CATNet/blob/main/configs/isaid/cat_mask_rcnn_r50_1x_isaid.py">CAT Mask R-CNN</a>
    </td>
    <td align="center">ResNet-50</td>
    <td align="center">1x</td>
    <td align="center">&cross;</td>
    <td align="center">46.2</td>
    <td align="center">38.5</td>
    <td align="center">
      <a href="https://dl.catcatdev.com/catnet/cat_mask_rcnn_r50_1x_isaid-571255e9.pth">model</a> |
      <a href="https://dl.catcatdev.com/catnet/cat_mask_rcnn_r50_1x_isaid.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/yeliudev/CATNet/blob/main/configs/isaid/cat_mask_rcnn_r50_aug_1x_isaid.py">CAT Mask R-CNN</a>
    </td>
    <td align="center">ResNet-50</td>
    <td align="center">1x</td>
    <td align="center">&check;</td>
    <td align="center">47.6</td>
    <td align="center">40.1</td>
    <td align="center">
      <a href="https://dl.catcatdev.com/catnet/cat_mask_rcnn_r50_aug_1x_isaid-e1fb2a8a.pth">model</a> |
      <a href="https://dl.catcatdev.com/catnet/cat_mask_rcnn_r50_aug_1x_isaid.json">metrics</a>
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
    <td align="center">76.3</td>
    <td align="center">—</td>
    <td align="center">
      <a href="https://dl.catcatdev.com/catnet/catnet_r50_3x_dior-ae22577c.pth">model</a> |
      <a href="https://dl.catcatdev.com/catnet/catnet_r50_3x_dior.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/yeliudev/CATNet/blob/main/configs/dior/catnet_r50_aug_3x_dior.py">CATNet</a>
    </td>
    <td align="center">ResNet-50</td>
    <td align="center">3x</td>
    <td align="center">&check;</td>
    <td align="center">78.6</td>
    <td align="center">—</td>
    <td align="center">
      <a href="https://dl.catcatdev.com/catnet/catnet_r50_aug_3x_dior-7f48c486.pth">model</a> |
      <a href="https://dl.catcatdev.com/catnet/catnet_r50_aug_3x_dior.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/yeliudev/CATNet/blob/main/configs/dior/cat_rcnn_r50_3x_dior.py">CAT R-CNN</a>
    </td>
    <td align="center">ResNet-50</td>
    <td align="center">3x</td>
    <td align="center">&cross;</td>
    <td align="center">77.7</td>
    <td align="center">—</td>
    <td align="center">
      <a href="https://dl.catcatdev.com/catnet/cat_rcnn_r50_3x_dior-a6c58f5f.pth">model</a> |
      <a href="https://dl.catcatdev.com/catnet/cat_rcnn_r50_3x_dior.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/yeliudev/CATNet/blob/main/configs/dior/cat_rcnn_r50_aug_3x_dior.py">CAT R-CNN</a>
    </td>
    <td align="center">ResNet-50</td>
    <td align="center">3x</td>
    <td align="center">&check;</td>
    <td align="center">81.9</td>
    <td align="center">—</td>
    <td align="center">
      <a href="https://dl.catcatdev.com/catnet/cat_rcnn_r50_aug_3x_dior-c46a7bf9.pth">model</a> |
      <a href="https://dl.catcatdev.com/catnet/cat_rcnn_r50_aug_3x_dior.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center" rowspan="4">
      <a href="https://doi.org/10.1016/j.isprsjprs.2014.10.002">NWPU<br>VHR-10</a>
    </td>
    <td align="center">
      <a href="https://github.com/yeliudev/CATNet/blob/main/configs/vhr/catnet_r50_6x_vhr.py">CATNet</a>
    </td>
    <td align="center">ResNet-50</td>
    <td align="center">6x</td>
    <td align="center">&cross;</td>
    <td align="center">95.8</td>
    <td align="center">—</td>
    <td align="center">
      <a href="https://dl.catcatdev.com/catnet/catnet_r50_6x_vhr-0be22cfa.pth">model</a> |
      <a href="https://dl.catcatdev.com/catnet/catnet_r50_6x_vhr.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/yeliudev/CATNet/blob/main/configs/vhr/catnet_r50_aug_6x_vhr.py">CATNet</a>
    </td>
    <td align="center">ResNet-50</td>
    <td align="center">6x</td>
    <td align="center">&check;</td>
    <td align="center">97.4</td>
    <td align="center">—</td>
    <td align="center">
      <a href="https://dl.catcatdev.com/catnet/catnet_r50_aug_6x_vhr-e2a969c8.pth">model</a> |
      <a href="https://dl.catcatdev.com/catnet/catnet_r50_aug_6x_vhr.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/yeliudev/CATNet/blob/main/configs/vhr/cat_rcnn_r50_6x_vhr.py">CAT R-CNN</a>
    </td>
    <td align="center">ResNet-50</td>
    <td align="center">6x</td>
    <td align="center">&cross;</td>
    <td align="center">96.4</td>
    <td align="center">—</td>
    <td align="center">
      <a href="https://dl.catcatdev.com/catnet/cat_rcnn_r50_6x_vhr-a1af678e.pth">model</a> |
      <a href="https://dl.catcatdev.com/catnet/cat_rcnn_r50_6x_vhr.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/yeliudev/CATNet/blob/main/configs/vhr/cat_rcnn_r50_aug_6x_vhr.py">CAT R-CNN</a>
    </td>
    <td align="center">ResNet-50</td>
    <td align="center">6x</td>
    <td align="center">&check;</td>
    <td align="center">97.7</td>
    <td align="center">—</td>
    <td align="center">
      <a href="https://dl.catcatdev.com/catnet/cat_rcnn_r50_aug_6x_vhr-8bb41746.pth">model</a> |
      <a href="https://dl.catcatdev.com/catnet/cat_rcnn_r50_aug_6x_vhr.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center" rowspan="4">
      <a href="https://doi.org/10.1109/access.2020.3005861">HRSID</a>
    </td>
    <td align="center">
      <a href="https://github.com/yeliudev/CATNet/blob/main/configs/hrsid/cat_mask_rcnn_r50_3x_hrsid.py">CAT Mask R-CNN</a>
    </td>
    <td align="center">ResNet-50</td>
    <td align="center">3x</td>
    <td align="center">&cross;</td>
    <td align="center">71.7</td>
    <td align="center">58.2</td>
    <td align="center">
      <a href="https://dl.catcatdev.com/catnet/cat_mask_rcnn_r50_3x_hrsid-42c4e091.pth">model</a> |
      <a href="https://dl.catcatdev.com/catnet/cat_mask_rcnn_r50_3x_hrsid.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/yeliudev/CATNet/blob/main/configs/hrsid/cat_mask_rcnn_r50_aug_3x_hrsid.py">CAT Mask R-CNN</a>
    </td>
    <td align="center">ResNet-50</td>
    <td align="center">3x</td>
    <td align="center">&check;</td>
    <td align="center">73.3</td>
    <td align="center">59.6</td>
    <td align="center">
      <a href="https://dl.catcatdev.com/catnet/cat_mask_rcnn_r50_aug_3x_hrsid-b43e2648.pth">model</a> |
      <a href="https://dl.catcatdev.com/catnet/cat_mask_rcnn_r50_aug_3x_hrsid.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/yeliudev/CATNet/blob/main/configs/hrsid/cat_rcnn_r50_3x_hrsid.py">CAT R-CNN</a>
    </td>
    <td align="center">ResNet-50</td>
    <td align="center">3x</td>
    <td align="center">&cross;</td>
    <td align="center">70.5</td>
    <td align="center">—</td>
    <td align="center">
      <a href="https://dl.catcatdev.com/catnet/cat_rcnn_r50_3x_hrsid-19886a3d.pth">model</a> |
      <a href="https://dl.catcatdev.com/catnet/cat_rcnn_r50_3x_hrsid.json">metrics</a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/yeliudev/CATNet/blob/main/configs/hrsid/cat_rcnn_r50_aug_3x_hrsid.py">CAT R-CNN</a>
    </td>
    <td align="center">ResNet-50</td>
    <td align="center">3x</td>
    <td align="center">&check;</td>
    <td align="center">72.8</td>
    <td align="center">—</td>
    <td align="center">
      <a href="https://dl.catcatdev.com/catnet/cat_rcnn_r50_aug_3x_hrsid-14a65873.pth">model</a> |
      <a href="https://dl.catcatdev.com/catnet/cat_rcnn_r50_aug_3x_hrsid.json">metrics</a>
    </td>
  </tr>
</table>

## Citation

If you find this project useful for your research, please kindly cite our paper.

```bibtex
@techreport{liu2021learning,
  title={Learning to Aggregate Multi-Scale Context for Instance Segmentation in Remote Sensing Images},
  author={Liu, Ye and Li, Huifang and Hu, Chao and Luo, Shuang and Luo, Yan and Chen, Chang Wen},
  number={arXiv:2111.11057},
  year={2021}
}
```
