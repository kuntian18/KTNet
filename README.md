## Knowledge Mining and Transferring for Domain Adaptive Object Detection



## Abstract

![](/figs/architecture-Small.png)


With the thriving of deep learning, CNN-based object detectors have made great progress in the past decade. However, the domain gap between training and testing data leads to a prominent performance degradation and thus hinders their application in the real world. To alleviate this problem, Knowledge Transfer Network (KTNet) is proposed as a new paradigm for domain adaption. Specifically, KTNet is constructed on a base detector with intrinsic knowledge mining and relational knowledge constraints. First, we design a foreground/background classifier shared by source domain and target domain to extract the common attribute knowledge of objects in different scenarios. Second, we model the relational knowledge graph and explicitly constrain the consistency of category correlation under source domain, target domain, as well as cross-domain conditions. As a result, the detector is guided to learn object-related and domain-independent representation. Extensive experiments and visualizations confirm that transferring objectspecific knowledge can yield notable performance gains. The proposed KTNet achieves state-of-the-art results on three cross-domain detection benchmarks



## Installation 

Check [INSTALL.md](https://github.com/kuntian18/KTNet/blob/master/INSTALL.md) for installation instructions. 

The implementation of our method is heavily based on FCOS ([\#f0a9731](https://github.com/tianzhi0549/FCOS/tree/f0a9731dac1346788cc30d5751177f2695caaa1f)) and [EPM](https://github.com/chengchunhsu/EveryPixelMatters).

## Dataset

We construct the training and testing set by three following settings:

- Cityscapes -> Foggy Cityscapes
  - Download Cityscapes and Foggy Cityscapes dataset from the [link](https://www.cityscapes-dataset.com/downloads/). Particularly, we use *leftImg8bit_trainvaltest.zip* for Cityscapes and *leftImg8bit_trainvaltest_foggy.zip* for Foggy Cityscapes.
  - Download and extract the converted annotation from the following links: [Cityscapes and Foggy Cityscapes (COCO format)](https://drive.google.com/file/d/1uvcyIPwR_4ZwFRJSiFtcBjlG8i5k6khE/view?usp=sharing).
  - Extract the training sets from *leftImg8bit_trainvaltest.zip*, then move the folder `leftImg8bit/train/` to `Cityscapes/leftImg8bit/` directory.
  - Extract the training and validation set from *leftImg8bit_trainvaltest_foggy.zip*, then move the folder `leftImg8bit_foggy/train/` and `leftImg8bit_foggy/val/` to `Cityscapes/leftImg8bit_foggy/` directory.
- Sim10k -> Cityscapes (class car only)
  - Download Sim10k dataset and Cityscapes dataset from the following links: [Sim10k](https://fcav.engin.umich.edu/projects/driving-in-the-matrix) and [Cityscapes](https://www.cityscapes-dataset.com/downloads/). Particularly, we use *repro_10k_images.tgz* and *repro_10k_annotations.tgz* for Sim10k and *leftImg8bit_trainvaltest.zip* for Cityscapes.
  - Download and extract the converted annotation from the following links: [Sim10k (VOC format)](https://drive.google.com/file/d/1OK59-_5JK3ADJgB0z82t-5FqwCjHYHJd/view?usp=sharing) and [Cityscapes (COCO format)](https://drive.google.com/file/d/1uvcyIPwR_4ZwFRJSiFtcBjlG8i5k6khE/view?usp=sharing).
  - Extract the training set from *repro_10k_images.tgz* and *repro_10k_annotations.tgz*, then move all images under `VOC2012/JPEGImages/` to `Sim10k/JPEGImages/` directory and move all annotations under `VOC2012/Annotations/` to `Sim10k/Annotations/`.
  - Extract the training and validation set from *leftImg8bit_trainvaltest.zip*, then move the folder `leftImg8bit/train/` and `leftImg8bit/val/` to `Cityscapes/leftImg8bit/` directory.
- KITTI -> Cityscapes (class car only)
  - Download KITTI dataset and Cityscapes dataset from the following links: [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d) and [Cityscapes](https://www.cityscapes-dataset.com/downloads/). Particularly, we use *data_object_image_2.zip* for KITTI and *leftImg8bit_trainvaltest.zip* for Cityscapes.
  - Download and extract the converted annotation from the following links: [KITTI (VOC format)](https://drive.google.com/file/d/1lMGPX9iT1z3KZ6DEzG-p2JSfnVgvOjEQ/view?usp=sharing) and [Cityscapes (COCO format)](https://drive.google.com/file/d/1uvcyIPwR_4ZwFRJSiFtcBjlG8i5k6khE/view?usp=sharing).
  - Extract the training set from *data_object_image_2.zip*, then move all images under `training/image_2/` to `KITTI/JPEGImages/` directory.
  - Extract the training and validation set from *leftImg8bit_trainvaltest.zip*, then move the folder `leftImg8bit/train/` and `leftImg8bit/val/` to `Cityscapes/leftImg8bit/` directory.



After the preparation, the dataset should be stored as follows:

```
[DATASET_PATH]
└─ Cityscapes
   └─ cocoAnnotations
   └─ leftImg8bit
      └─ train
      └─ val
   └─ leftImg8bit_foggy
      └─ train
      └─ val
└─ KITTI
   └─ Annotations
   └─ ImageSets
   └─ JPEGImages
└─ Sim10k
   └─ Annotations
   └─ ImageSets
   └─ JPEGImages
```



**Format and Path**

Before training, please checked [paths_catalog.py](https://github.com/kuntian18/KTNet/blob/master/fcos_core/config/paths_catalog.py) and enter the correct data path for:

- `DATA_DIR`
- `cityscapes_train_cocostyle`, `cityscapes_foggy_train_cocostyle` and `cityscapes_foggy_val_cocostyle` (for Cityscapes -> Foggy Cityscapes).
- `sim10k_trainval_caronly`, `cityscapes_train_caronly_cocostyle` and `cityscapes_val_caronly_cocostyle` (for Sim10k -> Cityscapes).
- `kitti_train_caronly`, `cityscapes_train_caronly_cocostyle` and `cityscapes_val_caronly_cocostyle` (for KITTI -> Cityscapes).



For example, if the datasets have been stored as the way we mentioned, the paths should be set as follows:

- Dataset directory (In L8):

  ```
  DATA_DIR = [DATASET_PATH]
  ```

- Train and validation set directory for each dataset:

  ```
  "cityscapes_train_cocostyle": {
      "img_dir": "Cityscapes/leftImg8bit/train",
      "ann_file": "Cityscapes/cocoAnnotations/cityscapes_train_cocostyle.json"
  },
  "cityscapes_train_caronly_cocostyle": {
      "img_dir": "Cityscapes/leftImg8bit/train",
      "ann_file": "Cityscapes/cocoAnnotations/cityscapes_train_caronly_cocostyle.json"
  },
  "cityscapes_val_caronly_cocostyle": {
      "img_dir": "Cityscapes/leftImg8bit/val",
      "ann_file": "Cityscapes/cocoAnnotations/cityscapes_val_caronly_cocostyle.json"
  },
  "cityscapes_foggy_train_cocostyle": {
      "img_dir": "Cityscapes/leftImg8bit_foggy/train",
      "ann_file": "Cityscapes/cocoAnnotations/cityscapes_foggy_train_cocostyle.json"
  },
  "cityscapes_foggy_val_cocostyle": {
      "img_dir": "Cityscapes/leftImg8bit_foggy/val",
      "ann_file": "Cityscapes/cocoAnnotations/cityscapes_foggy_val_cocostyle.json"
  },
  "sim10k_trainval_caronly": {
    "data_dir": "Sim10k",
      "split": "trainval10k_caronly"
  },
  "kitti_train_caronly": {
      "data_dir": "KITTI",
      "split": "train_caronly"
  },
  ```
  
  

**(Optional) Format Conversion**

If you want to construct the dataset and convert data format manually, here are some useful links:

- [yuhuayc/da-faster-rcnn](https://github.com/yuhuayc/da-faster-rcnn)
- [krumo/Detectron-DA-Faster-RCNN](https://github.com/krumo/Detectron-DA-Faster-RCNN)



## Training

Let's take the Cityscapes -> Foggy Cityscapes experiment as an example.


Run the bash files directly:

- Using VGG-16 as backbone with 2 GPUs

  ```
  bash ./scripts/train_city_vgg16.sh
  ```

- Using ResNet-101 as backbone with 2 GPUs

  ```
  bash ./scripts/train_city_resnet.sh
  ```

- (Optional) Using VGG-16 as backbone with single GPU

  ```
  bash ./scripts/single_gpu/train_city_vgg16_single_gpu.sh
  ```

  

or type the bash commands:

- Using VGG-16 as backbone with 2 GPUs

  ```
  python -m torch.distributed.launch \
      --nproc_per_node=2 \
      --master_port=$((RANDOM + 10000)) \
      tools/train_net_da.py \
      --config-file ./configs/da_ktnet_city_VGG_16.yaml
  ```

- Using ResNet-101 as backbone with 2 GPUs

  ```
  python -m torch.distributed.launch \
      --nproc_per_node=2 \
      --master_port=$((RANDOM + 10000)) \
      tools/train_net_da.py \
      --config-file ./configs/da_ktnet_city_R_101.yaml
  ```

- (Optional) Using VGG-16 as backbone with single GPU

  ```
  python tools/train_net_da.py \
      --config-file ./configs/da_ktnet_city_VGG_16.yaml \
  ```







## Evaluation

The trained model can be evaluated by the following command.

```
python tools/test_net.py \
	--config-file [CONFIG_PATH] \
	MODEL.WEIGHT [WEIGHT_PATH] \
	TEST.IMS_PER_BATCH 4
```

- `[CONFIG_PATH]`: Path of config file
- `[WEIGHT_PATH]`: Path of model weight for evaluation



For example, the following command evaluates the model weight `vgg_city.pth` for Cityscapes -> Foggy Cityscapes using VGG-16 backbone.

```
python tools/test_net.py \
	--config-file configs/da_ktnet_city_VGG_16.yaml \
	MODEL.WEIGHT "vgg_city.pth" \
	TEST.IMS_PER_BATCH 4
```



Note that the commands for evaluation are completely derived from FCOS ([\#f0a9731](https://github.com/tianzhi0549/FCOS/tree/f0a9731dac1346788cc30d5751177f2695caaa1f)).

Please see [here](https://github.com/tianzhi0549/FCOS/tree/f0a9731dac1346788cc30d5751177f2695caaa1f#inference) for more details.





## Result

We provide the experimental results and parameter weights of our best models (using dynamic conditional queues) in this section.



| Dataset                        | Backbone | mAP  | mAP@0.50 | mAP@0.75 | mAP@S | mAP@M | mAP@L | Model                            |
| ------------------------------ | -------- | ---- | -------- | -------- | ----- | ----- | ----- | ------------------------------------------------------------ |
| Cityscapes -> Foggy Cityscapes | VGG-16   | 24.3 | 43.6     | 22.8     | 4.6   | 22.3  | 44.7  | [link](https://drive.google.com/file/d/1R93EHqpoY4jfmPZg6CqhQQ8tTkzipmUH/view?usp=sharing) |
| Sim10k -> Cityscapes           | VGG-16   | 30.6 | 52.5     | 31.0     | 6.5   | 33.4  | 62.6  | [link](https://drive.google.com/file/d/16wMz3ljXOVHimwJOn7J0ObiyrAF4PP5X/view?usp=sharing) |
| KITTI -> Cityscapes            | VGG-16   | 23.3 | 46.9     | 21.2     | 6.7   | 29.1  | 44.8  | [link](https://drive.google.com/file/d/1DsFZ8TC7M4LrjiUtAJ-FrkHeMxzbTfv8/view?usp=sharing) |



**Environments**

- Hardware
  - 2 NVIDIA 1080 Ti GPUs

- Software
  - PyTorch 1.7.0
  - Torchvision 0.2.1
  - CUDA 10.2


<details open>
<summary> <font size=3>Todo list</font> </summary>

- [x] Release preliminary code and checkpoints
  
- [ ] Release dynamic conditional queues (CQ) for discriminative category features
 
- [ ] Release extension implementations on other two detector architectures
 
</details>


## Citations

Please consider citing our paper in your publications if the project helps your research.

```
@inproceedings{tian2021knowledge,
  title={Knowledge mining and transferring for domain adaptive object detection},
  author={Tian, Kun and Zhang, Chenghao and Wang, Ying and Xiang, Shiming and Pan, Chunhong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9133--9142},
  year={2021}
}
```
