# CDKD-SOD

# CDKD-SOD
The official PyTorch implementation of **Cross-modal Depth-privileged Knowledge Distillation for Underwater Salient Object Detection with Only RGB Images** ([IEEE CAOS 2026](https://doi.org/10.1109/CAOS69069.2026.11666192)).

We test our code with the image built from the provided `Dockerfile`: Ubuntu 22.04, CUDA 12.8.1 with cuDNN, and PyTorch 2.8.0. You can modify the `Dockerfile` as you want.

# Introduction
CDKD-SOD is a cross-modal knowledge distillation framework for underwater salient object detection. It transfers the detection capability of an RGB-Depth teacher network to a lightweight student network that uses only RGB images at inference time.

The teacher network is [PotCrackSeg](https://github.com/lab-sun/PotCrackSeg), an RGB-D fusion network with two SegFormer-B2 encoders. The student network keeps the RGB stream of the teacher and replaces the encoder with SegFormer-B0. An extra convolution-batch-normalization-ReLU branch in the student decoder learns pseudo-depth features from RGB images. The student is supervised by ground truth together with the teacher's semantic predictions, fused predictions, and fused features.

# Dataset
We train and test on [USOD10K](https://github.com/LinHong-HIT/USOD10K), which contains 10,255 pairs of underwater RGB images and estimated depth maps at a resolution of 640×480. The official split is 7,178 training pairs, 2,051 validation pairs, and 1,026 test pairs.

Download USOD10K from the official repository, then arrange it as follows. `train.txt`, `val.txt`, and `test.txt` list image names without the extension, one name per line. RGB images, depth maps, and ground-truth masks that share the same name are stored as `.png` files.

```
dataset/
├── train.txt
├── val.txt
├── test.txt
├── RGB/
├── depth/
└── labels/
```

# Pretrained weights
The encoders are initialized with the SegFormer pretrained weights. Download MiT-B0 and MiT-B2 from the [SegFormer](https://github.com/NVlabs/SegFormer) release, and place them as below. `run_demo_PotCrackSeg.py` reads the teacher backbone weight from `pretrained/segformer/mit_b2.pth`.

```
pretrained/
├── mit_b0.pth
├── mit_b2.pth
└── segformer/
    └── mit_b2.pth
```

To reproduce the reported results, put the trained checkpoints in `weights_backup`:

```
weights_backup/
├── PotCrackSeg/
│   └── final.pth
└── Teacher_StudentT/
    └── final.pth
```

`PotCrackSeg/final.pth` is the RGB-D teacher. `Teacher_StudentT/final.pth` is the RGB-only student. If you train the models yourself, copy the selected teacher checkpoint to `weights_backup/PotCrackSeg/final.pth` before training or testing the student.

# Usage
* Clone this repo
```
$ git clone https://github.com/lab-sun/CDKD-SOD.git
```
* Build the docker image
```
$ cd ~/CDKD-SOD
$ docker build -t docker_image_cdkd .
```
* Download the dataset
```
$ (You should be in the CDKD-SOD folder)
$ mkdir ./dataset
$ (download USOD10K and arrange RGB, depth, labels, and the split lists as described above)
```
* To reproduce our results, prepare the pretrained weights described above.
```
$ (You should be in the CDKD-SOD folder)
$ docker run -it --shm-size 8G -p 1234:6006 --name docker_container_cdkd --gpus all -v ~/CDKD-SOD:/workspace docker_image_cdkd
$ (currently, you should be in the docker)
$ cd /workspace
$ (To reproduce the results of the RGB-only student network)
$ python3 run_demo_Teacher_Student.py --gpu 0
$ (To reproduce the results of the RGB-D teacher network PotCrackSeg)
$ python3 run_demo_PotCrackSeg.py --gpu 0
```
The visualization results will be saved in the `./runs` folder.
* To train CDKD-SOD (teacher network PotCrackSeg and the RGB-only student network)
```
$ (You should be in the CDKD-SOD folder)
$ docker run -it --shm-size 8G -p 1234:6006 --name docker_container_cdkd --gpus all -v ~/CDKD-SOD:/workspace docker_image_cdkd
$ (currently, you should be in the docker)
$ cd /workspace
$ (To train the RGB-D teacher network PotCrackSeg)
$ python3 train_PorCrackSeg.py --gpu 0
$ (Copy the selected teacher checkpoint to ./weights_backup/PotCrackSeg/final.pth)
$ (To train the RGB-only student network. The paper reports the student trained for 5 epochs.)
$ python3 train_Teacher_Student.py --gpu 0 --epoch_max 5
```
Teacher checkpoints are saved in `./PotCrackSeg/PotCrackSeg/`. Student checkpoints are saved in `./Teacher_Student/Student/`. Both training scripts clear these folders at the start of a new run.
* To see the training process
```
$ (fire up another terminal)
$ docker exec -it docker_container_cdkd /bin/bash
$ cd /workspace
$ tensorboard --bind_all --logdir=./Teacher_Student/tensorboard_log/
$ (fire up your favorite browser with http://localhost:1234, you will see the tensorboard)
```
The teacher log is in `./PotCrackSeg/tensorboard_log/`. Use that folder as `--logdir` when you are training PotCrackSeg.

If you have the error `docker: Error response from daemon: could not select device driver`, please first install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on your computer.

# Citation
If you use CDKD-SOD in your academic work, please cite:
```
@INPROCEEDINGS{feng2026cdkdsod,
  author={Feng, Zhen and Sun, Yuxiang},
  booktitle={2026 IEEE Canadian Atlantic Ocean Symposium (CAOS)},
  title={Cross-Modal Depth-Privileged Knowledge Distillation for Underwater Salient Object Detection with Only RGB Images},
  year={2026},
  pages={160-164},
  doi={10.1109/CAOS69069.2026.11666192}}
```

# Acknowledgement
The teacher network is [PotCrackSeg](https://github.com/lab-sun/PotCrackSeg). The encoders are [SegFormer](https://github.com/NVlabs/SegFormer). The experiments use the [USOD10K](https://github.com/LinHong-HIT/USOD10K) dataset. Some of the training code follows the style of [CEKD](https://github.com/lab-sun/CEKD) and [RTFNet](https://github.com/yuxiangsun/RTFNet).
