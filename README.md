# SFAKD: Smoothed Frequency Augmentation for Knowledge Distillation

## Introduction

**SFAKD (Smoothed Frequency Augmentation for Knowledge Distillation)** is a novel knowledge distillation method that enhances a student model’s feature representation by leveraging **frequency-domain feature augmentation**. Unlike conventional KD methods focused only on spatial feature alignment, SFAKD introduces a complementary **frequency domain branch** to smooth and modulate the student’s features in the Fourier domain. The student’s final convolutional feature map is first transformed via a 2D FFT, then passed through a **learnable complex frequency filter** (applied as an element-wise multiplication in the frequency domain). Next, a **tunable Gaussian mask** is applied on the shifted frequency spectrum to softly suppress redundant low-frequency components. This **“soft frequency smoothing”** retains the complete frequency structure while highlighting discriminative details, avoiding the information loss and structural damage of hard frequency truncation. An inverse FFT then brings the modulated feature back to the spatial domain.

To preserve spatial information, SFAKD also employs a parallel **spatial feature augmentation** branch. The student’s original spatial feature is passed through a \$1\times1\$ convolution, and then **dynamically fused** with the frequency-enhanced feature via learnable weights. This fused feature (containing both spatial structure and frequency-enhanced details) is fed into an **adaptive bottleneck projector**. The projector uses an intermediate channel dimension of \$\left\lfloor\frac{s\_n + t\_n}{2}\right\rfloor\$ (half the sum of student and teacher feature channels) to ensure sufficient capacity for alignment while controlling parameter size. It is implemented with a lightweight sequence of conv layers including a **depthwise separable convolution** to reduce computation. A global average pooling is applied to the projector’s output, which is then **aligned with the teacher’s final feature** via a strict \$L^2\$ loss during training. Through this **frequency–spatial collaborative distillation paradigm**, SFAKD substantially improves the student’s ability to capture the teacher’s core knowledge patterns. Notably, SFAKD’s design is simple and low-overhead, requiring only a few additional layers, yet it yields significant performance gains. Experiments on CIFAR-100 and ImageNet demonstrate that SFAKD achieves state-of-the-art results across many teacher–student pairs, often outperforming prior distillation methods. In some cases, the SFAKD-trained student even **matches or surpasses the teacher’s accuracy**, highlighting the efficacy and generality of the approach.

*SFAKD is implemented on top of the open-source **MDistiller** framework.* This means that SFAKD integrates seamlessly with the training scripts, model zoo, and evaluation tools provided by MDistiller. In the following, we describe how to set up the environment and use SFAKD, and we provide guidelines for reproducing our results.

## Installation and Setup

Our code is based on **MDistiller** (Megvii Research’s knowledge distillation library). Please ensure you have the following environment prerequisites (consistent with MDistiller’s requirements):

* Python 3.8 (or later)
* PyTorch 2.4.0
* torchvision 0.19.0

First, clone this repository and install the dependencies:

```bash
# Install required packages
pip install -r requirements.txt

```

This will install MDistiller and the SFAKD extension in your environment.

**Optional:** By default, MDistiller uses Weights & Biases (wandb) for logging. If you want to disable wandb logging, set `CFG.LOG.WANDB = False` in the configuration (e.g. in `mdistiller/engine/cfg.py`).

## Data Preparation

SFAKD supports distillation on **CIFAR-100** and **ImageNet**. You should prepare datasets and pre-trained teacher models as follows:

* **CIFAR-100:** If not already available, the CIFAR-100 dataset will be automatically downloaded by torchvision upon first use. For distillation, we utilize pre-trained teacher model weights. Download the provided CIFAR-100 teacher checkpoints from the MDistiller release page (e.g., `cifar_teachers.tar`) and extract them to the `./download_ckpts` directory. This includes teacher models like ResNet56, ResNet110, ResNet32x4, WRN-40-2, VGG13, etc., which are used in our experiments. The training configs reference these weights so that the teacher models can be loaded without additional training.

* **ImageNet:** Download the ImageNet-1k dataset (ILSRC2012) and place it under `./data/imagenet/` with the standard folder structure (`train/` and `val/` subfolders). For teacher models on ImageNet, we use standard pretrained weights (e.g., from torchvision or the MDistiller checkpoints). Ensure the teacher network (e.g., ResNet34 or ResNet50) is either loaded from MDistiller’s checkpoint release or uses torchvision’s pretrained model (the config can specify the weight path or use `pretrained=True` by default for certain models).

## Training & Evaluation

We provide configuration files under `configs/` for various teacher-student pairs on CIFAR-100 and ImageNet using the SFAKD distiller. You can launch training using MDistiller’s training script `tools/train.py` with our SFAKD configs.

**Training on CIFAR-100:** For example, to distill a ResNet32x4 teacher into a ResNet8x4 student using SFAKD on CIFAR-100, run:

```bash
# Train ResNet32x4 -> ResNet8x4 with SFAKD (CIFAR-100)
python3 tools/train.py --cfg configs/cifar100/sfakd/res32x4_res8x4.yaml
```

This will load the ResNet32x4 teacher (from `download_ckpts`) and train the ResNet8x4 student with our SFAKD method. All training hyper-parameters (optimization settings, distillation loss weights, etc.) are specified in the YAML config. You can override parameters via command line if needed; for instance, to change batch size or learning rate:

```bash
python3 tools/train.py --cfg configs/cifar100/sfakd/res32x4_res8x4.yaml SOLVER.BATCH_SIZE 128 SOLVER.LR 0.1
```



**Training on ImageNet:** Similarly, to distill a ResNet34 teacher into a ResNet18 student on ImageNet using SFAKD, run:

```bash
# Train ResNet34 -> ResNet18 with SFAKD (ImageNet)
python3 tools/train.py --cfg configs/imagenet/r34_r18/sfakd.yaml -d imagenet
```

Here `-d imagenet` specifies the ImageNet dataset (as opposed to the default CIFAR). Ensure the ImageNet data path is correctly set (`./data/imagenet` as prepared above). Training on ImageNet uses SGD for 100 epochs with an initial LR of 0.2 and step decays at epoch 30, 60, 90 (these defaults are set in the config).

**Evaluation:** After training, you can evaluate the distilled student’s performance using the `tools/eval.py` script. For example:

```bash
# Evaluate a CIFAR-100 model (ResNet8x4 student)
python3 tools/eval.py -m resnet8x4 -c output/sfakd_res32x4_res8x4/student_best.pth

# Evaluate an ImageNet model (MobileNet-V1 student)
python3 tools/eval.py -m MobileNetV1 -c output/sfakd_res50_mv1/student_best.pth -d imagenet
```

In the above, `-m` specifies the student model architecture and `-c` provides the checkpoint file to evaluate. You can also evaluate the teacher models (to verify teacher accuracy) by using `-m <TeacherModelName>` and the corresponding teacher weight path. For reference, MDistiller’s repository provides pretrained teacher accuracies and evaluation scripts for all supported models.

## Results and Benchmarks

### CIFAR-100 Experiments

We compare SFAKD with the standard Knowledge Distillation (KD) baseline across various teacher–student pairs on CIFAR-100. Below, we report Top-1 accuracies (%) for students distilled with vanilla KD vs. with **SFAKD**. We group results by **isomorphic** (homogeneous) architecture pairs and **heterogeneous** pairs.

**Homogeneous teacher–student pairs (CIFAR-100):**

| Teacher <br> Student | ResNet56 <br> ResNet20 | ResNet110 <br> ResNet32 | ResNet32x4 <br> ResNet8x4 | WRN-40-2 <br> WRN-16-2 | WRN-40-2 <br> WRN-40-1 | VGG13 <br> VGG8 |
| :------------------: | :--------------------: | :---------------------: | :-----------------------: | :--------------------: | :--------------------: | :-------------: |
|   **KD (baseline)**  |          70.66         |          73.08          |           73.33           |          74.92         |          73.54         |      72.98      |
|   **SFAKD (ours)**   |        **72.5**        |         **75.1**        |          **78.1**         |        **77.9**        |        **76.5**        |     **75.8**    |

**Heterogeneous teacher–student pairs (CIFAR-100):**

| Teacher <br> Student | ResNet32x4 <br> ShuffleNet-V1 | WRN-40-2 <br> ShuffleNet-V1 | VGG13 <br> MobileNet-V2 | ResNet50 <br> MobileNet-V2 | ResNet32x4 <br> MobileNet-V2 |
| :------------------: | :---------------------------: | :-------------------------: | :---------------------: | :------------------------: | :--------------------------: |
|   **KD (baseline)**  |             74.07             |            74.83            |          67.37          |            67.35           |             74.45            |
|   **SFAKD (ours)**   |            **77.8**           |           **78.0**          |         **71.0**        |          **72.5**          |           **78.5**           |

*SFAKD consistently outperforms vanilla KD across all evaluated pairs.* In **homogeneous** distillation settings, SFAKD yields substantial gains over KD (often +2% to +5% absolute). Notably, for a large teacher like ResNet32x4 and a smaller ResNet8x4 student, SFAKD improves the student from 73.33% to 78.1%, narrowing the gap to the teacher (≈79.4% on CIFAR-100) to nearly 1%. In some cases, SFAKD-trained students even surpass their teachers: for example, VGG8 surpasses its VGG13 teacher (75.8% vs. \~74%) and a WRN-40-1 student exceeds the WRN-40-2 teacher. These results highlight SFAKD’s ability to transfer knowledge so effectively that a student can outperform a larger teacher model. In **heterogeneous (cross-architecture)** scenarios, SFAKD likewise demonstrates strong improvements. For instance, when distilling a WRN-40-2 (Wide ResNet) teacher into a ShuffleNet-V1 student, SFAKD boosts the student to 78.0%, which is above the teacher’s accuracy (\~75%). Similarly, a MobileNet-V2 student distilled from a ResNet50 gains about 5% over KD, reaching 72.5%. Overall, SFAKD exhibits **broad applicability** and robustness for both similar-architecture and different-architecture distillation, achieving **state-of-the-art** results on CIFAR-100.

### ImageNet Experiments

We also evaluate SFAKD on the ImageNet benchmark, using standard teacher–student pairs:

| Teacher <br> Student | ResNet34 <br> ResNet18 | ResNet50 <br> MobileNet-V1 |
| :------------------: | :--------------------: | :------------------------: |
|   **KD (baseline)**  |          71.03         |            70.50           |
|   **SFAKD (ours)**   |        **72.5**        |          **73.0**          |

On ImageNet, SFAKD again provides notable gains over baseline KD. With a ResNet34 teacher, a ResNet18 student reaches 72.5% top-1 (vs. 71.0% with KD), a solid +1.5% improvement. With a ResNet50 teacher and MobileNet-V1 student, SFAKD achieves 73.0%, which is **+2.5%** better than KD and even surpasses some prior advanced distillation methods on this challenging heterogenous pair. These results underscore that SFAKD’s benefits translate to large-scale data: the frequency-based augmentation and alignment strategy helps the student network learn richer representations and generalize better on ImageNet.

**Overall, SFAKD sets a new state-of-the-art for knowledge distillation on both CIFAR-100 and ImageNet**, outperforming conventional KD and existing methods like DKD, ReviewKD, etc., especially in challenging cross-architecture cases. We recommend SFAKD for scenarios where maximizing student model performance is critical, as it offers an excellent trade-off of simplicity and effectiveness, requiring minimal architectural changes for substantial gains.

## How to Cite

If you find this project or the SFAKD method useful in your research, please cite the paper:

```bibtex
@article{you2025sfakd,
  title={{Smooth the Signal, Distill the Essence: A Frequency-Aware Modulation Framework for Knowledge Distillation}},
  author={You, Y. and Zhang, ... and others},
  year={2025},
  journal={arXiv preprint arXiv:XXXXX}  // or Conference submission
}
```

*(Replace “XXXXX” with the actual identifier when available.)*

## License

This project is released under the MIT License. See the [LICENSE](./LICENSE) file for details. SFAKD builds upon the MDistiller framework which is also MIT-licensed, allowing for free academic and commercial use. Enjoy distilling with SFAKD!
