# Online Motion Style Transfer

## Overview
This repository provides a re-implementation of the [Style-ERD: Responsive and Coherent Online Motion Style Transfer](https://openaccess.thecvf.com/content/CVPR2022/papers/Tao_Style-ERD_Responsive_and_Coherent_Online_Motion_Style_Transfer_CVPR_2022_paper.pdf) model using [moai](https://github.com/moverseai/moai). The original implementation is available in the [Online-Motion-Style-Transfer](https://github.com/tianxintao/Online-Motion-Style-Transfer) repository. In addition to the original dataset, this repository offers extra training scripts for the recently introduced [100Style](https://www.ianxmason.com/100style/) dataset, which has been also retargeted to the SMPL skeleton. You can find the retargeted dataset in the [SMooDi](https://github.com/neu-vi/SMooDi) repository.

The proposed restyling framework is composed of three main modules:
- **Style Transfer Module**
- **Style Supervision Module**
- **Content Supervision Module**

The architecture features an encoder—implemented as a recurrent module with residual connections—and a decoder. For style supervision, the FT-Att Discriminator integrates feature and temporal attention mechanisms, while the content supervision module employs a perceptual loss along with a pre-trained content classification network.

## Training on the Original Dataset
To train the ERD model using the original dataset, run the following command:

```bash
moai run fit conf/xia/main.yaml \
  +XIA_PATH_DATA=<PATH_TO_NPZ> \
  +XIA_PATH_SKEL_PATH=<PATH_TO_CMU_SKELETON> \
  +CRITIC_REAL_W=1.0 +CRITIC_FAKE_W=1.0 \
  +LATENT_DIM=32 +ENCODER_LAYER_NUM=2 +DECODER_LAYER_NUM=4 \
  +NEUTRAL_LAYER_NUM=4 +STYLE_LAYER_NUM=6 \
  +POSE_W=0.5 +VELOCITY_W=1.0 +QUAT_W=0.05 \
  +ADVERSARIAL_W=1.0 \
  +EPISODE_LENGTH=24 +EPISODE_STRIDE=12 +EPISODE_LENGTH_MINUS_ONE=23 \
  +GRADIENT_PENALTY_W=128
```

## Training on the 100Style Dataset
This implementation extends the original model by incorporating training scripts for the latest 100Style dataset.

*Note: SMPL is required.*

To train the model on the 100Style dataset, execute the following command:

```bash
moai run fit conf/hundred/main.yaml \
  +DATA_ROOT=<PATH_TO_100_STYLE> \
  +SMPL_MODELS_ROOT=<PATH_TO_SMPL_BODY_ARTIFACTS> \
  +CRITIC_REAL_W=1.0 +CRITIC_FAKE_W=1.0 \
  +LATENT_DIM=32 +ENCODER_LAYER_NUM=2 +DECODER_LAYER_NUM=4 \
  +NEUTRAL_LAYER_NUM=4 +STYLE_LAYER_NUM=6 \
  +POSE_W=0.5 +VELOCITY_W=1.0 +QUAT_W=0.05 \
  +ADVERSARIAL_W=1.0 \
  +STYLE_TO_TRANSFER=<STYLE_TO_BE_TRANSFERED> \
  +EPISODE_LENGTH=24 +EPISODE_STRIDE=12 +EPISODE_LENGTH_MINUS_ONE=23 \
  +CKPT="" +GRADIENT_PENALTY_W=128
```

## Evaluating on the 100Style Dataset

To test the model and visualize the results using rerun, run the following command:

```bash
moai run test conf/hundred/main.yaml \
  +DATA_ROOT=<PATH_TO_100_STYLE> \
  +SMPL_MODELS_ROOT=<PATH_TO_SMPL_BODY_ARTIFACTS> \
  +CRITIC_REAL_W=1.0 +CRITIC_FAKE_W=1.0 \
  +LATENT_DIM=32 +ENCODER_LAYER_NUM=2 +DECODER_LAYER_NUM=4 \
  +NEUTRAL_LAYER_NUM=4 +STYLE_LAYER_NUM=6 \
  +POSE_W=0.5 +VELOCITY_W=1.0 +QUAT_W=0.05 \
  +ADVERSARIAL_W=1.0 \
  +STYLE_TO_TRANSFER=<STYLE_TO_BE_TRANSFERED> \
  +EPISODE_LENGTH=24 \
  +EPISODE_STRIDE=12 \
  +EPISODE_LENGTH_MINUS_ONE=23 \
  +GRADIENT_PENALTY_W=128 \
  +CKPT=<PATH_TO_PRETRAINED_MODEL>
```