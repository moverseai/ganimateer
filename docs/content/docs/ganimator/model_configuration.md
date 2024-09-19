---
title: Model Architecture
weight: 2
prev: /docs/ganimator/data_preprocess
next: /docs/ganimator/motion_generation
---

## Train GANimator

{{< filetree/container >}}
  {{< filetree/folder name="conf/ganimator" >}}
    {{< filetree/file name="data.yaml" >}}
    {{< filetree/file name="main.yaml" >}}
    {{< filetree/file name="model.yaml" >}}
    {{< filetree/file name="optim.yaml" >}}
    {{< filetree/file name="scheduling.yaml" >}}
    {{< filetree/file name="util.yaml" >}}
  {{< /filetree/folder >}}
{{< /filetree/container >}}

### Data

GANimator uses a single training sample and this is the reason we use a loader that repeatedly loads the same `.npz` file for both the train and the validation processes:

```yaml {filename="data.yaml"}
# @package _global_

npz_filename: ???
train_data_iter: 15000

data:
  train:
    iterator:
      datasets:
        repeated_npz:
          filename: ${npz_filename}
          length: ${train_data_iter}

  val:
    loader:
      batch_size: 1
    iterator:
      datasets:
        repeated_npz:
          filename: ${npz_filename}
          length: 200
```

### Model Architecture

The model consists of 4 stages, with the first 3 stages comprising 2 generator and 2 discriminator, while the last stage includes 1 generator and 1 discriminator. The overall structure of the stage is the following:

```yaml {filename="model.yaml"}
generator_sX: #combined generators for stages 1-3
  noise0: [noise_level_0]
  noise1: [noise_level_1]
  generated: [prev]
  _out_: [fake]

discriminator_sX: #combined discriminator for stages 1-3
  fake0:
    - fake.stage0
    - motion_data_level_x
  fake1:
    - fake.stage1
    - motion_data_level_y
  _out_:
    - fake_score
    - real_score
```
The generation part of the stage receives an amplified noisy input for the 1st generator (`noise0`) and another one for the 2nd generator (`noise1`), the generated motion features of the previous stage (`generated`) and returns 2 generated (fake) motion features - one per generator. The corresponding discrimination part receives the 2 generated features (`fake0`, `fake1`) and the corresponding real ones (`motion_data_level_x`, `motion_data_level_y`) and returns a score for both.


### Gradual Learning

One major difference with the original GANimator implementation is the ability to train all strages sequentially as an end-to-end process. This is achieved by exploiting `moai`'s scheduling functionality:

```yaml {filename="metrics.yaml"}
# @package _global_
_moai_:
  _execution_:
    _schedule_:
      - _epoch_: 1
        _modifications_:
          forward_mode:
            modules:
              generator_s1: eval_nograd
        _fit_:
          _stages_:
            process_disc:
              _optimizer_: optim_disc_stage_1
              _flows_:
                - horizontal_input_mapping
                - stage_1
                - generator_s1
                - stage_2
                - generator_s2
                - disc_prediscrimination
                - discriminator_s2
            process_sample:
              _optimizer_: optim_gen_stage_1
              _flows_:
                - horizontal_input_mapping
                - stage_1
                - generator_s1
                - stage_2
                - generator_s2
                - sample_prediscrimination
                - discriminator_s2
            process_reco:
              _optimizer_: optim_gen_stage_1
              _flows_:
                - horizontal_input_mapping
                - stage_1
                - stage_1_reco
                - generator_s1
                - stage_2
                - stage_2_reco
                - generator_s2
                - reco_prediscrimination
        _val_:
          _datasets_:
            repeated_npz:
              _flows_:
                - stage_1_val
                - generator_s1
                - stage_2_val
                - generator_s2
                - val_prediscrimination
                - val_prediscrimination_post
```

The YAML snippet depicts the flow for training the 2nd stage given that generator(s) `s1` are trained and frozen. Note that we use `_epoch_` to indicate the stage number (i.e., `_epoch_: 0` corresponds to the 1st stage).

### Logging & Visualization

The used loss functions and validation metrics are logged to our local [ClearML](https://clear.ml/) server, while the generated motion features are converted to a [Mixamo](https://www.mixamo.com/) skeleton representation and are visualilzed in the [Rerun viewer](https://rerun.io/).

```yaml {filename="util.yaml"}
engine:
  modules:
    clearml:
      project_name: GANimator
      task_name: ${experiment.name}
      tags: [train]
    rerun:
      annotations:
        parents:
          mixamo:
            [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 13, 16, 17, 18, 13, 20, 21, 22]
        labels:
          mixamo:
            - "Hips" # 0
            - "LeftUpLeg" # 1
            - "LeftLeg" # 2
            - "LeftFoot" # 3
            - "LeftToeBase" # 4
            - "LeftToe_End" # 5
            - "RightUpLeg" # 6
            - "RightLeg" # 7
            - "RightFoot" # 8
            - "RightToeBase" # 9
            - "RightToe_End" # 10
            - "Spine" # 11
            - "Spine1" # 12
            - "Spine2" # 13
            - "Neck" # 14
            - "Head" # 15
            - "LeftShoulder" # 16
            - "LeftArm" # 17
            - "LeftForeArm" # 18
            - "LeftHand" # 19
            - "RightShoulder" # 20
            - "RightArm" # 21
            - "RightForeArm" # 22
            - "RightHand" # 23
```

To visualize the skeleton all we need is the joint positions and the [Mixamo](https://www.mixamo.com/) kinematic tree information.

To train the GANimator model, one should run the following command:
```
python -m moai run fit conf/run/main.yaml npz_filename=%input_npz_file_name%
```