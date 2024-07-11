---
title: Data Preprocessing
weight: 2
prev: /docs/ganimator/_index
next: /docs/ganimator/model_configuration
---

## Prepare GANimator Input

{{< filetree/container >}}
  {{< filetree/folder name="conf/bvh2npz" >}}
    {{< filetree/file name="data.yaml" >}}
    {{< filetree/file name="main.yaml" >}}
    {{< filetree/file name="flow.yaml" >}}
    {{< filetree/file name="monitoring.yaml" >}}
  {{< /filetree/folder >}}
{{< /filetree/container >}}

### Load BVH

```yaml {filename="main.yaml"}
- data/test/loader: torch
- src/data/test: bvh
```

The BVH loader is used with 1 parameters that needs to be defined externally - the `.bvh` filename. The `bvh_subset` corresponds to the used subset of the [Mixamo](https://www.mixamo.com/) kinematic tree. 

```yaml {filename="data.yaml"}
# @package _global_

bvh_filename: ???
bvh_subset:
  [
    0,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    1,
    2,
    3,
    4,
    5,
    7,
    8,
    9,
    10,
    31,
    32,
    33,
    34,
  ]
bvh_scale: 0.01
bvh_fps: 30

data:
  test:
    loader:
      shuffle: false
      drop_last: false
    iterator:
      datasets:
        bvh:
          filename: ${bvh_filename}
          subset: ${bvh_subset}
          scale: ${bvh_scale}
          fps: ${bvh_fps}
```

### Motion Features

The desired representation consists of T × (JQ+C+3) motion features where:

- T is the number of frames
- J is the number of joints
- Q corresponds to the 6D rotation representation
- C corresponds to the foot contact labels
- The last 3 features are the root joint x- and z-axis velocity and the y-axis position

The monads below help us transform the BVH information into the desired features (i.e., 6D rotations, joints velocities, etc.):

```yaml {filename="flow.yaml"}
deg2rad:
  degrees: [joint_rotations]
  _out_: [joint_rotations_rads]
euler_to_rotmat:
  euler: [joint_rotations_rads]
  _out_: [joint_rotation_matrices]
forward_kinematics:
  parents: [joint_parents]
  offsets: [joint_offsets]
  rotation: [joint_rotation_matrices]
  position: [root_position]
  _out_: [fk]
simple_velocity:
  positions: [fk.positions]
  _out_: [joint_velocities]
foot_contact:
  velocity: [joint_velocities]
  _out_: [contact_labels]
roma_rotmat_to_sixd:
  matrix: [joint_rotation_matrices]
  _out_: [joint_rotations_sixd]
root_features:
  position: [root_position]
  _out_: [root_features]
```

By concatenating the individual features described above, we end up with the `motion data` representation that will be used as input to the model:

```yaml {filename="flow.yaml"}
_mi_:
  expression:
    - ${mi:"cat(joint_rotations_sixd_flat, contact_labels_flat, root_features, zero_position, -1)"}
  _out_:
    - motion_data
```

The next step is to prepare the downsampled versions of the motion features for each GANimator stage. To do so, we employ a monad for preparing 6 pyramid levels:

```yaml {filename="flow.yaml"}
get_pyramid_lengths:
  tensor:
    - ${mi:"transpose(motion_data, -2, -1)"}
  _out_: [motion_data_pyramid]
_mi_alias:
  expression:
    - ${mi:"transpose(motion_data, -2, -1)"}
    - motion_data_pyramid.level_6
    - motion_data_pyramid.level_5
    - motion_data_pyramid.level_4
    - motion_data_pyramid.level_3
    - motion_data_pyramid.level_2
    - motion_data_pyramid.level_1
  _out_:
    - motion_data_level_6
    - motion_data_level_5
    - motion_data_level_4
    - motion_data_level_3
    - motion_data_level_2
    - motion_data_level_1
    - motion_data_level_0
```

which are later renamed for practical reasons.

The last two steps for completing the data preprocessing are the computation of the amplitudes for the pyramid levels and the z*.

The amplitudes are realized as the mean squared error between the original motion features and their downsampled versions and their reconstructions after upsampling:

```yaml {filename="flow.yaml"}
noise_scale:
  target:
    - motion_data_level_6
    - motion_data_level_5
    - motion_data_level_4
    - motion_data_level_3
    - motion_data_level_2
    - motion_data_level_1
    - motion_data_level_0
  reconstructed:
    - motion_data_level_6_recon
    - motion_data_level_5_recon
    - motion_data_level_4_recon
    - motion_data_level_3_recon
    - motion_data_level_2_recon
    - motion_data_level_1_recon
    - ${mi:"zeros(motion_data_level_0)"}
  _out_:
    - amps_level_6
    - amps_level_5
    - amps_level_4
    - amps_level_3
    - amps_level_2
    - amps_level_1
    - amps_level_0
```

while for the z* we sample a normal distribution and for a tensor with the same size as the lowest level of the pyramid:

```yaml {filename="flow.yaml"}
random_like:
  tensor: [motion_data_pyramid.level_1]
  _out_: [z_star_level_0]
```

Now we are ready to export the prepared representations as an `.npz` file using our NPZ exporter:

```yaml {filename="monitoring.yaml"}
export_data:
  append_npz:
    path:
      - ${export_filename}
    keys:
      - - z_star_level_0
        - amps_level_6
        - amps_level_5
        - amps_level_4
        - amps_level_3
        - amps_level_2
        - amps_level_1
        - amps_level_0
        - motion_data_level_6
        - motion_data_level_5
        - motion_data_level_4
        - motion_data_level_3
        - motion_data_level_2
        - motion_data_level_1
        - motion_data_level_0
        - contact_labels_raw
        - joint_rotation_matrices
        - root_position
        - joint_offsets
        - joint_parents
    combined:
      - true
    compressed:
      - true
```

The steps above are executed by running the following command:
```
python -m moai run test /path/to/ganimator/conf/bvh2npz/main.yaml --config-dir ./conf bvh_filename=%input_file_name% export_filename=%choose_a_name%
```