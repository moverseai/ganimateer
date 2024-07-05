---
title: Motion Generation
weight: 2
prev: /docs/ganimator/model_configuration
next: /docs/ganimator/_index
---

## Test & Inference

{{< filetree/container >}}
  {{< filetree/folder name="conf/ganimator" >}}
    {{< filetree/file name="data.yaml" >}}
    {{< filetree/file name="main.yaml" >}}
    {{< filetree/file name="model.yaml" >}}
    {{< filetree/file name="metrics.yaml" >}}
    {{< filetree/file name="util.yaml" >}}
  {{< /filetree/folder >}}
{{< /filetree/container >}}

### Metrics
Here we evaluate the trained GANimator using metrics from [GANimator](https://peizhuoli.github.io/ganimator/) and [SinMDM](https://sinmdm.github.io/SinMDM-page/). We measure the local and global `diversity` of the generated motions, as well as their quality in terms of plausibility (`fid`) and `coverage`:

```yaml {filename="metrics.yaml"}
# @package _global_

_moai_:
  _definitions_:
    _collections_:
      _metrics_:
        features:
          coverage:
            pred: [gen_feats]
            gt: [gt_feats]
            _out_: [coverage]
          gdiv:
            pred: [gen_feats]
            gt: [gt_feats]
            _out_: [ganimator_gdiv]
          ldiv:
            pred: [gen_feats]
            gt: [gt_feats]
            _out_: [ganimator_ldiv]
          mdm_gdiv:
            pred: [motion_embed]
            gt: [motion_embed_gt]
            _out_: [mdm_gdiv]
          mdm_ldiv:
            pred: [clips_embeds]
            gt: [clips_embeds_gt]
            _out_: [mdm_ldiv]
          fid:
            pred: [motion_embed]
            gt: [motion_embed_gt]
            _out_: [fid]
```

To run the GANimator evaluation use the following command:
```
python -m moai run test /path/to/ganimator/conf/run/main.yaml --config-dir ./conf npz_filename=%input_npz_file_name% model_ckpt=%path/to/trained/GANimator/checkpoint.ckpt% +mdm_ckpt=%path/to/SinMDM/t2m/text_mot_match/model/finest.tar%
```

### Export
The trained GANimator is able to generate variations of the learned motion, i.e., the same motion base but small variations in the high-level features, by sampling multiple codes from a Gaussian distribution. The generated motions can be exported in `.bvh` format using BVH exporter:

```yaml {filename="metrics.yaml"}
exporters:
  bvh:
    parents: [joint_parents]
    position: [ik.root_position]
    rotations: [euler]
    offsets: [joint_offsets]
    names:
      - 'Hips' # 0
      - 'LeftUpLeg' # 1
      - 'LeftLeg' # 2
      - 'LeftFoot' # 3
      - 'LeftToeBase' # 4
      - 'LeftToe_End' # 5
      - 'RightUpLeg' # 6
      - 'RightLeg' # 7
      - 'RightFoot' # 8
      - 'RightToeBase' # 9
      - 'RightToe_End' # 10
      - 'Spine' # 11
      - 'Spine1' # 12
      - 'Spine2' # 13
      - 'Neck' # 14
      - 'Head' # 15
      - 'LeftShoulder' # 16
      - 'LeftArm' # 17
      - 'LeftForeArm' # 18
      - 'LeftHand' # 19
      - 'RightShoulder' # 20
      - 'RightArm' # 21
      - 'RightForeArm' # 22
      - 'RightHand' # 23
```

To generate new motions use the following command:

```
python -m moai run test /path/to/ganimator/conf/run/main.yaml --config-dir ./conf model_ckpt=%path/to/trained/GANimator/checkpoint.ckpt% +mdm_ckpt=%path/to/SinMDM/t2m/text_mot_match/model/finest.tar% +out_name=%exported_file_name% +export_dir=%path/to/generated/motions%
```