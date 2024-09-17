---
title: GANimator
weight: 2
prev: /docs/
next: /docs/ganimator/data_preprocess
sidebar:
  open: true
---

## Overview
Here we provide guidelines for training and evaluating the re-implementation of the [GANimator](https://peizhuoli.github.io/ganimator/) model in [moai](https://github.com/moverseai/moai), as well as for generating the motion features used as input to the model. The original implementation can be found [here](https://github.com/PeizhuoLi/ganimator).

## Commands
The documentation is split into 3 sections briefly discussing the configurations behind each of the following commands:

{{% steps %}}

### Data Preprocess

**Convert** the available `BVH` animation file into motion features:
```
python -m moai run test /path/to/ganimator/conf/bvh2npz/main.yaml --config-dir ./conf bvh_filename=%input_file_name% export_filename=%choose_a_name%
```

### Train GANimator

**Train** GANimator using the converted motion features:
```
python -m moai run fit /path/to/ganimator/conf/run/main.yaml --config-dir ./conf npz_filename=%input_npz_file_name%
```

### Evaluate GANimator

**Evaluate** GANimator on the predefined metrics:
```
python -m moai run test /path/to/ganimator/conf/run/main.yaml --config-dir ./conf npz_filename=%input_npz_file_name% model_ckpt=%path/to/trained/GANimator/checkpoint.ckpt% +mdm_ckpt=%path/to/SinMDM/t2m/text_mot_match/model/finest.tar%
```

Alternativelly, for generating new motions with a trained GANimator without the need for metrics, run the following command:
```
python -m moai run test /path/to/ganimator/conf/run/main.yaml --config-dir ./conf model_ckpt=%path/to/trained/GANimator/checkpoint.ckpt% +mdm_ckpt=%path/to/SinMDM/t2m/text_mot_match/model/finest.tar% +out_name=%exported_file_name% +export_dir=%path/to/generated/motions%
```

{{% /steps %}}

### Contents

For more details about each step please select the corresponding card below:

{{< cards >}}
  {{< card link="data_preprocess" title="Data Preprocessing" icon="adjustments" >}}
  {{< card link="model_configuration" title="Model Architecture" icon="adjustments" >}}
  {{< card link="motion_generation" title="Motion Generation" icon="adjustments" >}}
{{< /cards >}}
