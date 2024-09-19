---
title: Getting Started
weight: 1
next: /docs/ganimator
prev: /docs
---

## Quick Start / `moai` Setup

🗿&nbsp;[moai](https://www.github.com/moverseai/moai) is a PyTorch-based AI Model Development Kit (MDK) that aims to improve data-driven model workflows, design and understanding. 


<!-- <img src="https://docs.github.com/assets/cb-77734/mw-1440/images/help/repository/use-this-template-button.webp" width="500">

[🌐 Demo ↗](https://moverseai.github.io/single-shot/) -->

#### Steps

{{% steps %}}

### Prerequisites
The following python packages are required for using the supported features and the underlying model development kit:

- {{< python-icon >}}&nbsp;Python 3.10
- {{< pytorch-icon >}}&nbsp;PyTorch 2.2.0 (cuda version)
- {{< nvidia-icon >}}&nbsp;Cuda 11.4

### Install `moai`
Clone the master branch from `moai` [repository](https://github.com/moverseai/moai/) and install it by opening a command line on the source directory and running:

```shell
pip install -e .
```

### Validate `moai` installation
To validate that the `moai` mdk has been successfully installed and there is no dependency missing, run the following `MNIST` example from the (outer) `moai` directory:

Train a simple multi-layer perceptron (MLP) using the `MNIST` data:

```shell
# train
python -m moai run fit moai/conf/examples/MNIST/main.yaml +DATA_ROOT=path/to/download/and/save/MNIST 
```

Test the trained by loading its checkpoint:
```shell
# train
python -m moai run test moai/conf/examples/MNIST/main.yaml +DATA_ROOT=path/to/download/and/save/MNIST +CKPT=path/to/trained/model/checkpoint
```

{{% /steps %}}

## Next
Note that `moai` operates as the backbone of the developed tools and plugins, which can be explored in the following sections:

{{< cards >}}
  {{< card link="../ganimator/" title="GANimator" icon="document-duplicate" >}}
  {{< card link="../rerun_animation/" title="Rerun Animation" icon="adjustments" >}}
{{< /cards >}}
