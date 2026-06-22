# MetaNCA

Code for the paper **"Architecture Generalization with MetaNCA"** (ALIFE 2026).

MetaNCA learns a *local rule network* that self-organizes the weights of a *task network*
using only local interactions on the task network's computation graph. Once meta-trained, the
rule generates weights for diverse architectures — feedforward MLPs, CNNs, and ResNets — without
backpropagation through the target network, and generalizes to architectures unseen during
meta-training.

## Repository structure

- **`metanca/`** — core library: the `TaskNet` wrapper, parameter-neighborhood construction (via
  jaxpr tracing of the computation graph), the Weight Transformer local rule (linear attention),
  and the local weight-update step. See [`metanca/README.md`](metanca/README.md).
- **`metanca_training/`** — Hydra-based training harness: the meta-training loop, experiment
  configs, data loading, callbacks, and the conventional (Adam) baseline. See
  [`metanca_training/README.md`](metanca_training/README.md).

## Installation

Python 3.12 (3.13 also works). JAX is installed with CUDA support via `jax[cuda12]`.

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.in \
            -r metanca/requirements.in \
            -r metanca_training/requirements.in \
            -e ./metanca -e ./metanca_training
# Dataset loading uses tensorflow-datasets, which requires TensorFlow:
pip install tensorflow-cpu
```

## Meta-training MetaNCA

Hydra entry point: `metanca_training/scripts/train.py`. Config groups live in
`metanca_training/configs/`.

```bash
# Dense MLPs on MNIST
python metanca_training/scripts/train.py dataset=mnist_flat model=mnist wandb=disabled

# ResNets on CIFAR-100
python metanca_training/scripts/train.py dataset=cifar100 model=cifar100_resnet wandb=disabled
```

Useful config groups: `dataset=` (`mnist_flat`, `mnist_image`, `cifar100`, `iris`),
`model=` (`mnist`, `mnist_conv`, `cifar100_resnet`, …), `positional_encoding=`,
`local_rule=`, `training=`. See `metanca_training/configs/` for all options.

## Conventional (Adam) baselines

The same architectures can be trained conventionally for comparison:

```bash
python metanca_training/scripts/train_regular_net.py \
    dataset=cifar100 model=cifar100_resnet \
    training.num_epochs=400 training.regular_net_arch_selection=both wandb=disabled
```

## Architectures used in the paper

| Experiment | dataset | model config |
|------------|---------|--------------|
| Dense MLPs | `mnist_flat` | `mnist` |
| Convolutional nets | `mnist_image` | `mnist_conv_large` |
| ResNets | `cifar100` | `cifar100_resnet` |

> Note: `mnist_conv_large` reconstructs the paper's 3-layer convolutional networks (channel
> widths 32m/64m/128m with m=2; VALID padding, pooling on the last two layers; up to ~2M
> parameters). Training architectures use kernel sizes 3 and 5, with kernel size 4 held out.
> `mnist_conv` is a smaller two-layer variant.

## Tests

```bash
pytest metanca/ metanca_training/
```

## Citation

```bibtex
@inproceedings{barot2026metanca,
  title     = {Architecture Generalization with MetaNCA},
  author    = {Barot, Meet and Berenberg, Daniel and Khajehabdollahi, Sina},
  booktitle = {Proceedings of the Artificial Life Conference (ALIFE)},
  year      = {2026}
}
```

## License

Licensed under the Apache License, Version 2.0 — see [`LICENSE`](LICENSE).
Copyright © 2026 Meet Barot, Daniel Berenberg, and Sina Khajehabdollahi.

The paper itself is published under a Creative Commons Attribution 4.0
International (CC BY 4.0) license.
