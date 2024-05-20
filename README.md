
# Spatially and Spectrally Consistent MorphVQ

This repository contains the implementation of the Spatially and Spectrally Consistent MorphVQ model, designed for biological shape matching and automatic landmark aquisition.
This model leverages unique techniques to maintain consistency in spatial and spectral domains, enhancing the quality and reliability of the shape correspondences produced.

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.6 or higher
- pip (Python package installer)
- git (version control system)

## Installation

Follow these steps to get the environment set up:

### Clone the repository

First, clone the repository to your local machine using git:

```bash
git clone https://github.com/oothomas/Spatially_and_Spectrally_Consistent_MorphVQ.git
cd Spatially_and_Spectrally_Consistent_MorphVQ
```

### Install required packages

Install the necessary Python packages using `pip`. It's recommended to create a virtual environment before installation to avoid conflicts with existing packages. Note that in some platforms you may have to replace `python` with `python3`, or set an alias.

```bash
# Optional: Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install requirements
pip install -r requirements.txt
```
## Folder Descriptions

### data
This folder contains raw and preprocessed data necessary for training and evaluation.
It includes the 30A dataset for training the model, an evaluation dataset with 103 specimens, and some raw data for example preprocessing that is required to obtain manifold models.

### datasets
This folder contains pytorch dataset objects for creating datasets and data loaders for training and evaluation.

### diffusion_net
This folder contains implementation details or utilities related to DiffusionNet.

### Tools
This folder contains scripts and utilities that assist in various tasks such as data preprocessing and visualization.

## Usage

To train the model with the default parameters specified in the configuration files, run the training script from the command line:

```bash
python train.py --config config.yaml
```

When run for the first time, train.py with precompute all required quantities need for training. 
This might take some time and progress will be reported on the console.
First, the model will train on the 30A dataset found in the data directory. This model runs for 30 epochs, and the best model weights as saved along the way in the runs folder.
Training can be monitored via tensorboard by initiating it from the command line:

```bash
python tensorboard --logdir=runs
```

Once training is complete inference on the 103 specimen validation dataset will begin,
which will then be followed by an exporting of the MorphVQ landmarks estimated by the inference procedure.

## Configuration

The `config.yaml` file contains all model configurations. Here's a brief overview of some of the key parameters:

- `train_dataset_path`: Path to the training dataset.
- `validation_dataset_path`: Path to the validation dataset.
- `total_epochs`: Total number of training epochs.
- `learning_rate`: Starting learning rate for the training.

You can edit the `config.yaml` file to adjust training parameters such as batch size, learning rate, or the number of epochs depending on the GPU available to you.

## Contributing

Contributions to this project are welcome! To contribute, please fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License

## Acknowledgments

This work has been inspired and influenced by several pioneering research contributions in the field of shape analysis and machine learning on 3D surfaces. We acknowledge and appreciate the foundational works provided by:

- Nicholas Sharp, Souhaib Attaiki, Keenan Crane, and Maks Ovsjanikov (2020) for their insights into learning on 3D surfaces as described in their paper "DiffusionNet: Discretization Agnostic Learning on Surfaces." *arXiv preprint arXiv:2012.00888*. [DOI: 10.48550/arxiv.2012.00888](https://doi.org/10.48550/arxiv.2012.00888).

- Mingze Sun, Shiwei Mao, Puhua Jiang, Maks Ovsjanikov, and Ruqi Huang (2023) for their significant contributions to deep functional maps in "Spatially and Spectrally Consistent Deep Functional Maps." *arXiv preprint arXiv:2308.08871*. [DOI: 10.48550/arxiv.2308.08871](https://doi.org/10.48550/arxiv.2308.08871).

- Nicolas Donati, Etienne Corman, and Maks Ovsjanikov (2022) for their development of orientation-aware functional maps as detailed in "Deep Orientation-Aware Functional Maps: Tackling Symmetry Issues in Shape Matching." *arXiv preprint arXiv:2204.13453*. [DOI: 10.48550/arxiv.2204.13453](https://doi.org/10.48550/arxiv.2204.13453).

