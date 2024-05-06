
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

Install the necessary Python packages using `pip`. It's recommended to create a virtual environment before installation to avoid conflicts with existing packages.

```bash
# Optional: Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install requirements
pip install -r requirements.txt
```

## Usage

To train the model with the default parameters specified in the configuration files, run the training script from the command line:

```bash
python train_dqfmnet.py --config config.yaml
```

You can edit the `config.yaml` file to adjust training parameters such as batch size, learning rate, or the number of epochs.

## Configuration

The `config.yaml` file contains all model configurations. Here's a brief overview of some of the key parameters:

- `train_dataset_path`: Path to the training dataset.
- `validation_dataset_path`: Path to the validation dataset.
- `total_epochs`: Total number of training epochs.
- `learning_rate`: Starting learning rate for the training.

## Contributing

Contributions to this project are welcome! To contribute, please fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License

## Acknowledgments
[]
