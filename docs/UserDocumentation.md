# Fashion Encoder - User Documentation
---

This is a user documentation for a package designated for training a evaluating models based on Fashion Encoder architecture.

## Table of Contents
1. [Enviroment Setup](#environment-setup)
2. [Preparation of the Datasets](#prepare-dataset)
3. [Running Experiments](#running-experiments)


## Environment Setup

We tested this package using conda environment manager, so we recommend using it. However, you should be able to install the dependencies manually.

__Hardware requirements:__
To run the experiments, we recommend using GPU with CUDA support as the model contains a convolutional neural network. We ran the experiments on NVIDIA Tesla V100 16/32GB

__Software requirements:__
When using conda, you don't need to install CUDA SDK (conda takes care of this).

> For more information about using with conda GPU, see [this guide](https://docs.anaconda.com/anaconda/user-guide/tasks/gpu-packages/) 

### Conda installation:
1. Install conda using [this installation guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Run `conda env create --file environment.yml` in the project root to create our environment
3. Activate the environment using `conda activate outfit-recommendation`

### Requirements:
In case, you can't use conda, you will need to install these dependencies:

- Python 3.7
- Tensorflow >= 2.1 (preferably GPU version)
- pillow
- scipy
- jupyter
- keras-tuner

---

## Prepare Datasets
Before running the experiment you will need to download and build the datasets.

### Download Maryland Polyvore
1. Download Maryland Polyvore Dataset from this link __TODO__
2. Move the folder `maryland_polyvore` into `data/raw`
3. Download the images from Maryland Polyvore dataset from this link [https://www.kaggle.com/dnepozitek/maryland-polyvore-images](https://www.kaggle.com/dnepozitek/maryland-polyvore-images)
4. Move the folder `images` into `data/raw/maryland_polyvore`


### Download Polyvore Outfits
1. Download Polyvore Outfits dataset from this link __TODO__
2. Move the whole folder `polyvore_outfits` into `data/raw/`


### Build the TFRecord Datasets
In order to run the experiments, it is first needed to build the TFRecord datasets form the raw files. 


 If you want to replicate the experiments done in our thesis, it is sufficient 

#### Build the Datasets with Features Extracted via CNN




#### Build the Training Datasets with Images (optional)
If you want to train the CNN together with the rest of the model you will need to build the datasets that contain images instead of extracted features.


## Running Experiments