# Fashion Encoder Training - User Documentation
---

This is a user documentation for a package designated for training a evaluating models based on Fashion Encoder architecture.

## Table of Contents
1. [Enviroment Setup](#environment-setup)
2. [Preparation of the Datasets](#prepare-datasets)
3. [Running Experiments](#running-experiments)


## Environment Setup

We tested this package using conda environment manager, so we recommend using it. However, you should be able to install the dependencies manually.

__Hardware requirements:__
To run the experiments, we recommend using GPU with CUDA support as the model contains a convolutional neural network. We tested the experiments on NVIDIA Tesla V100 16/32GB.

__Software requirements:__
When using conda, you don't need to install any aditional software such as CUDA SDK (conda takes care of this).

> For more information about using with conda GPU, see [this guide](https://docs.anaconda.com/anaconda/user-guide/tasks/gpu-packages/) 

### Conda installation:
1. Install conda using [this installation guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Run `conda env create --file environment.yml` in the project root to create our environment
3. Activate the environment using `conda activate outfit-recommendation`

### Requirements:
In case, you can't use conda, you will need to install these dependencies:

- Python 3.7
- Tensorflow >= 2.1 (preferably the GPU version)
- pillow
- scipy
- jupyter
- keras-tuner

---

## Prepare Datasets
Before running the experiment you will need to download and build the datasets.

### 1. Download the Datasets

#### Download Maryland Polyvore
1. Download Maryland Polyvore Dataset from this link __TODO__
2. Move the folder `maryland` into `data/raw`
3. Download the images from Maryland Polyvore dataset from this link [https://www.kaggle.com/dnepozitek/maryland-polyvore-images](https://www.kaggle.com/dnepozitek/maryland-polyvore-images)
4. Move the folder `images` into `data/raw/maryland`


#### Download Polyvore Outfits
1. Download Polyvore Outfits dataset from this link __TODO__
2. Move the whole folder `polyvore_outfits` into `data/raw/`


### 2. Build the TFRecord Datasets

We have prepaited scripts to build the datasets in the `bin` folder. You can execute the following commands to build the corresponding datasets:
- `bin/build_mp.sh`
- `bin/build_mp_images.sh`
- `bin/build_po.sh`
- `bin/build_po_images.sh`
- `bin/build_pod.sh`
- `bin/build_pod_images.sh`

Each script builds a training dataset, a validation FITB task and a test FITB task. The names have the following meaning: `mp` stands for Maryland Polyvore, `po` is Polyvore Outfits and `pod` is Polyvore Outfits Disjoint. The scripts with its names ending with `_images.sh` build the datasets with raw images, the other scripts extracts the visual features from the images using InceptionV3.

> Note that the building the dataset may take a few hours


---

## Running Experiments

> Make sure you have installed all the neccesary dependencies and that you have prepared the datasets before trying to run the experiments

### Training
In order to train and evaluate the model, you can use the `src.models.encoder.encoder_main` Python module.


### Debugging
We've found it handy to trace the 

### Hyperparameter Tuning
The hyperparameter tuning functionality is implemented in a module `src.models.encoder.param_tuning`. You can edit the `build` method to restrict the tuning to only some parameters or to modify the search space. As the file uses Keras Tuner in a straightforward way, we refer you to the official [Keras Tuner documentation](https://keras-team.github.io/keras-tuner/).

To execute the hyperparameter tuning, run `bin/hypertuning.sh`.

> We decided not to implement a CLI for the hyperparameter tuning because the Keras Tuner library provides a convenient way of setting up the tuning programmatically.


### Parameters

__`--mode {train,debug}`__
Type of action

__`--param-set PARAM_SET`__
Name of the hyperparameter set to use as base

__`--train-files TRAIN_FILES [TRAIN_FILES ...]`__
Paths to train dataset files

__`--valid-files VALID_FILES`__
Paths to validation dataset files

__`--test-files TEST_FILES`__
Paths to test dataset files

__`--batch-size BATCH_SIZE`__
Batch size

__`--filter-size FILTER_SIZE`__
Encoder filter size

__`--epoch-count EPOCH_COUNT`__
Number of epochs

__`--hidden-size HIDDEN_SIZE`__
Hidden size

__`--num-heads NUM_HEADS`__
Number of self-attention heads

__`--num-hidden-layers NUM_HIDDEN_LAYERS`__
Number of hidden layers (encoder blocks)

__`--checkpoint-dir CHECKPOINT_DIR`__
Path to a directory with checkpoints (resumes the training)

__`--with-weights WITH_WEIGHTS`__
Path to the directory with saved weights. 

__`--masking-mode {single-token,category-masking}`__
Mode of outfit masking

__`--valid-mode {fitb,masking}`__
Validation mode

__`--learning-rate LEARNING_RATE`__
Optimizer's learning rate

__`--valid-batch-size VALID_BATCH_SIZE`__
Batch size of validation dataset (by default the same as batch size)

__`--with-cnn [WITH_CNN]`__
Use CNN to extract features from images

__`--category-embedding [CATEGORY_EMBEDDING]`__
Merge learned category embedding to image feature vectors

__`--categories-count CATEGORIES_COUNT`__
Number of categories

__`--with-mask-category-embedding [WITH_MASK_CATEGORY_EMBEDDING]`__
Apply category embedding to the mask token

__`--target-gradient-from TARGET_GRADIENT_FROM`__
Value of valid accuracy, when gradient is let through target tensors, -1 for stopped gradient

__`--info INFO`__
Additional information about the configuration

__`--with-category-grouping [WITH_CATEGORY_GROUPING]`__
Categories are mapped into groups

__`--category-dim CATEGORY_DIM`__
Dimension of category embedding

__`--category-merge {add,multiply,concat}`__
Mode of category embedding merge

__`--use-mask-category [USE_MASK_CATEGORY]`__
Use true masked item category in FITB task

__`--category-file CATEGORY_FILE`__
Path to polyvore outfits categories

__`--categorywise-train [CATEGORYWISE_TRAIN]`__
Compute loss function only between items from the same category

__`--early-stop-patience EARLY_STOP_PATIENCE`__
Number of epochs to wait for improvement

__`--early-stop-delta EARLY_STOP_DELTA`__
Minimum change to qualify as improvement

__`--early-stop EARLY_STOP`__
Enable early stopping

__`--loss {cross,distance}`__
Loss function

__`--margin MARGIN`__
Margin of distance loss function

