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
To run the experiments, we recommend using GPU with CUDA support as the model contains a convolutional neural network. We ran the experiments on NVIDIA Tesla V100 16/32GB.

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

