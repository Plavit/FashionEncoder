# Fashion Encoder - User Documentation
---




## Environment Setup

We tested this package using conda environment manager, so we recommend using it. However, you can install the dependencies manually.

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

### Download
1. Download Maryland Polyvore Dataset from this link [https://github.com/xthan/polyvore-dataset](https://github.com/xthan/polyvore-dataset)