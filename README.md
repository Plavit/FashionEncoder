# FashionEncoder

Outfit recommendation model base on the Transformer's encoder.

## Docs
- [User Documentation](docs/UserDocumentation.md)

## Data Sources
- [Maryland Polyvore](https://github.com/xthan/polyvore-dataset) 
- [Polyvore Outfits](https://github.com/mvasil/fashion-compatibility)

### Requirements
For seamless experience we recommend to use `enviroment.yml` to create a conda environment. Or make sure that you have the following dependencies installed:
- Python 3.7
- Tensorflow >= 2.1
- pillow
- scipy
- jupyter
- keras-tuner

## Project Organization


    ├── README.md          <- The top-level README for developers using this project.
    |
    ├── bin                <- Executable scripts
    |
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling
    │   └── raw            <- The original, immutable data dump
    │
    ├── docs               <- Project documentation and reports
    │   └── report.md      <- Project report
    │
    ├── models             <- Trained and serialized models
    |
    └── src          <- Source code for use in this project
        ├── data     <- Scripts to process data
        │   ├── build_dataset.py    <- Builds Maryland Polyvore training dataset
        |   ├── build_fitb.py       <- Builds Maryland Polyvore FITB dataset
        │   ├── build_po_dataset.py <- Builds Polyvore Outfits training dataset
        |   ├── build_po_fitb.py    <- Builds Polyvore Outfits FITB dataset
        │   └── input_pipeline.py   <- Provides input pipelines
        │
        ├── models          <- Model definition and code required for training
        │   └── encoder     <- Fashion Encoder model
        │       ├── encoder_main.py     <- Training
        │       ├── fashion_encoder.py  <- Definition of the model
        │       ├── layers.py           <- Custom layers used in the model
        │       ├── metrics.py          <- Loss functions and metrics
        │       ├── param_tuning.py     <- Hyperparameter tuning
        │       ├── params.py           <- Hyperparameter sets
        │       └── utils.py            <- Helper methods
        │
        └── notebooks  <- Jupyter Notebooks with experiments and data exploration
            ├── df_attr_comparison.py <- Visualization of nearest negihbors with their attributes
            ├── df_exploration.py     <- Deep Fashion dataset statistical exploration
            ├── df_neighbors.py       <- Model validation
            ├── n_neighbors.py        <- sk-learn n-neighbors wrapper
            └── polyvore_neighbors.py <- Nearest neighbors visualization of Polyvore dataset embedding