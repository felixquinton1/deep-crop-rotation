# Deep Crop Rotation

While annual crop rotations play a crucial role for agricultural optimization, they have been largely ignored for automated crop type mapping .
We take advantage of the increasing quantity of annotated satellite data to propose the first deep learning approach modeling simultaneously the inter- and intra-annual agricultural dynamics of parcel classification. Along with simple training adjustments, our model provides an improvement of over 6.6 mIoU points over the current state-of-the-art of crop classification. Furthermore, we release the first large-scale multi-year agricultural dataset with over 300 000 annotated parcels.

## Requirements
 - PyTorch + Torchnet
 - Numpy + Pandas + Scipy + scikit-learn 
 - pickle
 - os
 - json
 - argparse
 
 The code was developed in python 3.7.7 with pytorch 1.8.1 and cuda 11.3 on a debian, ubuntu 20.04.3 environment.
 
## Downloads
 
### Multi-year Sentinel-2 dataset
We use the Multi-year Sentinel-2 dataset available here: *lien du dépo*

### Pre-trained models

Pre-trained models are available in the `model_saved` repository. 

## Code
This repository contains the scripts to train a multi-temporal scale PSE-LTAE model. 
The implementations of the PSE-LTAE can be found in `models`. 

Use the `train.py` script to train the 130k-parameter L-TAE based classifier with declaration with multi-year modeling (by default). 
You will only need to specify the path to the dataset folder:

`python train.py --dataset_folder path_to_multi_year_sentinel_2_dataset`

If you don't want to use temporal features, add: `--tempfeat false`

Choose the years used to train the model with: `--year` 

Use a pre-trained model with: `--test_mode true --loaded_model path_to_your_dataset`

## Use your own data

  If you want to train a model with your own data, you need to respect a specific architecture:
  - A main repository should contain two sub folders: `DATA` and `META` and a normalisation file.
  - META: contains the `labels.json` file containing the ground truth.
  - DATA: contains a sub folder by year containing a .pkl file by parcel.
Each parcel of the dataset should appears for each year with the same name in the DATA folder.
For each year you will need to precise the length of the temporal sequence with the option `--sly`.
