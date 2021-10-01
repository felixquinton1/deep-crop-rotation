# Deep Crop Rotation

While annual crop rotations play a crucial role for agricultural optimization, they have been largely ignored for automated crop type mapping .
We take advantage of the increasing quantity of annotated satellite data to propose the first deep learning approach modeling simultaneously the inter- and intra-annual agricultural dynamics of parcel classification. Along with simple training adjustments, our model provides an improvement of over 6.6 mIoU points over the current state-of-the-art of crop classification. Furthermore, we release the first large-scale multi-year agricultural dataset with over 300 000 annotated parcels.

![](./gfx/teaser.png)

<p align="center">
  <img src="./gfx/declaration_en.jpg" alt="Sublime's custom image"/>
</p>
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
We use the Multi-year Sentinel-2 dataset available [here](https://zenodo.org/record/5535882). 



## Code
This repository contains the scripts to train a multi-temporal scale PSE-LTAE model. 
The implementations of the PSE-LTAE can be found in `models`. 

Use the `train.py` script to train the 130k-parameter L-TAE based classifier with 2 years declarations and multi-year modeling (2018, 2019 and 2020). 
You will only need to specify the path to the dataset folder:

`python3 train.py --dataset_folder path_to_multi_year_sentinel_2_dataset`

If you want to use a specific number of year for temporal features add: `--tempfeat number_of_year`

Choose the years used to train the model with: `--year` 

### Pre-trained models

Two pre-trained models are available in the `models_saved` repository: Mdec with 2 years temporal features and Mmixed without temporal features.

Use a pre-trained model with: `--test_mode true --loaded_model path_to_your_model --tempfeat number_of_years_used_to_train_the_model`

## Use your own data

If you want to train a model with your own data, you need to respect a specific architecture:
  - A main repository should contain two sub folders: `DATA` and `META` and a normalisation file.
  - META: contains the `labels.json` file containing the ground truth, `dates.json` containing each date of acquisition and `geomfeat.json` containing geometrical features (dates.json and geomfeat.json are optional).
  - DATA: contains a sub folder by year containing a .npy file by parcel.
  
Each parcel of the dataset should appears for each year with the same name in the DATA folder.
You must specify the number of acquisitions in the year that has the most acquisitions with the option `--lms length_of_the_sequence`.
You also need to add your own normalisation file in train.py 

## Credits 
 - The original PSE-LTAE model adapted for our purpose can be found [here](https://github.com/VSainteuf/lightweight-temporal-attention-pytorch)
 - This model was inspired by the works of Vaswani et al. on the Transformer. 
