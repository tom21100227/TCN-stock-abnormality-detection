# Abnormality Detection in Stock Market with Unsupervised Learning

This repo is the Final Project submission for Bowdoin College's CSCI 3465 Financial Machine Learning in Spring 2024 by Tom Han and Linguo Ren. For project description, please [refer to the poster attached in the repo.](./poster.pdf)

> 🪼Kaylie’s only friends🦦 : Tom Han and Linguo Ren

## File Structure

 - `eda/*`: Exploratory Data Analysis, including all the R code used for this project
 - `load_data.py`: used to extract data, courtesty of Professor Byrd. 
 - `tcnAutoencoder.py`: The actual implementation of TCN-Autoencoder in PyTorch.
 - `test_TCN.py`: Validation for if `tcnAutoencoder` is working.
 - `run_tcnAE.*`: Different scripts to train TCN-AE on HPC and our macbook.
 - `make_figures.ipynb`: used to generate some figures in the poster.
 - `api.py`: Getting data from various sources.
 - `clustering*`: Codes used to run the clustering algorithm.
 - `data`: A symlink that links to the `hft_data` location on Bowdoin's HPC, would not work locally.
 - `archive/*`: unused script that were a part of this project. 

## Getting started: 

1. Download the actual HFT data from lobster or create a working version of the symlink pointing to microwave.
2. Start with `run_tcnAE.ipynb`, it should load the data, preprocess and train the model for 1000 epoch by default
3. Get creative!

> Please note that when I run the script on my computer it took up 16 GB of memory. You've been warned. 
