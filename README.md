# Abnormality Detection in Stock Market with Unsupervised Learning

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
