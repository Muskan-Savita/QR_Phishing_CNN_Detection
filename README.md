QR Phishing CNN Detection

A small CNN that classifies QR code images as benign or malicious (quishing), plus an experiment on how well it holds up when the QR images are distorted (rotated, blurred, noisy, brightness-shifted).

Dataset
1,000 QR code images: 500 benign, 500 malicious (PNG).
Source: Mendeley Data, DOI 10.17632/cmhh7744sp.1.
This project was inspired by SefG25/QR_Phishing_CNN_Detection. The model, experiments and distortion code in this repo were written and run separately.
Split: 80% train / 20% validation (200 validation images), fixed seed 42.
Model

Simple Keras CNN, images resized to 128x128:

Rescaling(1/255) -> 3 x [Conv2D(3x3, ReLU) + MaxPooling2D] -> Flatten -> Dense(128, ReLU) -> Dropout -> Dense(1, sigmoid)

Optimizer: Adam. Loss: binary cross-entropy. Batch size: 32.

Experiment 1: Baseline and hyperparameters

I changed one thing at a time: number of epochs, conv filter sizes, and dropout rate.

Run	Filters	Dropout	Epochs	Val Acc (final)	Val Acc (best)
1	16-32-64	0.3	15	0.865	0.880
2	16-32-64	0.3	10	0.870	0.870
3	16-32-64	0.3	25	0.865	0.885
4	32-64-128	0.3	15	0.875	0.880
5	16-32-64	0.5	15	0.860	0.875
6	32-64-128	0.5	15	0.875	0.885

Observations

All runs land between 86% and 87.5%. The validation set has only 200 images, so a one-image difference is 0.5%, and these differences are small.
Larger filters (32-64-128) gave a slightly better final accuracy than 16-32-64.
More epochs did not help. In the baseline run, training accuracy climbed to about 97% while validation accuracy stayed around 85-88% and validation loss started rising after epoch 9, which is a sign of overfitting.
Dropout 0.5 made little difference.

The configuration used for the next experiment was Run 4 (32-64-128, dropout 0.3, 15 epochs), the first run to reach the top final validation accuracy.

Experiment 2: Clean vs distorted images

A distorted copy of the whole dataset was created (same file names, same split) using:

random rotation of up to 15 degrees (white fill)
Gaussian blur with radius 0.5 to 1.5
brightness scaling between 0.8 and 1.2
Gaussian noise (standard deviation 15)

First the clean-trained model was only evaluated on the distorted validation images (no retraining). Then a new model was trained on the distorted training images.

Model	Tested on	Accuracy
Clean-trained	Clean val	0.875
Clean-trained	Distorted val	0.680
Distorted-trained	Distorted val	0.810
Distorted-trained	Clean val	0.805

Show Image

Observations

The clean-trained model dropped from 87.5% to 68.0% on distorted images (about 19.5 percentage points).
Retraining on distorted data brought distorted accuracy back up to 81.0% (about +13 points).
The distorted-trained model was a bit worse on clean images (80.5% vs 87.5%), so there is a trade-off between robustness and clean accuracy.
Limitations
Small dataset (1,000 images) and a small validation set (200 images).
Each configuration was trained once, so small differences can be noise.
Distortions are synthetic and simple; real-world camera captures may behave differently.
Only image-based detection; no URL or metadata analysis.
Files
Lov_hack_QR_Project.ipynb: training and evaluation notebook (Google Colab)
qr_cnn_clean_model.keras: model trained on clean images
qr_cnn_distorted_model.keras: model trained on distorted images
experiment_results.csv: Experiment 1 results
clean_vs_distorted.csv, clean_vs_distorted.png: Experiment 2 results
Tech

Python, TensorFlow / Keras, NumPy, pandas, Pillow, Matplotlib, Google Colab.

