# ML-Final-Project
Code for my Machine Learning Final Project

Binary classification of images for person detection using HOG + SVM and Random Forests.

1. Dataset Download
    1. Go to the BDD100k website: http://bdd-data.berkeley.edu/download.html
    2. Download both the images(red) and the labels(green)
    ![Images in Red, Labels in Green](images_RM/bdd100k%20helper%20image.png)
    3. Extract the zip files in their respective data directory. If lost look for the hidden helper file.

2. Python Environment Setup
    1. Use your preferred method for environment setup

3. Run the Scripts
    1. The first script that should be run AFTER extracting the images and labels in the correct directory is the "flatten_dataset.py" script. This will move all the files to the parent directory and delete the leftover empty directories.
    2. Then run the "parse_labels.py" script to get the unique count of objects in the images.
    3. Run the "random_sample.py" script. If you want to change which object you classify by or the number of samples pooled from the total 100k images you may do so within this file.
    4. Run the "train_model.py" script to train the models and see results.
