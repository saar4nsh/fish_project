# Overview
The Detectron2.ipynb Jupyter notebook contains a complete pipeline for instance segmentation using the Detectron2 library, specifically for cell segmentation in FISH. It includes steps for environment setup, data preparation, dataset registration, model configuration for both GPU and CPU, and a training and evaluation loop.

## 1. Environment Setup
Download all the dependencies required as mentioned in the code cells.

## 2. Dataset Preparation
We assume a specific directory structure with color-coded masks. The script automates the process of copying and converting images from source folders into the required structure for Detectron2.

The expected dataset structure is as follows:
<dataset_dir>/
├── cell-only-images/
│ ├── image1.png
│ ├── image2.png
│ └── ...
└── cell-instance-masks/
├── image1.png
├── image2.png
└── ...

Source Images (Folder A): /path/to/cell-instance-masks
Source Masks (Folder B): /path/to/cell-only-images
Destination Masks (Folder C) [Path to be defined by user]
Destination Images (Folder D) [Path to be defined by user]

Testing Data Preparation
Source Images (Folder A): /path/to/cell-instance-masks-ground-truth
Source Masks (Folder B): /path/to/cell-only-images
Destination Masks (Folder C): [Path to be defined by user]
Destination Images (Folder D) [Path to be defined by user]  

## 3. How to Run

1. Ensure you have the required dependencies installed.
2. Place your images and masks into the required folder structure. You may need to modify the file paths in the notebook to match your local setup.
3. Run the cells to register your dataset. Verify if they have been imported correctly by running `visualize_dataset(dataset_dir)`
4. Choose either the GPU or CPU configuration function by running the appropriate setup_cfg function cell. 
5. Set the hyperparameters for training in `train_instance_segmentation(dataset_dir)`. Also write the path for your outputs in `output_dir` as defined in `setup_cfg`. 
6. Run the train_instance_segmentation function to start the training process.
7. The final model and predictions are saved to the `output_dir` as defined in `setup_cfg`. 
8. The code automatically runs the inference section to give validation results.
9. To test your model on a custom image, either use the `trained_cfg` or the saved model. 
10. To generate test set performance statistics, run the "Evaluation on Test Data" section.  (Note: Define the test dataset path.)