# Binary Classification of Machine Failure

This repository contains a binary classification project focused on predicting machine failures using deep learning techniques. The project uses Keras with TensorFlow as a backend. The goal is to build a model that can effectively classify whether a machine is likely to fail based on provided data.

## Dataset
The project utilizes the dataset from a Kaggle competition. The dataset contains essential features for predicting machine failures. You can find the competition dataset at the following link:

https://www.kaggle.com/competitions/playground-series-s3e17

## Project Overview

The primary components of this project are as follows:

- **`code`**: This directory contains the main Python script, `dnn.py`, which executes the deep learning techniques for machine failure prediction. It also includes development notebooks (`dnn.ipynb`) for experimenting with new features and an EDA notebook (`eda.ipynb`) used for exploratory data analysis.

- **`data`**: This directory holds the training and test datasets used for training and evaluating the model. The datasets are stored in separate directories, `train_set` and `test_set`, respectively.

- **`HyperOptModelsHistory`**: This directory tracks the historical results of different models obtained through hyperparameter optimization using Keras Tuner. This information is useful for comparing model performance.

- **`environment.yml` and `requirements.txt`**: These files specify the required dependencies to recreate the project environment using Conda or pip.

## Getting Started

To replicate the environment and run the project, follow these steps:

1. Clone this repository to your local machine:

   ```
   git clone https://github.com/alessandroAbati/Binary-Classification-of-Machine-Failure
   cd Binary-Classification-of-Machine-Failure
   ```

2. Set up the environment using Conda or pip:

   - Using Conda:
     ```
     conda env create -f environment.yml
     conda activate <environment_name>
     ```

   - Using pip:
     ```
     pip install -r requirements.txt
     ```

3. Prepare the datasets:
   - Place your training dataset in the `data/train_set` directory.
   - Place your test dataset in the `data/test_set` directory.

   *Note: the dataset 

4. Run the main script for model training and evaluation:

   ```
   python code/dnn.py
   ```

The script performs data preprocessing, feature engineering, hyperparameter tuning, model training, and evaluation. It also generates ROC curves and displays the Area Under the Curve (AUC) score.

## Additional Information

- The `create_features` function in the script performs feature engineering to improve the model's predictive performance.

- The project uses a custom hyperparameter search strategy using Keras Tuner to find the best model configuration.

- The model is trained on the training dataset and evaluated on the validation dataset, and the best model is selected for final evaluation.

- The project handles class imbalance by applying class weights during model training.

- The `dnn.ipynb` notebook provides a flexible environment for further development and experimentation with new features or techniques.

## Results

The script generates an AUC score as a measure of the model's performance. The ROC curve is plotted to visualize the trade-off between true positive rate and false positive rate. Additionally, the script generates predictions on the validation dataset.

## Contributors

- [Your Name](https://github.com/yourusername)

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- Special thanks to the creators and maintainers of Keras, TensorFlow, and Keras Tuner for providing powerful tools for deep learning model development.

Feel free to customize this README to provide more details specific to your project or add any additional sections that may be relevant.
