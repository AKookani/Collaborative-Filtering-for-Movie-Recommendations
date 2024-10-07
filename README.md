# Collaborative-Filtering-for-Movie-Recommendations

This repository contains two different implementations of a movie recommendation system using neural networks:

1. **Dot Product-based Model (RecommenderNet)**: Uses user and movie embeddings, along with dot products and biases, to predict ratings.
2. **Neural Collaborative Filtering (NeuralCF)**: Uses concatenated embeddings and dense layers to model more complex interactions between users and movies.

## Features
- Both models are based on the MovieLens and Netflix datasets for training and evaluation.
- Each model learns latent representations of users and movies using embeddings.
- Predictions are made based on learned embeddings, providing personalized movie recommendations.
- Flexibility to choose between two different model architectures.

## Dependencies
- `pandas`
- `numpy`
- `matplotlib`
- `tensorflow`
- `keras`
- `pathlib`
- `zipfile`
- `sklearn`

## Installation
1. Install the required libraries:
    ```bash
    pip install pandas numpy matplotlib tensorflow keras pathlib zipfile scikit-learn
    ```
2. Clone or download this repository.

## Models Overview

### 1. Dot Product-based Model (`recommender_net.ipynb`)
This implementation is based on a simple dot product between user and movie embeddings, with additional bias terms for both users and movies. It uses the **MovieLens dataset** to train the model.

- **Features**:
  - Uses 50-dimensional embeddings for users and movies.
  - Computes the dot product of user and movie embeddings and applies sigmoid activation to scale predictions to the [0, 1] range.
  - Includes bias terms for users and movies.
  - Trained using **binary cross-entropy** loss.

- **Output**: 
    - Personalized movie recommendations for a user.
    - Plot of training and validation loss across epochs.

### 2. Neural Collaborative Filtering (NeuralCF) (`neural_cf.ipynb`)
This implementation uses a **Neural Collaborative Filtering (NCF)** approach, where user and movie embeddings are concatenated and passed through several fully connected (dense) layers to learn more complex interactions between users and movies. It uses the **Netflix Movie Rating dataset** for training.

- **Features**:
  - Customizable embedding dimensions and number of hidden layers.
  - Uses **ReLU** activation in hidden layers and **sigmoid** activation for the output layer.
  - Evaluates the model using **Mean Absolute Error (MAE)**.

- **Output**: 
    - Mean Absolute Error for the predicted ratings.
    - Trained model can generate predictions for user-movie pairs.

## Data

### 1. MovieLens Dataset (used in `recommender_net.ipynb`)
The dataset can be downloaded automatically by the script from the following link:
- [MovieLens Small Dataset (ml-latest-small.zip)](http://files.grouplens.org/datasets/movielens/ml-latest-small.zip)

### 2. Netflix Dataset (used in `neural_cf.ipynb`)
This dataset is used for the **NeuralCF** implementation and can be downloaded from:
- [Netflix Movie Rating Dataset (Kaggle)](https://www.kaggle.com/datasets/rishitjavia/netflix-movie-rating-dataset?select=Netflix_Dataset_Rating.csv)

## Example Output

### RecommenderNet (Dot Product-based Model)
```
Showing recommendations for user: 249
====================================
Movies with high ratings from user
--------------------------------
Fight Club (1999) : Action|Crime|Drama|Thriller
Serenity (2005) : Action|Adventure|Sci-Fi
...
--------------------------------
Top 10 movie recommendations
--------------------------------
In the Name of the Father (1993) : Drama
Monty Python and the Holy Grail (1975) : Adventure|Comedy|Fantasy
...
```

### NeuralCF (Collaborative Filtering Model)
```
Mean Absolute Error: 0.642
```

## Model Architecture

### 1. RecommenderNet
The model computes a dot product between user and movie embeddings and adds bias terms. The output is scaled between 0 and 1 using sigmoid activation. The architecture includes:
- User and movie embeddings (50-dimensional vectors).
- Bias terms for users and movies.
- Sigmoid activation for the final rating prediction.

### 2. NeuralCF
This model uses concatenated user and movie embeddings, followed by several dense layers to learn non-linear interactions. It includes:
- Customizable embedding dimensions (default is 10).
- Dense layers for learning complex interactions between user and movie embeddings.
- Sigmoid activation for output to predict ratings in the range [0, 1].

This README includes both the original `RecommenderNet` and the new `NeuralCF` model, providing instructions on how to use each, and describing the key differences in architecture and datasets.
