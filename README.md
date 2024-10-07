# Collaborative-Filtering-for-Movie-Recommendations

This project implements a movie recommendation system using the [MovieLens](https://grouplens.org/datasets/movielens/) dataset. It uses Keras and neural network embeddings to predict ratings for unseen movies, offering personalized movie recommendations for users.

## Features
- Loads and preprocesses the MovieLens dataset.
- Encodes users and movies as integer indices for embeddings.
- Trains a neural network model to predict ratings based on user and movie embeddings.
- Provides personalized movie recommendations based on predicted ratings.
- Plots training and validation loss over epochs to monitor model performance.

## Dependencies
- `pandas`
- `numpy`
- `matplotlib`
- `keras`
- `pathlib`
- `zipfile`

## Installation
1. Install the required libraries:
    ```bash
    pip install pandas numpy matplotlib keras pathlib
    ```
2. Clone or download this repository.

## Data
The MovieLens dataset is used for training and validation. It can be downloaded from the following link:

- [MovieLens Small Dataset (ml-latest-small.zip)](http://files.grouplens.org/datasets/movielens/ml-latest-small.zip)

The script automatically downloads and extracts this data, focusing on the `ratings.csv` and `movies.csv` files.

## How to Run
1. **Load and Preprocess Data**:
   The script downloads and preprocesses the MovieLens dataset by encoding users and movies as integer indices.

2. **Split Data for Training and Validation**:
   The dataset is split into training (90%) and validation (10%) sets, with the ratings normalized between 0 and 1.

3. **Build the Recommender Model**:
   A neural network model is constructed, embedding both users and movies into 50-dimensional vectors. The model computes a match score between user and movie embeddings using a dot product, then applies a sigmoid activation function to output a rating between 0 and 1.

4. **Train the Model**:
   The model is trained using binary cross-entropy loss and the Adam optimizer. The script displays both training and validation loss over epochs.

   ```bash
   python recommender.py
   ```

5. **Visualize Model Performance**:
   After training, the script plots the training and validation loss across epochs.

6. **Generate Movie Recommendations**:
   The script generates personalized movie recommendations for a randomly chosen user based on their predicted ratings for movies they haven't watched yet.

## Example Output

```
Showing recommendations for user: 249
====================================
Movies with high ratings from user
--------------------------------
Fight Club (1999) : Action|Crime|Drama|Thriller
Serenity (2005) : Action|Adventure|Sci-Fi
Departed, The (2006) : Crime|Drama|Thriller
Prisoners (2013) : Drama|Mystery|Thriller
Arrival (2016) : Sci-Fi
--------------------------------
Top 10 movie recommendations
--------------------------------
In the Name of the Father (1993) : Drama
Monty Python and the Holy Grail (1975) : Adventure|Comedy|Fantasy
Princess Bride, The (1987) : Action|Adventure|Comedy|Fantasy|Romance
Lawrence of Arabia (1962) : Adventure|Drama|War
Apocalypse Now (1979) : Action|Drama|War
Full Metal Jacket (1987) : Drama|War
Amadeus (1984) : Drama
Glory (1989) : Drama|War
Chinatown (1974) : Crime|Film-Noir|Mystery|Thriller
City of God (2002) : Action|Adventure|Crime|Drama|Thriller
```

## Model Architecture
The neural network model (`RecommenderNet`) consists of:
- User and movie embeddings (50-dimensional vectors).
- Bias terms for users and movies.
- A dot product of user and movie embeddings, followed by bias addition and a sigmoid activation.

The model is compiled with:
- **Loss**: Binary Cross-Entropy
- **Optimizer**: Adam
- **Learning Rate**: 0.001

## Results
After training, the model provides predictions for user ratings on unseen movies. Based on these predictions, top movie recommendations are generated for a given user.
