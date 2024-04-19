# Movie-Recommendation-System-with-Machine-Learning-in-python
Machine Learning Project implementation using python for movie recommendation system
**Objective**
The objectives of the movie recommendation system project could include:

1. Enhanced User Experience: Provide personalized movie recommendations to users, improving their overall experience by suggesting movies tailored to their preferences.

2. Increase User Engagement: Encourage users to explore a wider range of movies by suggesting relevant options they may not have discovered otherwise, thus increasing their engagement with the platform.

3. Boost Revenue: By increasing user satisfaction and engagement, the platform can potentially increase revenue through increased subscriptions, rentals, or purchases.

4. Data Analysis and Insights: Gain insights into user preferences and behavior through data analysis of movie ratings, which can inform future content selection, marketing strategies, and platform improvements.

5. Machine Learning Implementation: Apply machine learning techniques such as collaborative filtering to develop a sophisticated recommendation system that continuously improves over time as more data is collected.

6. Scalability and Efficiency: Develop a recommendation system that can handle large volumes of data efficiently, ensuring scalability as the user base grows and the dataset expands.

7. Evaluation and Optimization: Continuously evaluate and optimize the recommendation algorithm to improve its accuracy and relevance, ensuring that users receive high-quality recommendations.

**Data Source**
For the  code, you can use a small synthetic dataset to demonstrate the functionality. Here's a simple example dataset you can use:

python
import numpy as np

# Sample movie ratings (rows: users, columns: movies)
ratings = np.array([
    [5, 4, 0, 0, 3, 0],
    [0, 0, 5, 4, 0, 0],
    [4, 0, 0, 0, 0, 2],
    [0, 0, 4, 5, 0, 0],
    [0, 0, 4, 0, 0, 5]
])


In this example:

- Each row represents a user.
- Each column represents a movie.
- The values represent the ratings given by users to movies. A rating of 0 indicates that the user has not rated the movie.

You can modify this dataset or replace it with a larger dataset from sources like MovieLens, IMDb, or Kaggle for more realistic recommendations. Make sure the dataset follows a similar structure, with rows representing users, columns representing movies, and values representing ratings.

**Imports Library**
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

**Import Data**
ratings = np.array([
    [5, 4, 0, 0, 3, 0],
    [0, 0, 5, 4, 0, 0],
    [4, 0, 0, 0, 0, 2],
    [0, 0, 4, 5, 0, 0],
    [0, 0, 4, 0, 0, 5]
])

**Describe Data**
print("Data Description:")
print("-" * 20)
print("Rows represent users, and columns represent movies.")
print("Ratings are on a scale from 0 to 5, where 0 indicates no rating given.")
print("-" * 20)

**Data Visualization**
plt.figure(figsize=(8, 6))
sns.heatmap(ratings, annot=True, cmap='YlGnBu', linewidths=0.5, linecolor='grey', cbar=False)
plt.xlabel('Movies')
plt.ylabel('Users')
plt.title('Movie Ratings Matrix')
plt.show()

**Data Preprocessing**
mean_ratings = np.mean(ratings, axis=0)
ratings_filled = ratings.copy()
ratings_filled[ratings_filled == 0] = mean_ratings

**Target Variable**
# Calculate similarity matrix
similarity_matrix = cosine_similarity(ratings_filled)

# Target variable: Movie ID for which recommendations are needed
target_movie_id = 0

**Feature Variable**
# Feature variables: Similarity scores with other movies
feature_variables = similarity_matrix[target_movie_id]

**Train Test Split**
ratings_train, ratings_test = train_test_split(ratings_filled, test_size=0.2, random_state=42)

**Modeling**
def get_movie_recommendations(movie_id, k=2):
    # Get similarity scores for the given movie
    movie_similarities = similarity_matrix_train[movie_id]
    # Get indices of most similar movies
    similar_movie_indices = np.argsort(movie_similarities)[::-1][1:k+1]
    # Return indices of recommended movies
    return similar_movie_indices

**Model Evaluation**
def evaluate_model(test_ratings, similarity_matrix, k=2):
    num_users, num_movies = test_ratings.shape
    total_recall = 0
    
    for user_id in range(num_users):
        for movie_id in range(num_movies):
            if test_ratings[user_id, movie_id] != 0:
                # Get indices of recommended movies
                recommended_movies = get_movie_recommendations(movie_id, similarity_matrix, k)
                # Check if the actual movie is in the recommended list
                if movie_id in recommended_movies:
                    total_recall += 1
    
    recall = total_recall / np.count_nonzero(test_ratings)
    return recall

# Evaluate model
recall_at_2 = evaluate_model(ratings_test, similarity_matrix_train, k=2)
print("Recall@2:", recall_at_2)

**Prediction**
target_movie_id = 0
recommendations = get_movie_recommendations(target_movie_id, similarity_matrix_train, k=2)
print("Recommended movies for movie {}: {}".format(target_movie_id, recommendations))

**Explanation**
 Let's divide the code expalnation into sections:

1. *Importing Libraries*: 
   - numpy, matplotlib.pyplot, and seaborn are imported for data manipulation, visualization, and plotting.
   - cosine_similarity from sklearn.metrics.pairwise is imported to calculate cosine similarity between items.
   - train_test_split from sklearn.model_selection is imported to split the data into training and testing sets.

2. *Data Description*:
   - A brief description of the dataset is printed, explaining that rows represent users, columns represent movies, and ratings range from 0 to 5.

3. *Visualizing the Ratings Matrix*:
   - The movie ratings matrix is visualized using a heatmap, displaying the ratings given by users to different movies.

4. *Preprocessing*:
   - Missing values (0s) in the ratings matrix are replaced with mean ratings to handle sparsity.

5. *Train-Test Split*:
   - The data is split into training and testing sets using a 80-20 split ratio. The training set will be used to calculate the similarity matrix.

6. *Calculating Similarity Matrix*:
   - Cosine similarity is calculated for the training set, resulting in a similarity matrix that represents the similarity between movies based on user ratings.

7. *Defining Functions*:
   - get_movie_recommendations: Takes a movie ID and the similarity matrix as input, and returns the indices of the most similar movies.
   - evaluate_model: Takes the test ratings, similarity matrix, and the value of k as input, and calculates the recall@k metric for the recommendation system.

8. *Model Evaluation*:
   - The evaluate_model function is called to compute the recall@2 metric for the recommendation system using the test set.

9. *Predictions*:
   - The get_movie_recommendations function is called to make movie recommendations for a given movie (here, movie ID 0) based on the calculated similarity matrix for the training set. These recommendations are printed to the console.

Overall, this code demonstrates building a simple movie recommendation system, including data preprocessing, model evaluation, and making predictions based on item similarity.
