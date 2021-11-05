# Iris-Classification
This is a basic classification project for machine learning beginners to predict the species of a new iris flower.

- It is called a hello world program of machine learning and 
it’s a classification problem where we will predict the flower class based on its petal length, petal width, sepal length, and sepal width.

- First part of the code is for importing packages and dataset into the environment (Spyder). Then we get into the summarizing the dataset with Pandas method such as .info() and
.describe().

- In order to visualize the dataset and the correlation between its variables, we use seaborn library and its methods.

- After we understand the data, we split the dataset into two subsets as train and test.

- We did not know which algorithms would be the best for this classification problem thus we created a for loop with Algorithms and we found out Linear Discriminant Analysis (LDA)
was the best algorithm for this problem.

- Later on knowing the best algoritm for this problem is Linear Discriminant Analysis (LDA) we defined our LDA model. After model fittin/training, we predicted on X_test using .predict() method.

- Finally we checked the accuracy of the model using predicted values and y_test values.

As a result, test accuracy was 1.0 and i am afraid that maybe model did not learn but overfitted.
