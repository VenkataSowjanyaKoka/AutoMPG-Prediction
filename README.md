# AutoMPG-Prediction
Linear Regression in PySpark

# Linear Regression algorithm to predict mpg given various other factors known to impact mpg rating.

# Steps followed and questions answered in the analysis

1.	What is the shape of the data contained in AutoMPG.csv?
2.	What features (or attributes) are recorded for each automobile?
3.	Provide a schema of the AutoMPG data set to verify that all relevant features contain numeric data type. Are there any columns/features that is not applicable in developing a Linear Regression algorithm? That is, does not meet the requirements/assumptions to use a Linear Regression model. If so, eliminate those columns from further analysis and regenerate the schema to ensure that the ‘offending’ column is removed from further analysis. Remember, it should not be permanently removed from the dataset. 
4.	Evaluate the correlation between mpg and each of the independent variables (pairwise mpg and cylinders, mpg and displacement, etc.). On the basis of individual correlation coefficients, can you determine which independent variables are useful in predicting mpg? Write response to your analysis as comments in your source code/notebook. Remember, Correlation coefficient value ranges from -1 to +1; closer to 1, stronger the relationship. 
5.	Provide a listing of summary descriptive statistics such as average and standard deviation for each relevant attribute. Round all float data to two decimal places.
6.	Collapse all independent variables/features into a single vector in preparation for Linear Regression analysis. Verify that the shape is now for 2, instead of 8 features. Print the first 5 rows of the new data.
7.	For training the regression model and its subsequent evaluation, generate the training data and test data from your refined AutoMPG dataset, the dataset will be split 80/20 so that 80% of the data will be used to train the model and the remaining 20% to evaluate the model. Provide a shape for each dataset.
8.	Using the training data, evaluate the correlation between mpg and each of the independent Use the training data to fit a regression model to predict mpg given values for the number of cylinders, displacement, hp,  weight, acceleration, model_year and origin. 
9.	For the trained model, what is the y-intercept value? And what are the coefficients for each of the independent variables in the fitted model. 
10.	For the trained model, print the Mean Sum of Squared Error and R-Square values. What does this tell you about the usefulness of the fitted model to predict mpg? Is the model any good? Remember, R-Square values range from 0 to 1.00; closer to 1.00 the better the predictive power of the model. 
11.	Now that you have trained your model, evaluate it using the test data. Using the values of R-Square and Mean sum of Squared Error, what can you say about the reliability of the trained model to predict mpg with test data. 

