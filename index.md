# For structured data we use Spark SQL, SparkSession acts a pipeline between data and sql statements


```python
from pyspark.sql import SparkSession
```

# Sparksession is like a class and we need to create an instance of a class to utilize


```python
spark = SparkSession.builder.appName("AutoMPG_Linear_Regression").getOrCreate()
```
# Reading the CSV File as a dataframe
# if infer schema is given false it expects to give schema
# if header is given false it takes its own column names like c_0,c_1...

```python

AutoMPG_DF = spark.read.csv("/Users/sowjanyakoka/Desktop/Spring2020/MachineLearning/AutoMPG.csv", inferSchema = True, header = True)
```


```python
#Question(1).What is the shape of the data contained in AutoMPG.csv?
#(Answer):The shape of the data is the number of rows and columns (m = rows, n = columns) present in the dataset
print("Shape :",(AutoMPG_DF.count(),len(AutoMPG_DF.columns)))
```

    Shape : (392, 9)



```python
#Question(2).What features (or attributes) are recorded for each automobile?
#(Answer):The features recorded for each automobile can be known by the column names in the dataframe
AutoMPG_Features = AutoMPG_DF.columns
print("The features recorded for each automobile are :", AutoMPG_Features)
```

    The features recorded for each automobile are : ['mpg', 'cylinders', 'displacement', 'hp', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']



```python
#Question(3).Provide a schema of the AutoMPG data set to verify that all relevant features contain numeric data type. 
#(Answer):Schema is the outline of the dataframe which gives us an outline(column_name, datatype, possibility of null values) of each column in the dataset
print("The schema of the AutoMPG data set is:")
#Displaying the schema of the dataset
AutoMPG_DF.printSchema()
```

    The schema of the AutoMPG data set is:
    root
     |-- mpg: double (nullable = true)
     |-- cylinders: integer (nullable = true)
     |-- displacement: double (nullable = true)
     |-- hp: double (nullable = true)
     |-- weight: double (nullable = true)
     |-- acceleration: double (nullable = true)
     |-- model_year: integer (nullable = true)
     |-- origin: integer (nullable = true)
     |-- car_name: string (nullable = true)
    



```python
##Question(3).Are there any columns/features that is not applicable in developing a Linear Regression algorithm?That is, does not meet the requirements/assumptions to use a Linear Regression model.
#(Answer):Yes, the car_name column/feature is not applicable in developing a Linear Regression algorithm because it does not meet the following requirements of Linear Regression,
#-We use numeric input variables to predict a numeric output variable.(car_name is a categorical variable)
#-All data columns must be integer or float data type.(car_name is of string datatype) 
#===================================================================================================================
#Question(3).If so, eliminate those columns from further analysis and regenerate the schema to ensure that the ‘offending’ column is removed from further analysis. Remember, it should not be permanently removed from the dataset.
#(Answer):
AutoMPG_LinearRegression_DF = AutoMPG_DF['mpg','cylinders','displacement','hp','weight','acceleration','model_year','origin']
print("The schema of the AutoMPG_LinearRegression_DF data set is:")
#Displaying the schema of the dataset for further analysis
AutoMPG_LinearRegression_DF.printSchema()
```

    The schema of the AutoMPG_LinearRegression_DF data set is:
    root
     |-- mpg: double (nullable = true)
     |-- cylinders: integer (nullable = true)
     |-- displacement: double (nullable = true)
     |-- hp: double (nullable = true)
     |-- weight: double (nullable = true)
     |-- acceleration: double (nullable = true)
     |-- model_year: integer (nullable = true)
     |-- origin: integer (nullable = true)
    



```python
#Question(4).Evaluate the correlation between mpg and each of the independent variables (pairwise mpg and cylinders, mpg and displacement, etc.)
##(Answer):
#Loading the libraries for correlation function
from pyspark.sql.functions import corr
```


```python
##(Answer):4{1}-Performing correlation between mpg and cylinders
print("The correlation value between mpg and cylinders is:")
AutoMPG_LinearRegression_DF.select(corr('mpg','cylinders')).show()
```

    The correlation value between mpg and cylinders is:
    +--------------------+
    |corr(mpg, cylinders)|
    +--------------------+
    | -0.7776175081260227|
    +--------------------+
    



```python
##(Answer):4{2}-Performing correlation between mpg and displacement
print("The correlation value between mpg and displacement is:")
AutoMPG_LinearRegression_DF.select(corr('mpg','displacement')).show()
```

    The correlation value between mpg and displacement is:
    +-----------------------+
    |corr(mpg, displacement)|
    +-----------------------+
    |    -0.8051269467104577|
    +-----------------------+
    



```python
##(Answer):4{3}-Performing correlation between mpg and hp
print("The correlation value between mpg and hp is:")
AutoMPG_LinearRegression_DF.select(corr('mpg','hp')).show()
```

    The correlation value between mpg and hp is:
    +-------------------+
    |      corr(mpg, hp)|
    +-------------------+
    |-0.7784267838977761|
    +-------------------+
    



```python
##(Answer):4{4}-Performing correlation between mpg and weight
print("The correlation value between mpg and weight is:")
AutoMPG_LinearRegression_DF.select(corr('mpg','weight')).show()
```

    The correlation value between mpg and weight is:
    +------------------+
    | corr(mpg, weight)|
    +------------------+
    |-0.832244214831575|
    +------------------+
    



```python
##(Answer):4{5}-Performing correlation between mpg and acceleration
print("The correlation value between mpg and acceleration is:")
AutoMPG_LinearRegression_DF.select(corr('mpg','acceleration')).show()
```

    The correlation value between mpg and acceleration is:
    +-----------------------+
    |corr(mpg, acceleration)|
    +-----------------------+
    |    0.42332853690278693|
    +-----------------------+
    



```python
##(Answer):4{6}-Performing correlation between mpg and model_year
print("The correlation value between mpg and model_year is:")
AutoMPG_LinearRegression_DF.select(corr('mpg','model_year')).show()
```

    The correlation value between mpg and model_year is:
    +---------------------+
    |corr(mpg, model_year)|
    +---------------------+
    |   0.5805409660907859|
    +---------------------+
    



```python
##(Answer):4{7}-Performing correlation between mpg and origin
print("The correlation value between mpg and origin is:")
AutoMPG_LinearRegression_DF.select(corr('mpg','origin')).show()
```

    The correlation value between mpg and origin is:
    +------------------+
    | corr(mpg, origin)|
    +------------------+
    |0.5652087567164604|
    +------------------+
    



```python
##Question(4).On the basis of individual correlation coefficients, can you determine which independent variables are useful in predicting mpg?
#Write response to your analysis as comments in your source code/notebook.
#Remember, Correlation coefficient value ranges from -1 to +1; closer to 1, stronger the relationship.
#====================================================================================================================
##(Answer):
#Based on individual correlation coefficients we can observe that MPG is,
#-first it is highly correlated with weight(negative correlation):(-0.83)
#-second it is highly correlated with displacement(negative correlation):(-0.80)
#-third it is equally correlated with cylinders and hp(negative correlation):(-0.77)
#-moderately correlated with model_year and origin(positive correlation):(0.58,0.56)
#-lightly correlated with acceleration(positive correlation):(0.42)
#===================================================================================
##Question(4).Which independent variables are useful in predicting mpg?
##(Answer): Independent variables that are highly correlated with mpg i.e; weight, displacement, cylinders and hp are useful in predicting mpg
```


```python
#Question(5).Provide a listing of summary descriptive statistics such as average and standard deviation for each relevant attribute.
#===================================================================================================================================
#(Answer):To see the descriptive statistics of the data set
print("Descriptive Statistics of the data set :")
AutoMPG_LinearRegression_DF.describe().show(truncate = False)
```

    Descriptive Statistics of the data set :
    +-------+-----------------+------------------+------------------+------------------+------------------+------------------+-----------------+------------------+
    |summary|mpg              |cylinders         |displacement      |hp                |weight            |acceleration      |model_year       |origin            |
    +-------+-----------------+------------------+------------------+------------------+------------------+------------------+-----------------+------------------+
    |count  |392              |392               |392               |392               |392               |392               |392              |392               |
    |mean   |23.44591836734694|5.471938775510204 |194.41198979591837|104.46938775510205|2977.5841836734694|15.541326530612228|75.9795918367347 |1.5765306122448979|
    |stddev |7.805007486571802|1.7057832474527845|104.64400390890465|38.49115993282846 |849.4025600429486 |2.75886411918808  |3.683736543577868|0.8055181834183057|
    |min    |9.0              |3                 |68.0              |46.0              |1613.0            |8.0               |70               |1                 |
    |max    |46.6             |8                 |455.0             |230.0             |5140.0            |24.8              |82               |3                 |
    +-------+-----------------+------------------+------------------+------------------+------------------+------------------+-----------------+------------------+
    



```python
#Question(6).Collapse all independent variables/features into a single vector in preparation for Linear Regression analysis. 
#===========================================================================================================================
#(Answer):#Feature Engineering
#Loading vector libraries to combine all variables into one column to perform linear regression between mpg and independent variables
from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler
#Combining input columns(independent variables) as a vector and naming it as features
vec_assemble = VectorAssembler(inputCols = ['cylinders','displacement','hp','weight','acceleration','model_year','origin'], outputCol  = 'features')
#Taking data from dataset and combining 
Features_DF = vec_assemble.transform(AutoMPG_LinearRegression_DF)
#Extracting the features column and output column for model generation
Model_DF = Features_DF.select('features','mpg')
```


```python
Model_DF.show(20,False)
```

    +--------------------------------------+----+
    |features                              |mpg |
    +--------------------------------------+----+
    |[8.0,307.0,130.0,3504.0,12.0,70.0,1.0]|18.0|
    |[8.0,350.0,165.0,3693.0,11.5,70.0,1.0]|15.0|
    |[8.0,318.0,150.0,3436.0,11.0,70.0,1.0]|18.0|
    |[8.0,304.0,150.0,3433.0,12.0,70.0,1.0]|16.0|
    |[8.0,302.0,140.0,3449.0,10.5,70.0,1.0]|17.0|
    |[8.0,429.0,198.0,4341.0,10.0,70.0,1.0]|15.0|
    |[8.0,454.0,220.0,4354.0,9.0,70.0,1.0] |14.0|
    |[8.0,440.0,215.0,4312.0,8.5,70.0,1.0] |14.0|
    |[8.0,455.0,225.0,4425.0,10.0,70.0,1.0]|14.0|
    |[8.0,390.0,190.0,3850.0,8.5,70.0,1.0] |15.0|
    |[8.0,383.0,170.0,3563.0,10.0,70.0,1.0]|15.0|
    |[8.0,340.0,160.0,3609.0,8.0,70.0,1.0] |14.0|
    |[8.0,400.0,150.0,3761.0,9.5,70.0,1.0] |15.0|
    |[8.0,455.0,225.0,3086.0,10.0,70.0,1.0]|14.0|
    |[4.0,113.0,95.0,2372.0,15.0,70.0,3.0] |24.0|
    |[6.0,198.0,95.0,2833.0,15.5,70.0,1.0] |22.0|
    |[6.0,199.0,97.0,2774.0,15.5,70.0,1.0] |18.0|
    |[6.0,200.0,85.0,2587.0,16.0,70.0,1.0] |21.0|
    |[4.0,97.0,88.0,2130.0,14.5,70.0,3.0]  |27.0|
    |[4.0,97.0,46.0,1835.0,20.5,70.0,2.0]  |26.0|
    +--------------------------------------+----+
    only showing top 20 rows
    



```python
#Question(6).Verify that the shape is now for 2, instead of 8 features. 
#=======================================================================
#(Answer):The shape of the data is the number of rows and columns (m = rows, n = columns) present in the dataset
print("Shape :",(Model_DF.count(),len(Model_DF.columns)))
```

    Shape : (392, 2)



```python
##Question(6).Print the first 5 rows of the new data.
#====================================================
#(Answer):
Model_DF.show(5,False)
```

    +--------------------------------------+----+
    |features                              |mpg |
    +--------------------------------------+----+
    |[8.0,307.0,130.0,3504.0,12.0,70.0,1.0]|18.0|
    |[8.0,350.0,165.0,3693.0,11.5,70.0,1.0]|15.0|
    |[8.0,318.0,150.0,3436.0,11.0,70.0,1.0]|18.0|
    |[8.0,304.0,150.0,3433.0,12.0,70.0,1.0]|16.0|
    |[8.0,302.0,140.0,3449.0,10.5,70.0,1.0]|17.0|
    +--------------------------------------+----+
    only showing top 5 rows
    



```python
#Question(7).For training the regression model and its subsequent evaluation, 
#generate the training data and test data from your refined AutoMPG dataset, the dataset will be split 80/20 
#so that 80% of the data will be used to train the model and the remaining 20% to evaluate the model. 
#=================================================================================================================
#(Answer):
#Splitting the data for training and testing purpose
Train_DF, Test_DF = Model_DF.randomSplit([0.80,0.20])
```


```python
##Question(7).Provide a shape for each dataset.
#==============================================
##(Answer):
#Generating shape of training dataset
print("Shape of Training Dataset :" ,(Train_DF.count(), len(Train_DF.columns)))
```

    Shape of Training Dataset : (314, 2)



```python
##Question(7).Provide a shape for each dataset.
#==============================================
#(Answer):
#Generating shape of testing dataset
print("Shape of Testing Dataset :" ,(Test_DF.count(), len(Test_DF.columns)))
```

    Shape of Testing Dataset : (78, 2)



```python
#Question(8).Using the training data, evaluate the correlation between mpg and each of the independent 
#Use the training data to fit a regression model to predict mpg given values for the number of cylinders, displacement, hp,  weight, acceleration, model_year and origin.
#========================================================================================================================================================================
#(Answer):
#Using the ml(machine learning) import the required model using sub library regression
from pyspark.ml.regression import LinearRegression
#Defining the model to understand which column is being dependent 
#Independent variables data is taken in the fit
Linear_Model = LinearRegression(labelCol = 'mpg').fit(Train_DF)
```


```python
#Question(9).For the trained model, what is the y-intercept value? 
#==================================================================
#(Answer):
#To see the intercept of the equation
print("Intercept of the equation: ")
print(Linear_Model.intercept)
```

    Intercept of the equation: 
    -17.761877657834987



```python
##Question(9).And what are the coefficients for each of the independent variables in the fitted model. 
#=====================================================================================================
#(Answer):
#To see the coefficients of each independent variable
print("Coefficients of each independent variable:")
print(Linear_Model.coefficients)
```

    Coefficients of each independent variable:
    [-0.38406428255042524,0.02175257282762645,-0.005660947872021063,-0.007347303126901514,0.20137544057335854,0.7441960604959144,1.318677872653346]



```python
#Question(10).For the trained model, print the Mean Sum of Squared Error and R-Square values. 
#============================================================================================
#(Answer):
#To see the predictions using the model for the input variables 
Training_Predictions = Linear_Model.evaluate(Train_DF)
#Checking the Mean Square Error(Closer to zero the better)
print("The Mean Square Error values for Training Data is:")
print(Training_Predictions.meanSquaredError)
#Train R square value
print("The R-Square value for Training Data is:")
print(Training_Predictions.r2)
```

    The Mean Square Error values for Training Data is:
    11.05561225608178
    The R-Square value for Training Data is:
    0.8250766410806196



```python
##Question(10)What does this tell you about the usefulness of the fitted model to predict mpg? 
#Is the model any good? Remember, R-Square values range from 0 to 1.00; closer to 1.00 the better the predictive power of the model.
#===================================================================================================================================
#(Answer): In case of R2(coefficient of determination) it indicates a measure of how close the data are to the fitted regression line. 
#R-Square = Explained variation / Total variation
#0 indicates that the model explains none of the variability of the response data around its mean.
#1 indicates that the model explains all the variability of the response data around its mean.
#So, an R-square value of 0.82 means that the fit explains 82% of the total variation in the data about the average.
#Whereas (MSE) is the average of square of the errors. The larger the number the larger the error, closer to zero the better
#We can say the model is good because,
            #The lower the value of MSE the better and 0 means the model is perfect, our MSE is 10.918
            #In case of r2 the closer the value is to 1 the high is the predictive power, our r2 is 0.816
```


```python
#Question(11).Now that you have trained your model, evaluate it using the test data. 
#Using the values of R-Square and Mean sum of Squared Error, what can you say about the reliability of the trained model to predict mpg with test data.
#======================================================================================================================================================
#(Answer): 
Test_Predictions = Linear_Model.evaluate(Test_DF)
#Test R square values
print("The R-Square value for Test Data is:")
print(Test_Predictions.r2)
#Checking the Mean Square Error(Closer to zero the better)
print("The Mean Square Error values for Test Data is:")
print(Test_Predictions.meanSquaredError)
##This r2 percentage indicates that our Linear Regression model can predict
#with more than 81% of accuracy in terms of predicting the mpg given the independent features of the Automobile. 
#The other 19% can be attributed toward errors that cannot be explained by the model. 
#Our Linear Regression line fits the model really well which indicates the model is good and reliable
# We can say the model is quite reliable as the R2 value and Mean sum of Squared Error values are between 80-85% and 10-20 respectively 
#even after training the model with 70% train data and 30% test data for more than 10 times
#and as we know lower the value of MSE and R2 value closer to 1, the better the model is and also reliable
```

    The R-Square value for Test Data is:
    0.7904488856876198
    The Mean Square Error values for Test Data is:
    10.581780529461945



```python
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
```


<style>.container { width:100% !important; }</style>



```python

```


