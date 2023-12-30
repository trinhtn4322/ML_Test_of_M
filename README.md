# Deal with Multiple Feature of a Imbalance Dataset

## Introducing the topic
This is an entrance test for the Junior Data Scientist position of company M. I failed the previous test. This is the version I have completed after listening to feedback so it still has many unknown elements. complete, I posted it here for the purpose of sharing the methods I used and more than half that I want to use to apply to this position once again so **please see for reference only**

Here is the topic:
- Task 1: Train a model, evaluate it on your training data, and use it to predict labels for the test data
- Task 2: If you have to pick only 20 features, which features will you choose to have the best performance
- Task 3: Describe your approach and model performance. Visualize a few insights you have from working with the dataset that is useful for this prediction task

**Submission**

The submission including the following files:
- Your code in one file, code.py
- Your predictions in one file, predicted.csv. This will have three columns, index, predicted_label,
predicted_probability. predicted_label is either 0 or 1. predicted_probability is between 0 and 1 – the
probability you predict that the label is 1
- top_20_features.csv. Only 1 column and 20 rows, each row is a feature name from feat_1..634
- report.pdf. Concisely describe your approach and show the insights in the Task 3. Make it short and sweet,
at most 3 pages

## Introducing the data and the difficulties encountered
**This test is not public so the data will not be public either**
- The data set includes 491737 rows, 638 columns. Compared to the size of the test, this is considered a large dataset.
- ![image](https://github.com/trinhtn4322/ML_Test_of_M/assets/115331941/2d3510ad-ae9a-42ff-9e84-a18efee3db85)

- Data has some problems such as hidden feature columns, a very large number of features, and therefore a large number of NaNs.
- Target has 2 values 0 and 1, which is a problem about **Classification**
- The data is imbalanced towards 0, more than half of the feature columns also have many 0 values, making it difficult to find important features.

![image](https://github.com/trinhtn4322/ML_Test_of_M/assets/115331941/94366076-e871-4b9d-8a9b-ebfc111ca5f3)

# Based on the difficulties encountered, the problem will be divided into 2 parts to be solved

## Introduction to Features Selection
We will propose 3 solutions to solve the Multiple Feature problem including:
### Filtering method
As mentioned above, the data set contains many NaN and 0 values, making it difficult to select features
![image](https://github.com/trinhtn4322/ML_Test_of_M/assets/115331941/7a2447a4-81fa-4377-ba16-c0cf2d038f77)

Therefore, we use the filtering method to filter out columns containing large missing values (NaN>50%, 0Values>80%)
- After filtering NaN value, we have 573 features left
- After filtering 0 values, we have 341 features left
- Although the number of features has been reduced by nearly half, the number is still quite large
- However, before moving on to the next filtering method, we need to resolve missing values and handle data types other than numeric.
The next filtering method based on Sattistical is using Variance and Correlation
- For **Variance method**, we will use variance to be able to identify columns containing meaningless data from which we can retain 209 features.
- For the **Correlation method**, because this is a classification problem, we should not evaluate the correlation between the feature column and the target column, but instead we should evaluate based on the correlation between the features with each other, from from which 160 features can be selected
  
### Embedded Methods - LASSO
Lasso is a linear regression method, used to reduce model complexity based on evaluating the importance of features through the penaties method.
- After many tests, I chose the penalty level for the model: **0.0005**
- I have retained 53 features so I can continue using the 3rd filtering method

### Feature Permutation / Shuffling
As the name suggests, **Shufflin method** will build a sample model, then continuously rotate the features thereby determining which features will be most important to the model.
- The model used to run the sample here is Model RandomForest
![image](https://github.com/trinhtn4322/ML_Test_of_M/assets/115331941/f9107ae0-2b2a-42f9-81c1-e82ea1c43581)

Based on the chart, it is easy to see some features that do not affect the model's performance at all as well as predictions, for example: feat_60, feat_68, ...

## Introduction to the Imbalance resolution model

As introduced above, this data set is quite heavily Imbalanced at a level of 98% - 2%.
- There are a number of commonly used solution methods such as Resampling, which means they will increase/decrease the amount of data to bring it to a 1:1 ratio.
- However, with a data set with a ratio of 98-2 like this, you should not use the above method. Instead we will incorporate resampling into the prediction model. And to do this we need to use the **Ensemble Learning** model, in this problem I will use 2 models: Bagging and Boosting

### Bagging
I will build data for the bagging model using the **bootstrap sampling** method, but the difference is that I will actively take this data set by dividing label 0 into 48 different sets (because the difference ratio is 48 label_0 /1 label_1 ) then each set of data label_0 will be combined with label_1 and then used in the Bagging model.

To talk more about this model, I use the child model Decision Tree, because decision tree is a good model in handling imbalance data and is also a popular model in Bagging.

### Boosting - XGBoost

Different from proactively modeling and data like on Bagging, this time I will use the available library to **Resampling** as well as use the available Booting model **XGBoost**

## Evaluation model

A special point when evaluating models on imbalanced data sets is that you should not use Accuracy but instead use Confusion Matrix and values such as F1, Recall, Percission, etc.
- However, perhaps because my knowledge is still not enough, I am still lacking in handling features as well as building models, but the performance of the two models is really poor.
Here is the cf of XGBoost:
![image](https://github.com/trinhtn4322/ML_Test_of_M/assets/115331941/f9893e4c-e9b4-493d-ad57-7dd79d4933ec)

ROC Histogram:
![image](https://github.com/trinhtn4322/ML_Test_of_M/assets/115331941/f860ff95-58bc-4c0b-b306-01f2ce0c0b63)

As can be seen, the model predicts positive values quite poorly and still does not resolve imbalanced data

![image](https://github.com/trinhtn4322/ML_Test_of_M/assets/115331941/7325c462-39e0-469d-b583-ceafd284acbd)

**In the future I will definitely continue to upgrade this notebook to achieve the highest performance possible**

## Introduction to code
The structure of the code includes:

- Import Library
- EDA
- Train Test Split
- Delete Columns Have High Percent Null Values
- Delete Columns with High Percent 0 Values
- HandleCharacteric Value
- Handle Null Value
- Handle Outlier
- Select Feature by Variance
- Select Feature by Correlation
- Select Feature By Lasso
- Select Feature By Shuffle
- Processing X_test
- Standard Normalization
- Bagging Model
- XGBoost Model
- Evaluation 2 Models
- Optimization
