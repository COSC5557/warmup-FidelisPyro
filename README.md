[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11718863&assignment_repo_type=AssignmentRepo)
# Warmup

Download the [Wine Quality
dataset](https://archive-beta.ics.uci.edu/dataset/186/wine+quality). Choose the
one that corresponds to your preference in wine.

## Regression

Build a regression model to predict the wine quality. You can choose any model
type you like; the purpose of this exercise is to get you started. Evaluate the
performance of your trained model -- make sure to get an unbiased performance
estimate!

## Classification

Now predict the wine quality as a class, i.e. model the problem as a
classification problem. Evaluate the performance of your trained model again.

## Submission

Upload your code and a brief description of your results.

Using LinearRegression on the red wine dataset I got a MSE and R2:  
Mean Squared Error: 0.45000046789225695  
R2 Score: 0.3114056110795572  

I tried adding 5-fold cross validation:  
Cross Validation Scores: [0.20673025 0.26317093 0.22348896 0.24709515 0.38855928]  
Mean Cross Validation Score: 0.2658089139867161  
Mean Squared Error: 0.45000046789225684  
R2 Score: 0.31140561107955744  

Then upped it to 10-fold:  
Cross Validation Scores: [0.18830235 0.23618141 0.34940708 0.12757142 0.18243356  
0.27918933 0.30226084 0.18842341 0.44887346 0.34259338]  
Mean Cross Validation Score: 0.2645236243305624  
Mean Squared Error: 0.45000046789225684  
R2 Score: 0.31140561107955744  

There really hasn't been an increase in performance through any of these steps,  
so I tried using PolynomialFeatures to test different combinations of features.  
The results using PolynomialFeatures:  
Cross Validation Scores: [0.21722851 0.24226421 0.37958055 0.17356121 0.21936137 0.33266977  
 0.32750351 0.2829137  0.40932313 0.40189283]  
Mean Cross Validation Score: 0.2986298783898622  
Mean Squared Error: 0.4361822020067934  
R2 Score: 0.3325504343236504  

This showed improvement, though still very minor improvement.

