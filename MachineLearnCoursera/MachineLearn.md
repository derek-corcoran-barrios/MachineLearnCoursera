---
title: "Human Activity Recognition Unilateral Dumbbell Biceps Curl"
author: "Derek Corcoran"
date: "Monday, June 15, 2015"
output: pdf_document
---

# **Human Activity Recognition Unilateral Dumbbell Biceps Curl**

### Summary

The objective of this project is to predict weather a Dumbbell Biceps Curl was properly executed. In order to do that, a machine learning algorithm will be used to predict when the exercise  was properly done (*Class A*), or if they made a mistake (*Class B to E*), our Data set consists of the information given by sensors attached to the body and/or dumbbell while the exercise  was done, and it was classified by a human (Personal Trainer). We used a random forest algorithm to build the classificator and we ended up with a robust model with a 0.97 crossvalidation accuracy and a 0.98 out of sample accuracy. 

### Data manipulation and training



The first thing we do is to load the training data and divide it. Usually 60% of it would be used as a training set, and 40% as a test set, but since this is a test, we will use 15% as training and 85% as testing for the validation of the dataset. 



The full dataset had 19622 observation, of which 2946 were used to train the model and 16676 were used to test the model, also we have a dataset of 20 to which we don't know the answer to, which will be used later for further tests.

We start with 159 variables, but we remove timestamps and we also preprocess the data by removing the near zero variable ones, and also remove all the columns with more than 50% NA.



The number of variables ends up being 53. We used 2946 cases to train and crossvalidate the data using bootstraping, and 16676 to estimate out of sample error.

# Model

A Random Forest model was build, using 53 variables. The crossvalidation calculated accuracy is 0.9686, a Kappa value of 0.9603, and an error rate of 2.14. Below we see more details about the model as well as the confusion matrix. In figure 1 we see the variability of the accuracy within the resampling of the training dataset. Also in figure 2 we can see the importance for calssification of the 20 most important variables





```
## Random Forest 
## 
## 2946 samples
##   53 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 2946, 2946, 2946, 2946, 2946, 2946, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9515568  0.9386997  0.007031725  0.008848743
##   27    0.9686076  0.9602718  0.006679045  0.008428075
##   53    0.9616650  0.9514857  0.006857975  0.008650205
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

### Out of sample error calculation

In the out of sample error calculation, an accuracy of 0.98 was calculated (95% interval between 0.9777 to 0.982).
We also can see that in the out of sample classification  that the accuracy for each class is very high with a range from 0.9719 for class B to 0.9961 for class A.


```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4742   91    0    0    0
##          B    0 3066   54   20   17
##          C    0   63 2838   47    2
##          D    0    5   11 2665   16
##          E    1    2    5    1 3030
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9799         
##                  95% CI : (0.9777, 0.982)
##     No Information Rate : 0.2844         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9746         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9998   0.9501   0.9759   0.9751   0.9886
## Specificity            0.9924   0.9932   0.9919   0.9977   0.9993
## Pos Pred Value         0.9812   0.9712   0.9620   0.9881   0.9970
## Neg Pred Value         0.9999   0.9881   0.9949   0.9951   0.9974
## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2844   0.1839   0.1702   0.1598   0.1817
## Detection Prevalence   0.2898   0.1893   0.1769   0.1617   0.1822
## Balanced Accuracy      0.9961   0.9717   0.9839   0.9864   0.9940
```



With all these we can say that we have built a powerful predictor to tell wether a bicep curl has been done in a proper way, which could be succesfully implemented in weight training.

# APPENDIX (FIGURES)

![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-8-1.png) ![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-8-2.png) 
