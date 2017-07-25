# Machine Learning- Handwritten Digit Recognition- Kaggle Competition 

Project description

Technology: R programming
	    Python

Classifiers & Techniques used: 
1) Random forest
2) Bagging with SVM
3) Extreme Gradient Boosting 

About:
1) Applied strong Machine learning techniques to efficiently apply classification on the digit recognizer MNIST dataset consisting of hand-written numbers. 
2) The intent of the project was to compare the efficiency of different advanced ML techniques and determine the best ML classifier for this problem based on various performance metrics like Confusion Matrix, Cross Validation, Area under ROC, Precision and Recall.
3) Achieved high Test Accuracy of 96.6% (with Random Forest Classifier) on Kaggle Competition. 

Link to the competition: https://www.kaggle.com/c/digit-recognizer


General:
1) Please find the predictions of the test data in the folder Predictions.
2) Please find the ROC graphs in the graphs section.
3) Please find the confusion matrix in the Confusion Matrix folder.
4) Please find the results of xtreme boosting tuning experiments with different parameters in
   the folder "Xtreme boosting tuning results".
5) Please find the code for these experiments in the folder "code".
6) Please run R code in RStudio. If unable to 'source' the code.
7) Mean accuracy, precision, recall and AUROC for Bagging with SVM and Random Forests is printed on the output screen after running the respective program.

Packages Used:
1) Classifiers
  - scikit-learn
2) Ploting graphs
  - matplotlib
3) Array manipulations
  - numpy
4) Interpolation
  - scipy

To run the code go to the Code folder in the project and run the following commands.
<cpu_cont> argument is optional. It will be defaulted to one. If we want to use all cpus, pass -1 or the number of cpus used in parallel processing.

- Bagging-SVM Tuning:
python svm_tuning.py <cpu_count>
- Bagging-SVM Predictions:
python svm_classifier.py <cpu_count>
- Random Forest Tuning
python random_forest_tuning.py <cpu_count>
- Random Forest Predictions
python random_forest_classifier.py <cpu_count>

For xtremeboosting,
- R code for extreme gradient boosting for predicting the digits is in
  xtremeboosting.r
- R code for AOC, Confusion Matrix, Precision and Recall of extreme gradient boosting 
  using kfold cross validation is in xtreme_boosting_with_kfold.r

 
Regarding R codes:

Results:
- The digit predictions of extreme gradient boosting is in result_xgb-new1.csv in the predictions folder
- We have run 7 experiments on Extreme gradient boosting. The result of each experiment
  is in the folder "Results of Expts".
  - In each folder, you will find 
	-Accuracy of each fold in "accuracy.txt"
	-AOC of each fold in "aoc.txt"
	-Confusion Matrix, Precision and Recall of each fold in "confusion_matrix.txt"

Packages used:
1) Extreme Gradient Boosting
  -xgboost

2) AOC
  -pROC
  -ROC

3) Confusion Matrix
  -caret
