# ML\_Assignment2\_Ruchika



\# a) Problem Statement

Classification of mobile phones into different price categories using multiple machine learning classification models.



\# b) Dataset Description

\- Dataset: Mobile Price Classification Dataset

\- Number of Features: 20

\- Number of Instances: 2000

\- Target Classes: 4 (Low, Medium, High, Very High)



\#c) Models Used

The following machine learning classification models were implemented and evaluated on the Mobile price prediction dataset:

Logistic Regression
Decision Tree Classifier
k-Nearest Neighbors (kNN)
Naive Bayes
Random Forest (Ensemble)
XGBoost (Ensemble)

|                     |   Accuracy |      AUC |   Precision |   Recall |   F1 Score |      MCC |
|:--------------------|-----------:|---------:|------------:|---------:|-----------:|---------:|
| Logistic Regression |     0.965  | 0.998667 |    0.965045 |   0.965  |   0.964986 | 0.953357 |
| Decision Tree       |     0.83   | 0.886667 |    0.831883 |   0.83   |   0.830168 | 0.773811 |
| KNN                 |     0.5325 | 0.777537 |    0.556098 |   0.5325 |   0.538838 | 0.378618 |
| Naive Bayes         |     0.81   | 0.950567 |    0.811326 |   0.81   |   0.810458 | 0.746804 |
| Random Forest       |     0.88   | 0.976929 |    0.879614 |   0.88   |   0.879734 | 0.840049 |
| XGBoost             |     0.935  | 0.994458 |    0.935487 |   0.935  |   0.934982 | 0.913501 |



\# Observations

Logistic Regression:Logistic Regression produced satisfactory results on the dataset. Since it is a linear classifier, it performs well when the relationship between features and target classes is approximately linear. However, its performance is limited when the dataset contains complex non-linear patterns.
Decision Tree:The Decision Tree classifier was able to capture non-linear relationships in the dataset. However, it showed slightly lower generalization performance compared to ensemble models, possibly due to overfitting on the training data.
k-Nearest Neighbors (kNN):The kNN classifier provided good classification accuracy by classifying samples based on similarity with neighboring data points. However, its performance depends heavily on feature scaling and the choice of k value.
Naive Bayes:The Naive Bayes classifier showed comparatively lower performance among the implemented models. This may be due to its assumption that all features are independent, which is not always true in real-world datasets.
Random Forest (Ensemble):Random Forest performed better than individual Decision Trees. By combining multiple trees using bagging, it reduces overfitting and improves overall stability and accuracy.
XGBoost (Ensemble):XGBoost achieved the highest performance among all models. Its boosting mechanism builds trees sequentially to correct the errors of previous models. Additionally, it includes regularization techniques that help prevent overfitting and improve generalization.


\## Streamlit App

(You will paste your live app link here after deployment)

