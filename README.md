# Predictive Modeling and Analysis of Firm’s fast Growth
This report delves into the predictive analysis of firm growth, categorizing firms into 'fast-growing'
and 'non-fast growing' using various statistical and machine learning techniques. The dataset comprises
121 variables across 57,085 observations, encompassing the financial and operational metrics of firms.
Variables were categorized into distinct groups like raw financial variables, quality indicators,
engineered features, and human resource-related factors. Additionally, dummy variables were
generated for categorical features. Several models were developed and evaluated, primarily focusing
on Logistic Regression, LASSO Regression, Random Forest, and Classification. Moreover, I decided
to calculate EBTDA to identify fast-growing companies based on whether their 'ebtda_to_assets'
values rank in the top 30% within their specific 'ind2_cat' and 'year' groups.
  • Logistic Regression and Logistic Regression with LASSO:
  • Models (M1 to M5) were built using different subsets of predictors.
  • Logistic Regression with cross-validation (LogisticRegressionCV) was used to
  optimize model parameters.
  • LASSO regression was applied for feature selection, exploring a range of lambda
  values.
  • Random Forest:
  • Parameters like max_features and min_samples_split were tuned using GridSearchCV.
  • The model's performance was evaluated using metrics like accuracy, AUC (Area Under
  Curve), and RMSE (Root Mean Square Error).
  • Among these models, the Random Forest classifier demonstrated superior performance
with the highest AUC and lowest RMSE, indicating its effectiveness in balancing false
positives and negatives and predicting firm growth accurately.
In the results section of the predictive modelling analysis, a summary table has been created to compare
the performance of different models, including Logistic Regression variants (M1 to M5), LASSO, and
Random Forest. This summary focuses on the number of predictors used in each model, along with
their Cross-Validation Root Mean Squared Error and Area Under Curve metrics. The Random Forest
model ('randomforestprob'), utilizing 44 predictors, exhibits the most promising results with the lowest
CV RMSE of 0.403 and the highest CV AUC of 0.788. This indicates a strong predictive accuracy and
the model's ability to distinguish between the different classes effectively. In contrast, the Logistic
Regression models (M1 to M5) show varying levels of performance. Model M1, with the fewest
predictors (9), has a CV RMSE of 0.456 and a CV AUC of 0.621, while Model M4, using a
considerably higher number of predictors (74), achieves a CV RMSE of 0.420 and a CV AUC of 0.743.
Notably, the LASSO model, with 129 predictors, balances complexity and performance effectively,
evidenced by a CV RMSE of 0.415 and a CV AUC of 0.758. These results suggest that the Random
Forest model outperforms the Logistic Regression variants in this analysis, achieving higher predictive
accuracy with a relatively moderate number of predictors. The LASSO model also demonstrates
significant effectiveness, providing a good balance between the number of predictors used and the
model's predictive power.
## Confusion Table and Discussion
The effectiveness of the models was further dissected by analyzing the confusion matrix, which
provided insights into the true positives, true negatives, false positives, and false negatives. This
analysis was pivotal in understanding the practical implications of model predictions, especially in
weighing the costs associated with incorrect predictions. The dataset was segmented into
manufacturing and service firms to understand sector-specific dynamics. Distinct Random Forest
models were developed for each sector. Key findings include:
  • Manufacturing firms had a distinct pattern in terms of the significance of predictors and
performance indicators in the model. The sector-specific model offered valuable insights into
the unique characteristics that drive rapid growth in the manufacturing industry, which differ
from those affecting the service sector.
  • Service firms have a model that identifies various major predictors of growth. The service
sector model exhibited distinct variations in performance compared to the manufacturing
model, as indicated by the differences in RMSE and AUC. This highlights the importance of
doing a sector-specific study while engaging in predictive modelling.
The division into two distinct parts offered a valuable understanding of the contrasting behaviours of
these industries and emphasized the significance of customized models for accurate prediction
analysis.
## Analysis of Loss Function Implementation in Predictive Modeling
The analysis code incorporates a loss function to assess several forecasting models for the expansion
of a company. The main objective of this function is to measure the financial consequences of
prediction errors, specifically differentiating between the expenses related to false positives (FP) and
false negatives (FN). The cost of false positives (FP) is fixed at $100,000, whereas the cost of false
negatives (FN) is $60,000. The relative cost ratio, which is essential for the loss function, is determined
by dividing the cost of false negatives (FN) by the cost of false positives (FP). An important component
of this approach involves determining the frequency of positive cases within the training data (y_train).
The prevalence rate plays a crucial role in calculating the anticipated loss during the model evaluation
phase. The models, such as logistic regression models and a Random Forest classifier, undergo a K-
Fold cross-validation process with 5 splits. This approach guarantees a resilient evaluation of the
models' performance. During the process of cross-validation, the model makes predictions for each
fold, and these predictions are then utilized to build Receiver Operating Characteristic (ROC) curves.
The curves display the rates of false positives (FPR), true positives (TPR), and a variety of thresholds.
A threshold that maximizes the trade-off between true positive rate (TPR) and false positive rate (FPR)
is determined for each fold, taking into account the cost of false negatives and the prevalence rate. The
anticipated loss for each fold is computed by taking into account the amount of false positives and
false negatives at the optimal threshold, adjusting them based on their respective costs, and
normalizing them by the number of observations in the fold. This calculation yields a distinct financial
consequence of the forecast inaccuracies. The procedure calculates the mean of the best thresholds and
projected losses for each model across all the folds, enabling a thorough assessment. More precisely,
the study provides information about the threshold and anticipated loss for the fifth fold, giving ian n-
depth understanding of how well the model performs on a specific portion of the data. In the last phase
of this study, the most effective model, chosen based on the average anticipated loss, is utilized on a
separate dataset. The Random Forest model is determined to be the optimal model in this scenario.
The model is trained using the entire training dataset and subsequently employed to make predictions
on the holdout data. The cross-validation results in an ideal threshold, which is then used to classify
the observations based on the predictions. A confusion matrix is created based on these predictions,
classifying them as true positives, true negatives, false positives, and false negatives. Subsequently,
the matrix is transformed into percentage values, offering a clearer perspective on the model's ability
to effectively categorize each category. To summarize, this strategy that relies on a loss function
emphasizes the significance of financial consequences in predictimodellinging. It helps in choosing
models that strike a balance between accuracy and economic feasibility. This method offers a more
comprehensive comprehension of the efficacy of the models, surpassing traditional accuracy metrics
by evaluating the monetary consequences of forecast errors.
## Manufacturing and Services Firms
Task 2 involves the application of predictmodellingling to distinguish and analyze manufacturing and
service organizations. The dataset is classified into manufacturing and service sectors according to
industry codes ('ind2'). Manufacturing businesses are categorized based on industry codes (27, 29, 28,
26, 30), and a binary variable ('manufacturing_dummy') is generated to indicate their presence. The
dataset is subsequently divided into two subsets: data_manu for manufacturing companies and
data_serv for service companies. An examination of these subcategories unveils diverse growth
patterns: roughly 8% of manufacturing companies and 11% of service companies are categorized as
'rapid growth'. Random Forest classifiers are utilized to construct prediction models for both domains.
The models are customized using variables that are relevant to each industry, such as elements relating
to finance and human resources. The data for each sector is partitioned into training and holdout sets,
preserving an 80-20 ratio. The Random Forest models for each sector are optimized using
GridSearchCV, with a specific emphasis on tuning parameters such as 'max_features' and
'min_samples_split'. The performance of each parameter combination is assessed using accuracy, ROC
AUC, and RMSE metrics. The outcomes of this optimization process are condensed, presenting the
performance metrics for each set of parameters. The optimal parameters determined are utilized to
train the RandomForestClassifier for every sector. The models are assessed using cross-validation,
wherein a loss function is employed to compute the anticipated loss and optimal classification
threshold. This takes into account the costs associated with false positives and false negatives, as well
as the prevalence of positive cases in the training data. In the industrial industry, the Random Forest
model, referred to as 'randomforestprob', exhibits the following performance metrics: a CV RMSE
(Cross-Validation Root Mean Squared Error) of 0.427, a CV AUC (Cross-Validation Area Under the
Curve) of 0.69, and an average optimal threshold of 0.699. The anticipated loss for Fold 5 in this model
is 15.059. In the service sector, the 'randomforestprob' model produces a cross-validated root mean
squared error (CV RMSE) of 0.403, a cross-validated area under the curve (CV AUC) of 0.785, and
an average optimal threshold of 0.563. The anticipated loss for Fold 5 is significantly reduced to 0.278.
Subsequently, the remaining data for each sector is utilized to conduct a more comprehensive
evaluation of the model's effectiveness. The Random Forest model applied to manufacturing
enterprises has an RMSE (Root Mean Square Error) of 0.441, an AUC (Area Under the Curve) of 0.68, and an expected loss of 14.882 when evaluated on the holdout data. The holdout data for service firms
yields an RMSE of 0.403, an AUC of 0.792, and a little greater anticipated loss of 0.291. Ultimately,
confusion matrices are produced for every sector, providing a breakdown of the proportions of accurate
positive predictions, accurate negative predictions, inaccurate positive predictions, and inaccurate
negative predictions. These matrices offer a comprehensive perspective on the model's predicted
precision in categorizing companies into 'rapid growth' and 'not fast growth' groups within each
industry. To summarize, this investigation provides a thorough perspective on the unique patterns of
growth in the manufacturing and service sectors utilizing Random Forest models. The performance
indicators of the models, such as RMSE, AUC, and anticipated loss, offer vital insights into their ability
to accurately predict and classify organizations according to their growth patterns, as well as their
financial ramifications.
## Conclusion
The analysis effectively utilized a range of statistical andmachine-learningg techniques to forecast the
expansion of the company. The Random Forest model proved to be the most efficient, demonstrating
exceptional performance in both the industrial and service industries. This study emphasizes the needto
dog sector-specific analysis in predictive momodellingnd provides a strong framework for future
analyses in related fields. Additional investigation could focus on enhancing the predicted
