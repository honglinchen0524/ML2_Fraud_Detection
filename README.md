# ML2_Fraud_Detection

Apply Hybrid and Ensemble Machine Learning for Credit Card Fraud Detection

1. Download dataset here: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Save as "creditcard.csv"
3. Run in terminal
   mkdir data results
4. Start preprocess
   python data_preprocess.py

5. Run MLP (edit line 10 for suffix: "\_34" and "\_10", and then run below command after changing suffix)
   Rscript mlp_fraud.R

6. Run Naive Bayes
   python naive_bayes.py \_34
   python naive_bayes.py \_10

7. Ensemble model and evaluate model
   python ensemble_evaluate.py
