- **Logistic Regression**: accuracy 0.857, AUC 0.910, MCC 0.589; favours precision over recall (misses many >50K cases). Best on: none; weakest on: none.
- **Decision Tree**: accuracy 0.867, AUC 0.911, MCC 0.612; favours precision over recall (misses many >50K cases). Best on: Accuracy, AUC, Precision, MCC; weakest on: Recall.
- **kNN**: accuracy 0.846, AUC 0.890, MCC 0.564; favours precision over recall (misses many >50K cases). Best on: none; weakest on: none.
- **Naive Bayes**: accuracy 0.536, AUC 0.749, MCC 0.330; favours recall over precision (over-predicts >50K). Best on: Recall; weakest on: Accuracy, AUC, Precision, F1, MCC.
- **Random Forest (Ensemble)**: accuracy 0.862, AUC 0.909, MCC 0.607; favours precision over recall (misses many >50K cases). Best on: F1; weakest on: none.

**Overall winner (best average rank across 6 metrics): Decision Tree** (MCC 0.612, AUC 0.911).