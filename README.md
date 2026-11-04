# Online Shopping Purchase Prediction

## a. Problem statement

When people visit an online shopping website, they may browse several pages but leave without buying anything. In this project, I wanted to see whether information about a visitor's browsing session could help predict if they would make a purchase.

I trained five machine learning classification models using the same dataset and compared their results. I also built a Streamlit app to evaluate the saved models on test data. The app displays the results without training the models again.

## b. Dataset description

I used the **Online Shoppers Purchasing Intention** dataset from the **UCI Machine Learning Repository (dataset ID 468)**.

- **Total records:** 12,330 shopping sessions
- **Input features:** 17
- **Target column:** `Revenue`
- **Target values:** `True` means a purchase was made; `False` means no purchase was made.

The input features describe shopping activity, including the number of pages visited, time spent on pages, bounce and exit rates, page values, month, and visitor information.

Of the 12,330 sessions, **10,422 (84.5%)** did not lead to a purchase, while **1,908 (15.5%)** did. This means the dataset has many more non-purchase records than purchase records.

**Data preparation:** I checked the dataset for missing values and duplicate rows. The notebook reported no missing values and 125 duplicate rows. I kept those rows for this experiment. I then used an **80:20 stratified split**, with 9,864 records for training and 2,466 for testing (`random_state=42`). Numerical features were scaled and categorical features were encoded within the model preprocessing pipelines.

## c. GitHub Repository Link

**GitHub repository:** https://github.com/lpriya2096/Online-Shoppers-Purchasing-Intention-ML-Assignment-2

**Live Streamlit app:** https://online-shoppers-purchasing-intention-ml-assignment-2-nz3biup75.streamlit.app/

The repository contains the notebook, Streamlit application, dependencies, test CSV, and saved models needed to run the project.

## d. Models used

I trained the five classification models required for the assignment and evaluated each one on the test set. I used **Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC)** to compare them.

### Model comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8808 | 0.8828 | 0.7366 | 0.3586 | 0.4824 | 0.4592 |
| Decision Tree | 0.8990 | 0.9172 | 0.7111 | 0.5864 | 0.6428 | 0.5883 |
| KNN | 0.8767 | 0.7633 | 0.6822 | 0.3822 | 0.4899 | 0.4493 |
| Naive Bayes | 0.7948 | 0.8070 | 0.3980 | 0.6335 | 0.4889 | 0.3844 |
| Support Vector Machine | 0.8873 | 0.8824 | 0.7047 | 0.4686 | 0.5629 | 0.5149 |

*These are the saved evaluation results from my updated `online_shoppers_intention.ipynb` notebook.*

### Observations about model performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | This model achieved the highest precision (0.7366). When it predicted a purchase, it was often correct. However, its recall was only 0.3586, so it missed many sessions where a purchase actually happened. |
| **Decision Tree** | This model gave the best overall results. It had the highest accuracy (0.8990), AUC (0.9172), F1 Score (0.6428), and MCC (0.5883). It identified more purchases than Logistic Regression while keeping reasonable precision. |
| **KNN** | KNN reached 0.8767 accuracy, but its recall was 0.3822. It did reasonably well overall but did not identify as many actual purchases as the Decision Tree. |
| **Naive Bayes** | The first version of this model had very low accuracy (0.2729). I tried tuning the Gaussian Naive Bayes `var_smoothing` parameter using cross-validation on the training data. With the selected value of `0.1`, its test accuracy improved to 0.7948. It achieved the highest recall (0.6335), but its precision was only 0.3980, so it also predicted purchases for many sessions that did not result in a purchase. |
| **Support Vector Machine** | SVM achieved 0.8873 accuracy and an F1 Score of 0.5629. Its results were strong compared with several other models, although it still missed more actual purchases than the Decision Tree and Naive Bayes. |
| **Overall winner for the dataset** | **Decision Tree** is my choice because it achieved the highest Accuracy, AUC, F1 Score, and MCC on this test set. I did not use accuracy alone to decide because most sessions in the dataset did not result in a purchase. |

### Streamlit application

I created a Streamlit app to make it easier to compare the trained models. The app lets users:

1. Upload a **test CSV** containing the input features and `Revenue` column.
2. Select one of the five saved classification models.
3. View the six evaluation metrics.
4. Check the confusion matrix to see correct and incorrect predictions.

The app loads the saved models and **does not retrain them** using the uploaded test data.

To run the app locally, install the dependencies and start Streamlit:

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Conclusion

This assignment helped me understand how different classification algorithms perform on the same data. I found that high accuracy does not always mean a model is good at identifying purchases, especially when purchase records are less common. The Decision Tree gave the best overall results in my experiment. I also learned how to tune Naive Bayes and use saved models in a Streamlit application.

