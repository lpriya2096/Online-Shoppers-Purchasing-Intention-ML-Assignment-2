
# Online Shoppers Purchasing Intention Using Machine Learning

**Machine Learning – Assignment 2 | M.Tech (AIML), BITS Pilani**

**Name : Lakshmipriya VS**

**BITS ID : 2026ak05650**

## a. Problem Statement

Many people visit online shopping websites, look at products, and leave without making a purchase. In this project, I used information about their browsing sessions to predict whether a visit would end in a purchase.

I trained five classification models on the same dataset and compared them using six evaluation metrics. I also created a Streamlit application that loads the trained models and evaluates them on uploaded test data. The application does not retrain the models.

The target variable is **`Revenue`**:
- **True:** The shopping session ended in a purchase.
- **False:** The shopping session did not end in a purchase.

## b. Dataset Description

**Dataset:** Online Shoppers Purchasing Intention Dataset  
**Source:** UCI Machine Learning Repository  
**Dataset link:** https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

The dataset contains **12,330 shopping sessions**, with **17 input features** and **one target column (`Revenue`)**. It includes information such as pages visited, time spent on pages, bounce rates, exit rates, page values, the month of the visit, and visitor type.

There are **10,422 sessions (84.5%) without a purchase** and **1,908 sessions (15.5%) with a purchase**. Because the two classes are not balanced, I considered other metrics alongside accuracy.

### Data Preparation

I checked the data for missing values and duplicates. The notebook reported no missing values and 125 duplicate rows; I kept the duplicate rows in this experiment. I used an **80:20 stratified train-test split** (`random_state=42`), giving **9,864 training records** and **2,466 test records**. Numerical features were scaled, and categorical features were one-hot encoded through preprocessing pipelines fitted on the training data.

## c. GitHub Repository Link

**GitHub repository:** https://github.com/lpriya2096/Online-Shoppers-Purchasing-Intention-ML-Assignment-2

**Live Streamlit app:** https://online-shoppers-purchasing-intention-ml-assignment-2-nz3biup75.streamlit.app/

The project repository is intended to contain the following files:

```text
Online-Shoppers-Purchasing-Intention/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_comparison.csv
├── online_shoppers_intention.ipynb
└── model/
    └── saved trained models for all five classifiers (.joblib)
```

The saved models include the preprocessing needed for evaluation when exported as complete pipelines. Model filenames must match the paths used in `app.py`.

## d. Models Used

I implemented the five classification models required for this assignment:

1. **Logistic Regression:** Uses a linear relationship to estimate the chance of a purchase.
2. **Decision Tree:** Makes predictions by learning a series of decision rules from the data.
3. **K-Nearest Neighbors (KNN):** Predicts based on similar records in the training data.
4. **Gaussian Naive Bayes:** Uses probabilities to classify each shopping session.
5. **Support Vector Machine (SVM):** Learns a boundary to separate purchasing and non-purchasing sessions.

### Model Comparison Table

These values are from the saved results in my updated `online_shoppers_intention.ipynb` notebook.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8808 | 0.8828 | 0.7366 | 0.3586 | 0.4824 | 0.4592 |
| Decision Tree | **0.8990** | **0.9172** | 0.7111 | 0.5864 | **0.6428** | **0.5883** |
| KNN | 0.8767 | 0.7633 | 0.6822 | 0.3822 | 0.4899 | 0.4493 |
| Naive Bayes | 0.7948 | 0.8070 | 0.3980 | **0.6335** | 0.4889 | 0.3844 |
| Support Vector Machine | 0.8873 | 0.8824 | 0.7047 | 0.4686 | 0.5629 | 0.5149 |

### Evaluation Metrics

- **Accuracy:** How often the model makes the correct prediction overall.
- **AUC:** How well the model separates purchase and non-purchase sessions across different thresholds.
- **Precision:** Out of the sessions predicted as purchases, how many were actually purchases.
- **Recall:** Out of the sessions that resulted in purchases, how many the model identified.
- **F1 Score:** A combined measure of precision and recall.
- **MCC:** A measure of prediction quality that considers all four outcomes in the confusion matrix; useful when the classes are imbalanced.

### Observations About Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | It had the highest precision (**0.7366**), so its purchase predictions were often correct. However, recall was only **0.3586**, which means it missed many actual purchases. |
| **Decision Tree** | It performed best overall, with the highest accuracy (**0.8990**), AUC (**0.9172**), F1 (**0.6428**), and MCC (**0.5883**). It gave a better balance between finding purchases and avoiding incorrect purchase predictions. |
| **KNN** | Its accuracy was **0.8767**, but recall was only **0.3822**. It classified many sessions correctly overall but missed a large share of actual purchases. |
| **Naive Bayes** | The initial version had low accuracy (**0.2729**). After tuning Gaussian Naive Bayes `var_smoothing` using cross-validation on the training data, the selected value was **0.1** and test accuracy increased to **0.7948**. It achieved the highest recall (**0.6335**) but lower precision (**0.3980**), so it also produced more false purchase predictions. |
| **Support Vector Machine** | It achieved **0.8873** accuracy and **0.5629** F1. It performed reasonably well, but its recall (**0.4686**) was lower than the Decision Tree and Naive Bayes models. |
| **Overall Winner** | **Decision Tree** was the best choice for this dataset based on the highest accuracy, AUC, F1, and MCC. I considered F1 and MCC alongside accuracy because non-purchase sessions make up most of the dataset. |

### Results and Best-Model Observation

- **Highest accuracy:** Decision Tree (0.8990)
- **Highest AUC:** Decision Tree (0.9172)
- **Highest precision:** Logistic Regression (0.7366)
- **Highest recall:** Naive Bayes (0.6335)
- **Highest F1:** Decision Tree (0.6428)
- **Highest MCC:** Decision Tree (0.5883)

The **Decision Tree** was the strongest overall model in my test results. Naive Bayes found the largest share of actual purchases, but it also had more false positives. This comparison showed me why choosing a model using accuracy alone can be misleading.

### Streamlit Application

The application allows users to upload a **labeled test CSV**, choose one of the five saved models, view all six evaluation metrics, and inspect a confusion matrix. The CSV must include the required input features and the `Revenue` target column. The application uses the saved trained models **without retraining**.

To run the application locally:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### What I Learned
Assignment helped me understand how different classification algorithms perform on the same data. I found that high accuracy does not always mean a model is good at identifying purchases, especially when purchase records are less common. The Decision Tree gave the best overall results in my experiment. I also learned how to tune Naive Bayes and use saved models in a Streamlit application.

