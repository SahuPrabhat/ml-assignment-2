# Machine Learning Assignment - 2
**Work Integrated Learning Programmes Division**  
**M.Tech (AIML) - BITS Pilani**  
**Student Name:** Prabhat Sahu  

---

## a. Problem Statement
The objective of this assignment is to implement, evaluate, and compare multiple machine learning classification algorithms on a real-world benchmark dataset, and deploy an interactive web application on Streamlit Community Cloud. 

The task is a binary classification problem: predicting whether a detected breast tissue biopsy is **Malignant (cancerous, class 0)** or **Benign (non-cancerous, class 1)** based on morphological characteristics extracted from cell nuclei images. Accurate automated classification helps pathologists make quicker, more reliable diagnostic decisions.

---

## b. Dataset Description
- **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source:** UCI Machine Learning Repository / Kaggle (Donated by Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian)
- **Number of Instances:** 569 samples (212 Malignant, 357 Benign)
- **Number of Features:** 30 continuous numeric input features (exceeds the required minimum of 12 features)
- **Target Variable:** Binary class (`0` = Malignant, `1` = Benign)

The 30 features represent ten real-valued characteristics computed for each cell nucleus:
1. `radius` (mean of distances from center to points on the perimeter)
2. `texture` (standard deviation of gray-scale values)
3. `perimeter`
4. `area`
5. `smoothness` (local variation in radius lengths)
6. `compactness` (perimeter^2 / area - 1.0)
7. `concavity` (severity of concave portions of the contour)
8. `concave points` (number of concave portions of the contour)
9. `symmetry`
10. `fractal dimension` ("coastline approximation" - 1)

For each characteristic, the dataset provides the **Mean**, **Standard Error (SE)**, and **"Worst" (mean of the three largest values)**, resulting in $10 \times 3 = 30$ continuous input features.

The data was split into **80% training set (455 samples)** and **20% test set (114 samples)** using stratified sampling. Continuous features were scaled using `StandardScaler` fitted on the training split.

---

## c. Github Repository Link
- **GitHub Repository:** [https://github.com/SahuPrabhat/ml-assignment-2](https://github.com/SahuPrabhat/ml-assignment-2)
- **Live Streamlit Web App:** [https://sahuprabhat-ml-assignment-2.streamlit.app](https://sahuprabhat-ml-assignment-2.streamlit.app)

---

## d. Models Used & Comparison Table

Five classification algorithms were trained and evaluated on the same 20% holdout test dataset (114 instances). All six required metrics were computed:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| **Decision Tree** | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| **KNN** | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| **Naive Bayes** | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| **Support Vector Machine** | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

---

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed exceptionally well, achieving top accuracy (98.25%) and the highest AUC (0.9954). With 30 continuous standardized features, the linear decision boundary separates benign and malignant samples cleanly. The L2 regularization prevents overfitting and produces well-calibrated probabilities. |
| **Decision Tree** | Obtained 92.11% accuracy and 0.9163 AUC. While decision trees are highly interpretable, a single tree creates orthogonal axis-aligned splits that struggle with smooth continuous boundaries between correlated features, leading to slightly lower generalization compared to linear and kernel methods. |
| **KNN** | Scored 95.61% accuracy and 0.9788 AUC with k=5. Because morphological measurements of cell nuclei cluster tightly according to malignancy, instance-based distance measures work well once features are standardized. Performance is slightly lower than SVM due to the 30-dimensional feature space. |
| **Naive Bayes** | Achieved 92.98% accuracy and a strong AUC of 0.9868. Gaussian Naive Bayes assumes all features are conditionally independent given the class label. In this dataset, features like radius, perimeter, and area are strongly correlated, violating the independence assumption and causing slight probability distortion. |
| **Support Vector Machine** | Tied for the highest accuracy (98.25%) and achieved 0.9950 AUC. The RBF kernel maps the 30-dimensional feature space into a higher-dimensional space where an optimal margin hyperplane cleanly separates malignant from benign cases with an MCC score of 0.9623. |
| **Overall, Winner for your dataset?** | **Logistic Regression** (tied with Support Vector Machine on Accuracy, F1, and MCC). Logistic Regression is chosen as the overall winner because it achieves a slightly superior AUC (0.9954 vs 0.9950), is computationally efficient, less prone to hyperparameter sensitivity, and directly outputs interpretable class probabilities for clinical risk evaluation. |

---

## Repository Structure
```text
project-folder/
│-- app.py                   # Streamlit web application
│-- requirements.txt         # Required python libraries
│-- README.md                # Project documentation and results
│-- test_data.csv            # Test partition used for evaluation (114 samples)
│-- model/
    │-- logistic_regression.pkl  # Trained Logistic Regression model
    │-- decision_tree.pkl        # Trained Decision Tree model
    │-- knn.pkl                  # Trained K-Nearest Neighbors model
    │-- naive_bayes.pkl          # Trained Gaussian Naive Bayes model
    │-- svm.pkl                  # Trained Support Vector Machine model
    │-- scaler.pkl               # Fitted StandardScaler object
    │-- metrics.json             # Pre-calculated test metrics
    │-- train_models.py          # Script used to train and save all models
```

---

## How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/SahuPrabhat/ml-assignment-2.git
   cd ml-assignment-2
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit web app:
   ```bash
   streamlit run app.py
   ```
4. Open the local URL (`http://localhost:8501`) in your browser to interact with the models, upload test data, and review metrics.
