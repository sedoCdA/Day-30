# SUV Purchase Prediction using Logistic Regression 

This project demonstrates a complete **end-to-end Machine Learning pipeline** using **Logistic Regression** to predict whether a customer will purchase an SUV based on their **Age** and **Estimated Salary**.

The project covers:

* Data Exploration
* Data Preprocessing
* Model Training
* Model Evaluation
* Visualization
* Model Improvement
* Interview Questions
* AI-assisted Explanation

---

# Part A 

## 1. Data Loading & Exploration

We first load the dataset using Pandas and explore its structure.

### Steps:

* Load dataset using `pd.read_csv()`
* Display first 5 rows using `head()`
* Check dataset shape using `shape`
* View column names
* Check data types using `info()`
* Check missing values using `isnull().sum()`

### Why this is important:

Data exploration helps us understand:

* Structure of dataset
* Data types (numerical/categorical)
* Whether cleaning is required

---

## 2. Data Preprocessing

### a) Handling Missing Values

We check for missing values. If present:

* Numerical → fill with mean/median
* Categorical → fill with mode

(In this dataset, usually no missing values are present.)

---

### b) Encoding Categorical Variables

The dataset contains a categorical column:

* Gender → Male/Female

We convert it into numerical format:

* Male → 1
* Female → 0

Why?
Machine learning models only understand numbers.

---

### c) Feature Selection

We select:

* Age
* EstimatedSalary

These are important features influencing purchase decision.

---

### d) Define X and y

* X → Features (independent variables)
* y → Target (Purchased)

---

## 3. Train - Test Split

We split data into:

* Training set (80%)
* Testing set (20%)

### Why?

* Train model on training data
* Test performance on unseen data

---

## 4. Feature Scaling

We apply **StandardScaler**:

* Converts data into standard normal distribution
* Mean = 0, Standard deviation = 1

### Why?

Logistic Regression is distance-based → scaling improves performance.

---

## 5. Model Training - Logistic Regression

We train the model using:

```python
LogisticRegression()
```

### Why Logistic Regression?

* Best for binary classification
* Simple and interpretable
* Fast to train

---

# Part B 

## 1. Model Evaluation

### a) Accuracy

Accuracy measures how many predictions are correct:

```
Accuracy = Correct Predictions / Total Predictions
```

---

### b) Confusion Matrix

A confusion matrix gives detailed performance:

|          | Predicted 0 | Predicted 1 |
| -------- | ----------- | ----------- |
| Actual 0 | TN          | FP          |
| Actual 1 | FN          | TP          |

Where:

* TP → True Positive
* TN → True Negative
* FP → False Positive
* FN → False Negative

### Importance:

It helps understand not just accuracy but **type of errors**.

---

## 2. Visualization — Decision Boundary

We can plot decision boundary using:

* Age vs Salary (2D)

### Interpretation:

* Shows how model separates classes
* Clear boundary = good model

---

## 3. Model Improvement

We tested different train-test splits:

| Split | Accuracy         |
| ----- | ---------------- |
| 80/20 | Good             |
| 75/25 | Slight change    |
| 70/30 | Slight variation |

### Insight:

Model performance slightly varies with split, but remains stable → good generalization.

---

# Part C 

## Q1: What is Logistic Regression?

Logistic Regression is a **supervised machine learning algorithm** used for **classification problems**, especially binary classification.

### Key Points:

* Despite name "regression", it is used for classification
* Uses **sigmoid function** to map values between 0 and 1

### Formula:

```
p = 1 / (1 + e^-z)
```

Where:

* p = probability
* z = linear combination of inputs

### Output:

* If p > 0.5 → Class 1
* Else → Class 0

---

## Q2: Train-Test Split and Scaling (Detailed Code)

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Explanation:

* `fit_transform()` → learns + applies scaling (train)
* `transform()` → applies same scaling (test)

---

## Q3: What is Confusion Matrix?

A confusion matrix is a table used to evaluate classification models.

### It represents:

* Correct predictions
* Incorrect predictions
* Type of errors

### Why important?

Accuracy alone can be misleading. Confusion matrix shows:

* False positives
* False negatives

---

# Part D 

## Prompt Used:

"Explain Logistic Regression with Python example using sklearn on SUV dataset."

---

## AI Output Evaluation

### 1. Code Correctness:

The code correctly:

* Loads data
* Preprocesses data
* Trains model
* Evaluates results

---

### 2. Steps Completeness:

Complete ML pipeline included:

* Data loading
* Preprocessing
* Scaling
* Training
* Evaluation

---

### 3. Critical Analysis:

* AI output is correct but requires human validation
* Important to verify:

  * Correct features
  * Proper scaling
  * Correct evaluation metrics

---

# Final Conclusion

* Logistic Regression works well for binary classification
* Feature scaling significantly improves results
* Model gives stable accuracy across splits
* Confusion matrix provides deeper insights than accuracy

