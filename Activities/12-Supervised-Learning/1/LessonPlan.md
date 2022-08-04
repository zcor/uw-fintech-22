# Module 12: Virtual Class I Lesson Plan (2 hours)
## Overview

The goal of this session is to familiarize students with supervised learners and the process of evaluating their performance.

Prior to the two hour class time starting, there will be 30 minutes of office hours.

In the first section of class, you will introduce supervised learning along with the logistic regression model.

The middle section of the class will introduce the support vector machine model.

The final section of class time will focus on trees and ensemble learners.

## Learning Objectives

By the end of the session, learners will be able to do the following:

* Apply the model-fit-predict pattern to common supervised machine learning models.

* Utilize logistic regressions, SVMs, decision trees and random forests for machine learning problems.

* Evaluate the performance of a supervised learning model.

---

## Time Tracker

| Start   | #  | Activity                                           | Time |
| ------- | -- | -------------------------------------------------- | ---- |
| 7:00 PM | 1  | Instructor Do: Introduction to Supervised Learning | 0:05 |
| 7:05 PM | 2  | Instructor Do: Logistic Regression                 | 0:15 |
| 7:20 PM | 3  | Instructor Do: Support Vector Machines             | 0:15 |
| 7:35 PM | 4  | Student Do: Support Vector Machines                | 0:15 |
| 7:50 PM | 5  | Instructor Do: Review Support Vector Machines      | 0:15 |
| 8:05 PM | 6  | Instructor Do: Decision Trees                      | 0:15 |
| 8:20 PM | 7  | Instructor Do: Random Forests                      | 0:15 |
| 8:35 PM | 8  | Student Do: Random Forests                         | 0:15 |
| 8:50 PM | 9  | Instructor Do: Review Random Forests               | 0:10 |
| 9:00 PM |    | END                                                |      |

---

## Instructor Do: Office Hours (0:30) <required>

Welcome to Office Hours! Remind the students that this is their time to ask questions and get assistance from their instructional staff as they’re learning new concepts and working on the challenge assignment. Feel free to use breakout rooms to create spaces focused on specific topics, or to have 1:1s with students.
Expect that students may ask for assistance such as the following:

* Challenge assignment
* Further review on a particular subject
* Debugging assistance
* Help with computer issues
* Guidance with a particular tool

---
## Class Activities

### 1. Instructor Do: Introduction to Supervised Learning (5 min)

Begin this module with a brief introduction to Supervised Learning:

* Supervised Learning is used to solve problems by using a set of features and fitting a model to a corresponding set of known answers called labels. This contrasts with the unsupervised method of learning that you saw previously, K-Means. In that case labels were inferred based on how close points were to a center value. With supervised learners, a dataset that contains the known labels is used to fit a model for a given set of inputs called features.

* Supervised Learning is widely used in FinTech to solve problems related to credit risk, automated trading, fraud detection and insurance risks just to name a few.

Ask the students if anyone has ever received a call or text from their bank asking if the last purchase made was fraudulent.

Explain that many organizations use supervised learners to detect fraud and these models are part of a front line defense against fraudulent activities. The model will flag the transaction and another piece of the automated system will contact the customer. If the customer responds with "No" they did not make the last transaction, then the automated system will redirect the customer to a human employee at the bank. For example, the [FICO Falcon: Cognitive Fraud Analytics](https://www.fico.com/en/latest-thinking/product-sheet/fico-falcon-platform-cognitive-fraud-analytics-fraud-focused-machine-learning) is a service that financial institutions use to detect fraud.

Expand on supervised learning further using the following talking points:

* Supervised learning requires us to feed a model data with the correct answers. It will learn from the data and answers and become better at predicting the correct answer as we show it more data.

* Supervised learners generally fall into one of two broad categories, Classification and Regression.

Engage the class with the following questions:

**Question**: What is classification?

**Answer**: Classification is used to predict **discrete** valued variable. A discrete variable has no middle, it's values cannot be divided. For example, consider a loan application that asks whether or not you own a car. The answers are Yes or No. You either own a car or you don't. There is no middle value and this type of variable, `car_ownership`, for example would be discrete.

**Question**: What is regression?

**Answer**: Regression is used to predict a **continuous** valued variable. Continuous values mean that the variable can always be divided into a smaller piece. Continuous variables no matter how small will have a middle that we can find. For example, a variable of distance is continuous. We can always find a smaller distance by dividing the current distance by half. In finance, prices and rates are usually continuous.

Explain that the scikit-learn (sklearn) library has many models for classification or regression that can be used depending on the output type (classification for discrete values and regression for continuous values). Sklearn makes it easy to use any of the supervised learning models by providing a common usage pattern: model-fit-predict.

Use the following talking points to explain the model-fit-predict pattern to students:

* All of the models we will study have been developed in such a way that they will follow the same pattern for training the model and generating predicted values.

* **Model** - We create an instance of a supervised learning model by importing the appropriate `class` from a library.

* **Fit** - Fitting the model is the process of using our training data and showing the model each set of conditions and the correct answer. As the model sees more data it will become better fitted. We will need to have our data preprocessed and properly scaled as a prerequisite for this stage.

* **Predict** - We will use testing data that we know the right answers for and feed this to our model. Our model will give as what it thinks is the answer and we will compare the results with the known right answers to determine how well the model performs.

Explain to students that we will use the model-fit-predict pattern to train, test, and evaluate a variety of machine learning models starting with logistic regression.
### 2. Instructor Do: Logistic Regression (15 min)

In this section students will learn about the logistic regression model.

**Files:**

[logistic_regression.ipynb](./Activities/01_Ins_Logistic_Regression/Solved/logistic_regression.ipynb)

Before launching the notebook, take a moment to highlight the following points:

* The **logistic regression** model is frequently used to predict binary outcomes. It uses a special type of function to decide the likelihood that a given observation will belong to one category or another. That function is the **sigmoid** activation function.

* The **sigmoid** activation function looks somewhat like an `S` and as the model becomes better fitted to a dataset the curve of the `S` becomes more vertical.

Slack out the following gif to the class:

![](./Images/12-2-logistical-regression-fit.gif)

Open a new notebook and add the following module imports:

* We will use a simulated dataset from `make_blobs()` which you can read more about on the [documentation page](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html).

* Using simulated datasets allows us to easily and quickly evaluate how a model will perform.

* We can create messier data by using a higher value for standard deviation. Low values of standard deviation will create clusters that have no overlap whereas high values will make our two clusters almost indistinguishable.

  ```python
  # Import Modules
  import pandas as pd
  import matplotlib.pyplot as plt
  from sklearn.model_selection import train_test_split
  from sklearn.datasets import make_blobs
  from sklearn.preprocessing import StandardScaler
  ```

  ```python
  # Generate test dataset
  X, y = make_blobs(
      n_samples=1000,
      centers=2,
      random_state=1,
      cluster_std=3)

  # Convert ndarray to pandas datatypes
  X = pd.DataFrame(X)
  y = pd.Series(y)
  ```

  ```python
  # Plot test data
  plt.scatter(
      x=X[0],
      y=X[1],
      c=y,
      cmap="bwr")

  plt.show()
  ```

* `make_blobs()` returns a set of features and a corresponding set of labels. We can plot the features and labels by using the label values to shade the points as seen in the plot.

* Next, we need to import the `LogisticRegression` model from `sklearn`.

* We will follow a pattern of creating a model, fitting it and then using it to generate predicted values. This pattern will apply to all of the supervised machine learning models we will study.

* We need to split our dataset into a testing and training datasets. We will use the training dataset when we fit our model. The testing dataset will be used to generate the predicted lables and we will have a set of known labels that we can compare the predictions with.

  ```python
  # Instantiate a Logistic Regression model
  from sklearn.linear_model import LogisticRegression
  lr = LogisticRegression()
  lr
  ```

  ```python
  # Split dataset
  X_train, X_test, y_train, y_test = train_test_split(X, y)
  X_train.shape
  ```

* `X_train` and `y_train` are the training features and labels whereas `X_test` and `y_test` are for testing.

* It's important that two things need to be scored and evaluated: the training AND test data.

* When scoring the training data, the accuracy of the model is applied against the training data it was created with. Usually, the more data that is trained, the higher the accuracy score.

* Scoring of test data consists of applying the model (which was learned using the training data) against only the test data. The test data is considered new data; the model has not seen or learned from this data. Testing using this test data is akin to an acid test: it’s a way to gauge how accurately the model might make predictions in real life.

* If the training score is significantly more accurate than the testing score, the model may be overtrained. Overtraining the model can result in overfitting, where the model learned rules for predictions that apply mostly just for the training data set. The goal is to have the scores as close to each other in accuracy as possible.

Next, scale the data, fit the model and generate the predicted values:

  ```python
  # Scale the data
  scaler = StandardScaler()
  X_scaler = scaler.fit(X_train)
  X_train_scaled = X_scaler.transform(X_train)
  X_test_scaled = X_scaler.transform(X_test)

  # Fit the data
  lr.fit(X_train_scaled, y_train)

  # Make predictions using the test data
  y_pred = lr.predict(X_test_scaled)

  results = pd.DataFrame({
      "Prediction": y_pred,
      "Actual": y_test
  }).reset_index(drop=True)

  results.head()
  ```

Pause for a moment and engage the class with the following:

**Question**: What is a confusion matrix?

**Answer:** A confusion matrix is a table used to describe the performance of the model. It highlights differences in the number of samples predicted to belong to a specific class, with the actual number of samples that do belong to that class.

**Question**: Would a user aggregate the columns or rows of a confusion matrix?

**Answer:** Both. Values for columns and rows will be aggregated and then compared. Columns will reflect the sum of predicted categorical outcomes, and the rows will reflect the actual sum of outcomes.

**Question**: What is a classification report?

**Answer:** It is a report that details the precision, recall, and accuracy of the predicted data points for each categorical class. A classification report can be used to determine the rate of false positives, false negatives, and the quality of the predictions.

Add the following module imports and generate a confusion matrix:

* The classification metrics take the predicted label values and the test label values. It counts the number of times the predicted set was right or wrong compared to the test set.

  ```python
  # Import Modules
  from sklearn.metrics import confusion_matrix,classification_report
  ```

  ```python
  # Print confusion matrix
  print(confusion_matrix(y_test, y_pred))
  ```

* We can generate our precision, recall and f-1 scores using a `classifcation_report`.

  ```python
  # Print classification report
  print(classification_report(y_test, y_pred))
  ```

* We can see on the **classification report** that there are two labels `0` and `1` that correspond to the target classes that the model is trying to predict. The model's performance can then be evaluated for each class based on metrics such as Accuracy, Precision, Recall, F1, and more.

Explain that almost all classification models in sklearn can be trained and evaluated using this same approach. Furthermore, because they all follow the model-fit-predict pattern, any model can be substituted and tested. Machine learning engineers can rapidly test a variety of models to see which has the best performance for their data.

Ask students if they have any questions about logistic regression or classification metrics and move on to the next section.

### 3. Instructor Do: Support Vector Machines (15 min)

In this section you will introduce the **support vector machine** to students.

**Files:**

[support_vector_machine.ipynb](./Activities/02_Ins_SVM/Solved/support_vector_machine.ipynb)

Explain to students that because sklearn follows a common pattern of model-fit-predict, machine learning engineers can train, test, and evaluate a variety of machine learning models. To illustrate this, we will look at a new model called a Support Vector Machine or SVM.

Use the following points to introduce SVMs to students:

* **Support vector machines** (SVM) are a widely applied model in FinTech especially in applications concerning credit risk and fraud detection. They are robust and have applications across many industries.

* Support Vector Machines take a unique approach to classification by trying to find the best boundary that separates the data points. Then SVM can assign a class to the points on one side of the boundary and another class to the points on the other side of the boundary.

* The idea behind SVMs is that a dataset and its labels are **projected** into a higher dimensional space.

* You maybe familiar with a globe being **projected** onto a map. SVMs are doing this in reverse. We are taking the map and projecting the globe.

* In this higher dimensional **projection** we will hopefully find that the labels and features are clearly separated. This boundary's **projection** is called a **hyperplane** and we can use it to classify the points.

* A hyperplane is a line that delineates data points into their corresponding classes. All items to the left of the line belong to class A. Items to the right belong to class B.

* The goal with hyperplanes is to get the margin of the hyperplane equidistant to the data points for all classes. This distance is considered the margin of separation.

* The margin is considered optimal when the distance from the hyperplane and the support vectors are equidistant.

![margin_of_separation.png](Images/margin_of_separation.png)

* The data closest to/within the margin of the hyperplane are called support vectors, and they are used to define boundaries of the hyperplane.

* These points are sometimes the most difficult to classify, because they live closest to the margin and could belong to either class.

![support_vectors.png](Images/support_vectors.png)

Educate students on the different orientations for hyperplanes. Provide an understanding of how the orientation and position of the hyperplane is decided.

![hyperplane_orientation.png](Images/hyperplane_orientation.png)

* Hyperplanes can be 2D, clearly delineating classes with non-overlapping data points or outliers.

  * Hyperplanes also support what's considered zero tolerance with perfect partition, which is a non-linear hyperplane that positions and orients the hyperplane to correctly classify overlapping or outlying data points.

    * This hyperplane could be a curved line or a circle, depending on the data points and their proximity to one another.

    * In order to establish zero tolerance with perfect partition, the SVM model may introduce a new `z-axis` dimension for non-linear hyperplanes.

* The `kernel` parameter is used to identify the orientation of the hyperplane. Kernelling, and how to use the `kernel` parameter, will be addressed later in the demo.

Open the provided notebook and add the following module imports:

* We will use a synthetic dataset generated by `make_blobs()` and plot the features shading by the label values.

* With a low standard deviation value of `1.25` the clusters will have no overlap.

  ```python
  # Import Modules
  import pandas as pd
  import matplotlib.pyplot as plt
  from sklearn.model_selection import train_test_split
  from sklearn.datasets import make_blobs
  from sklearn.preprocessing import StandardScaler
  ```

  ```python
  # Generate a test dataset
  X, y = make_blobs(
      n_samples = 500,
      centers = 2,
      random_state = 1,
      cluster_std = 1.25)

  # Convert ndarray to pandas datatypes
  X = pd.DataFrame(X)
  y = pd.Series(y)

  # Plot test data
  plt.scatter(
      x=X[0],
      y=X[1],
      c=y,
      cmap="bwr")

  plt.show()
  ```

Now demonstrate the `SVC` model from `sklearn`.

* The SVC constructor supports a number of arguments, with the `kernel` argument being the most important. Provide students with a brief overview of the `kernel` argument and what kernelling is.

* The `kernel` argument is used to express the dimensionality of the model. It is basically the degree of dimensionality needed to separate the data into classes.

* Communicate to students that a linear `kernel` value should be used for data sets with two classes. This will create a hyperplane that is a line. Non-linear data will result in a hyperplane that is an actual plane.

* The `kernel` argument accepts a number of values. These are listed and explained below. Advise students to consult the documentation to get additional detail on these parameter values.

  * **rbf** creates a non-linear hyperplane

  * **linear** creates a linear, 2D hyperplane

  * **poly** creates a non-linear hyperplane

  ```python
  # Instantiate a linear SVM model
  from sklearn.svm import SVC
  svm = SVC(kernel='linear')
  svm
  ```

* We can then split the dataset into testing and training sets before fitting the model.

* Once the model is fitted we can use it to generate a set of predicted values.

  ```python
  # Split Data
  X_train, X_test, y_train, y_test = train_test_split(X, y)
  X_train.shape

  # Scale the data
  scaler = StandardScaler()
  X_scaler = scaler.fit(X_train)
  X_train_scaled = X_scaler.transform(X_train)
  X_test_scaled = X_scaler.transform(X_test)

  # Fit the data
  svm.fit(X_train_scaled, y_train)

  # Make predictions using the test data
  y_pred = svm.predict(X_test_scaled)
  ```

* The classification report shows that the model fits perfectly which is to be expected since the clusters did not have any overlap.

  ```python
  # Import Module
  from sklearn.metrics import classification_report

  # Print classification report
  print(classification_report(y_test, y_pred))
  ```

* We can generate a new synthetic dataset using a higher value for standard deviation. This results in a dataset that is not as easy for the model to distinguish.

* This new dataset will have one cluster with 1000 instances and another with only 100. This is an example of class imbalance since we have many more observations of one class compared to the other.

  ```python
  # Generate dataset with overlap by increasing standard deviation
  X, y = make_blobs(
      n_samples=[1000, 100],
      random_state=1,
      cluster_std=5)

  # Convert ndarray to pandas datatypes
  X = pd.DataFrame(X)
  y = pd.Series(y)

  # Plot dataset
  plt.scatter(
      x=X[0],
      y=X[1],
      c=y,
      cmap="bwr")

  plt.show()
  ```

* Following the same pattern as before, we can split the dataset, fit the model and generate a set of predicted values.

* The classification report shows that this second dataset was harder to classify for SVM though it still performed well.

  ```python
  # Split data
  X_train, X_test, y_train, y_test = train_test_split(X, y)
  X_train.shape

  # Scale the data
  scaler = StandardScaler()
  X_scaler = scaler.fit(X_train)
  X_train_scaled = X_scaler.transform(X_train)
  X_test_scaled = X_scaler.transform(X_test)

  # Fit the data
  svm.fit(X_train_scaled, y_train)

  # Make predictions using the test data
  y_pred = svm.predict(X_test_scaled)

  # Print classification report
  print(classification_report(y_test, y_pred))
  ```

Ask students if they have any questions and then move on to the student activity.

### 4. Student Do: Support Vector Machines (15 min)

In this activity, students will build a Support Vector Machine (SVM) classifier that can be used to predict the loan status (approve or deny) given a set of input features.

Slack out the following files to the students.

**Files:**

[Instructions](./Activities/03_Stu_SVM/README.md)

[loans.csv](./Activities/03_Stu_SVM/Resources/loans.csv)

[Starter Code](./Activities/03_Stu_SVM/Unsolved/svm_loan_approver.ipynb)

**Instructions:**

1. Read the data into a Pandas DataFrame.

2. Separate the features `X` from the target `y`. In this case, the loan status is the target.

3. Separate the data into training and testing subsets.

4. Scale the data using `StandardScaler`.

5. Import and instantiate an SVM classifier using sklearn.

6. Fit the model to the data.

7. Calculate the accuracy score using both the training and the testing data.

8. Make predictions using the testing data.

9. Generate the confusion matrix for the test data predictions.

10. Generate the classification report for the test data.


**Bonus**: Compare the performance of the SVM model against the logistic regression model. Decide which model performed better, and be prepared to discuss these results in an upcoming activity. Performance results for the logistic regression model can be found below.

### 5. Instructor Do: Review Support Vector Machines (15 min)

**Files:**

[Solution code](./Activities/03_Stu_SVM/Solved/svm_loan_approver.ipynb)

Open the solution and explain the following:

* The SVM algorithm is an alternative to the logistic regression model when it comes to running classification engines. The SVM algorithm is often more accurate than other models.

* SVM models can be either 2D (linear) or multi-dimensional (poly). The dimension of the model is defined by the `kernel` argument.

  ```python
  # Instantiate a linear SVM model
  from sklearn.svm import SVC
  svm_model = SVC(kernel='linear')
  svm_model
  ```

* Once the model is created, it has to be fit with data.

  ```python
  # Fit the data
  svm_model.fit(X_train_scaled, y_train)
  ```

* The model can then be scored for accuracy.

  ```python
  # Score the accuracy
  print(f"Training Data Score: {svm_model.score(X_train_scaled, y_train)}")
  print(f"Testing Data Score: {svm_model.score(X_test_scaled, y_test)}")
  ```

* An accurate model can make precise predictions. The sklearn `predict` function is used to make predictions off of the new data.

  ```python
  # Make predictions using the test data
  y_pred = svm_model.predict(X_test_scaled)

  results = pd.DataFrame({
      "Prediction": y_pred,
      "Actual": y_test
  }).reset_index(drop=True)

  results.head()
  ```

* The last step is to evaluate the model. Just like with the logistic regression model, the `confusion_matrix` and `classification_report` libraries can be used to assess metrics and performance.

  ```python
  # Evaluate performance
  from sklearn.metrics import confusion_matrix
  confusion_matrix(y_test, y_pred)

  from sklearn.metrics import classification_report
  print(classification_report(y_test, y_pred))
  ```

**Bonus Question**: Ask students which model performed better in terms of accuracy, precision, and recall.

**Answer:** The SVM model performed better in terms of accuracy, precision, and recall.

Ask students if they have any remaining questions before moving on to the next section.
### 6. Instructor Do: Decision Trees (15 min)

In this lesson, students will be introduced to decision trees which form the basis for the family of tree-based algorithms.

**Files:**

[credit_data.csv](./Activities/04_Ins_Decision_Trees/Resources/credit_data.csv)

[decision-trees.ipynb](./Activities/04_Ins_Decision_Trees/Solved/decision-trees.ipynb)


Explain to students that in addition to logistic regression and SVMs, there's another family of supervised learning models called trees that are widely used in finance. To start, we will look at the basic tree model called a decision tree.

Introduce tree-based classifiers with the following points:

* Decision trees are widely used in Finance applications because unlike the logistic regression and support vector machines, decision trees are easily auditable. In other words, you can trace the decision logic throughout each step of the model to see how the model reached the final prediction. This may be critical if you need to justify a loan decision or other financial decision.

* In contrast to linear models, tree-based algorithms also have another strong advantage in that they can map non-linear relationships in data.

* In linear models, the relationship among input variables can be represented as a straight line, while non-linear models have a different shape.

* Predicting the price of a house based on its size is an example of a linear problem. This is because, as a general rule, the size of the house is directly proportional to the price of the house.

* Predicting if a credit application is going to be fraudulent or not may be an example of a non-linear problem, due to the complex relationship between the input features and the output prediction.

* Decision trees encode a series of `True/False` questions.

* `True/False` questions can be represented with a series of if/else statements.

* There are some key concepts that are important to know when working with decision trees:

  * **Root Node:** A node that is divided into two or more homogeneous sets and represents the entire population or sample data.
  * **Parent Node:** A node that is divided into sub-nodes.
  * **Child Node:** Sub-nodes of a parent node.
  * **Decision Node:** A sub-node that is split into further sub-nodes.
  * **Leaf or Terminal Node:** Nodes that do not split.
  * **Branch or Sub-Tree:** A subsection of the entire tree.
  * **Splitting:** The process of dividing a node into two or more sub-nodes.
  * **Pruning:** The process of removing sub-nodes of a decision node.
  * **Tree's Depth:** The number of decision nodes encountered before making a decision.

* Decision trees can become very complex and deep, depending on how many questions have to be answered. Deep and complex trees tend to overfit to the training data, and do not generalize well to new data.

Open the provided notebook and code the following activity:

* In the initial import cell, the `tree` module from `sklearn` is imported, since it offers a decision tree implementation for classification problems.

  ```python
  from sklearn import tree
  ```

* One interesting way to analyze a decision tree is to visualize it. The following libraries are imported to create a visual representation of the decision tree.

  ```python
  import pydotplus
  from IPython.display import Image
  ```

* Students should have already installed the [`pydotplus` package](https://anaconda.org/conda-forge/pydotplus).  If not, they can install it in their virtual environments by executing the following command in the terminal. This visualization step can also be skipped for anyone that is having installation issues.

  ```bash
  conda install -c conda-forge pydotplus
  ```

* The `credit_data.csv` is loaded in a DataFrame called `df`.

  ```python
  # Loading data
  file_path = Path("../Resources/credit_data.csv")
  df = pd.read_csv(file_path)
  ```

* Once the data is loaded into the `df` DataFrame, the features and target sets are created. The features set `X` contains all the columns except the `credit_risk` column. The `credit_risk` column is stored in the target variable `y`. We set the index to be the `id` column.

  ```python
  # Split target column from dataset
  y = df['credit_risk']
  X = df.drop(columns='credit_risk')

  # Set Index
  X = X.set_index('id')
  ```

* Next the categorical variables will need to be one-hot encoded with `get_dummies`.

  ```python
  # Encode the categorical variables using get_dummies
  X = pd.get_dummies(X)

  X.head()
  ```

Explain to students that in order to train and validate the decision tree model, the data is split into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```

Explain to students that in order to improve an algorithm's performance, the features data will be scaled using the `StandardScaler`. There is no need to scale the target data, since it contains the labels that we want to predict using the decision tree.

```python
# Creating StandardScaler instance
scaler = StandardScaler()

# Fitting Standard Scaler with the training data
X_scaler = scaler.fit(X_train)

# Scaling data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
```

* After scaling the features data, the decision tree model can be created and trained.

  ```python
  # Creating the decision tree classifier instance
  model = tree.DecisionTreeClassifier()
  ```

* The model is trained with the scaled training data.

  ```python
  model = model.fit(X_train_scaled, y_train)
  ```

* After fitting the model, some predictions are made using the scaled testing data.

  ```python
  predictions = model.predict(X_test_scaled)
  ```

* After observing the results, it can be seen that the model's accuracy (`0.68`).

Finally, comment to students that an interesting way to analyze a decision tree is by visualizing it, so a visual representation of the final decision tree is created using `pydotplus` library.

  ![Decision tree visualization](./Images/decision-trees-2.png)

* We can visualize the graph by using `export_graphiz`.

  ```python
  # Create DOT data
  dot_data = tree.export_graphviz(
      model, out_file=None, feature_names=X.columns, class_names=["0", "1"], filled=True
  )

  # Draw graph
  graph = pydotplus.graph_from_dot_data(dot_data)

  # Show graph
  Image(graph.create_png())
  ```

* To ease the tree visualization, the image can be saved as `PDF` or `PNG`.

  ```python
  # Saving the tree as PDF
  file_path = "../Resources/credit_tree.pdf"
  graph.write_pdf(file_path)

  # Saving the tree as PNG
  file_path = "../Resources/credit_tree.png"
  graph.write_png(file_path)
  ```

Explain that the size and shape of the tree indicates the complexity in the decision logic needed to solve the problem. For very complex data, the decision trees can become very large and nested. This often leads to overfitting to the training data (the tree is very good at predicting classes for data it has seen, but not good at predicting classes for new data points). The good news is that there are tree-based models that are robust against overfitting. Next, we will cover one of the most popular models called a Random Forest.

Answer any questions before moving on.

### 7. Instructor Do: Random Forests (15 min)

In this activity, students will be introduced to ensemble learning, weak learners, and random forests.

**Files:**

[loans_data_encoded.csv](./Activities/05_Ins_Random_Forest/Resources/credit_data.csv)

[random-forest.ipynb](./Activities/05_Ins_Random_Forest/Solved/random-forest.ipynb)

Explain to the class that they will be learning about a new and powerful family of machine learning models called ensemble learners. Explain that one particular ensemble learner, called a Random Forest, can improve upon a basic decision tree.

Before diving into Random Forests, take a moment to explain weak learners and ensemble learning:

Tell the class that if they took all the classification models they've used thus far and compared them, they'd find that some algorithms performed better than others, which is to be expected.

  * Even though some of the other algorithms performed worse, they were still able to execute independently and classify labels with decent accuracy.

  * Some algorithms actually fail at learning adequately. These algorithms/classifiers are considered weak learners.

  * Weak learners are a consequence of limited data to learn from. This may mean too few features, or the data provided does not allow for data points to be classified.

Provide more context around weak learners by defining them as algorithms/classifiers that are unable to learn from the data they are being provided accurately. This is why their predictions are only a little better than random chance. While the classifiers can make predictions, they are not representative of the relationship between inputs and targets.

* Weak learners are described as being only slightly better than random chance.

* Weak learners are still valuable in machine learning, because there are models that can combine many weak learners to create a more accurate and robust prediction engine. A single weak learner will make inaccurate predictions. Combined weak learners can perform just as well as any other strong learner.

* Ensemble learners are models that can combine weak learners to help improve accuracy and robustness, as well as decrease variance. GradientBoostedTree, XGBoost, and Random Forests are all examples of popular ensemble learning algorithms that can combine and use weak learners to make stronger predictions.

* Certain models, such as a decision tree, can be restricted or limited in ways that transform it into a weak learner. These limited decision trees only see a small percentage of the data and on their own aren't capable of making very good predictions about the entire dataset. However, just like a strong rope is woven together from many weaker threads, weak learners, like these limited decision trees, can be combined algorithmically to form a strong learner. This is the foundation of ensemble learning, taking many weak learners and combining them to form a strong predictive model.

* Decision trees are an example of a model that can be constrained to become a weak learner. In this case, the decision trees are restricted both in their depth and in the amount of data that they are allowed to see. These restricted trees (weak learners) tend to have very few branches on the decision tree. A single weak learner will make inaccurate and imprecise predictions because they are poor at learning adequately as a result of limited data, like too few features, or using data points that can't be classified.

Continue by introducing the random forest algorithm and highlight the following:

* Instead of having a single, complex tree like the ones created by decision trees, a random forest algorithm will sample the data and build several smaller, simpler decision trees.

* In a random forest, each tree is much simpler because it is built from a subset of the data.

* These simple trees are created by randomly sampling the data and creating a decision tree for just that small portion of data. This is known as a weak classifier, because it is only trained on a small piece of the original data, and by itself is only slightly better than a random guess. However, many _slightly better than average_ small decision trees can be combined to create a strong classifier, which has much better decision-making power.

* Some benefits of the random forest algorithm are:

  * It is robust against overfitting, because all of those weak classifiers are trained on different pieces of the data.

  * It can be used to rank the importance of input variables in a natural way.

  * It is robust to outliers and non-linear data. Random forest handles outliers by binning them. It is also indifferent to non-linear features.

Explain to students that in this demo, you are going to use the loan applications encoded dataset presented before. The goal of this demo is to predict fraudulent loan applications using a random forest.

Use the unsolved Jupyter notebook to live code the solution and highlight the following:

 In order to use the random forest implementation from `sklearn`, the `RandomForestClassifier` class from the `ensemble` module should be imported.

  ```python
  # Initial imports
  import pandas as pd
  from pathlib import Path
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.preprocessing import StandardScaler
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import classification_report
  %matplotlib inline

  # Needed for decision tree visualization
  import pydotplus
  from IPython.display import Image
  ```

The data is loaded into a Pandas DataFrame and then scaled and split into training and testing sets.

  ```python
  # Loading data
  file_path = Path("../Resources/credit_data.csv")
  df = pd.read_csv(file_path)
  df.head()

  # Split target column from dataset
  y = df['credit_risk']
  X = df.drop(columns='credit_risk')

  # Set Index
  X = X.set_index('id')

  # Encode the categorical variables using get_dummies
  X = pd.get_dummies(X)

  X.head()
  ```

Next, the data will be split into testing and trading datasets, and scaled using the `StandardScaler`.

  ```python
  # Splitting into Train and Test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

  # Creating StandardScaler instance
  scaler = StandardScaler()

  # Fitting Standard Scaler
  X_scaler = scaler.fit(X_train)

  # Scaling data
  X_train_scaled = X_scaler.transform(X_train)
  X_test_scaled = X_scaler.transform(X_test)
  ```

* When the random forest instance is created, there are two important parameters to set:

  * `n_estimators`: This is the number of random forests to be created by the algorithm. In general, a higher number makes the predictions stronger and more stable. However, a very large number can result in higher training time. A good approach is to start low and increase the number if the model performance is not adequate.

    * A [research study](https://doi.org/10.1007/978-3-642-31537-4_13) suggests that a range between `64` and `128` trees in a forest could be used for initial modelling.

  * `random_state`: This parameter defines the seed used by the random number generator. It is important to define a random state when comparing multiple models.

  ```python
  # Create a random forest classifier
  rf_model = RandomForestClassifier(n_estimators=100, random_state=1)
  ```

* Once the random forest model is created, it is fitted with the training data.

  ```python
  # Fitting the model
  rf_model = rf_model.fit(X_train_scaled, y_train)
  ```

* After fitting the model, some predictions are made using the scaled testing data.

  ```python
  # Making predictions using the testing data
  predictions = rf_model.predict(X_test_scaled)
  ```

* In order to evaluate the model the `classification_report` from `sklearn.metrics` is used.

  ```python
  # Displaying results
  print(classification_report(y_test, predictions))
  ```

After observing the results, it can be concluded that this model may not be the best one for preventing fraudulent loan applications. Explain to students that there are several strategies that may improve this model, such as:

* Reducing the number of features using principal component analysis (PCA).

* Creating new features based on new data from the problem domain.

* Increasing the number of estimators.

Finally, explain to students that a byproduct of the random forest algorithm is a ranking of feature importance (i.e., which features have the most impact on the decision).

* The `RandomForestClassifier` of `sklearn` provides an attribute called `feature_importances_`, where you can see which features were the most significant.

  ```python
  # Random Forests in sklearn will automatically calculate feature importance
  importances = rf_model.feature_importances_

  # Zip the feature importances with the associated feature name
  important_features = zip(X.columns,rf_model.feature_importances_)
  important_features

  # Create a dataframe of the important features
  importances_df = pd.DataFrame(important_features)

  # Rename the columns
  importances_df = importances_df.rename(columns={0: 'Feature', 1: 'Importance'})

  # Set the index
  importances_df = importances_df.set_index('Feature')

  # Sort the dataframe by feature importance
  importances_df = importances_df.sort_values(by='Importance',ascending=False)

  # Plot the top 10 most important features
  importances_df[0:10].plot(
      kind='barh',
      color='lightgreen',
      title= 'Feature Importance',
      legend=True)
  ```

* In this demo, it can be seen that the `age` of the person and the `amount` of the loan application are the more relevant features.

  ![](./Images/feature-importance.png)

* If we need to drop some features, analyzing feature importance could help to decide which features can be removed.

Answer any questions before moving on.

### 8. Student Do: Random Forests (15 min)

In this activity, students will build a Random Forest classifier that can be used to predict the loan status (approve or deny) given a set of input feature

Slack out the following files to the students.

**Files:**

[Instructions](./Activities/06_Stu_Random_Forest/README.md)

[loans.csv](./Activities/06_Stu_Random_Forest/Resources/loans.csv)

[Starter Code](./Activities/06_Stu_Random_Forest/Unsolved/random_forest.ipynb)

**Instructions:**

1. Read the data into a Pandas DataFrame.

2. Separate the features `X` from the target `y`. In this case, the loan status is the target.

3. Separate the data into training and testing subsets.

4. Scale the data using `StandardScaler`.

5. Import and instantiate an Random Forest classifier using sklearn.

6. Fit the model to the data.

7. Calculate the accuracy score using both the training and the testing data.

8. Make predictions using the testing data.

9. Generate the confusion matrix for the test data predictions.

10. Generate the classification report for the test data.

### 9. Instructor Do: Review Random Forests (10 min)

Open the provided solution code and review it with the students.

**Files:**

[Solution Code](./Activities/06_Stu_Random_Forest/Solved/random_forest.ipynb)

**Question**: What is the target column for the loans dataset?

**Answer**: The `status` column is the target column. It has two states, `approve` and `deny`. We drop it from the set of features before splitting the dataset into testing and training sets.

**Model**

* We create an instance of the `RandomForestClassifier` with a high number of estimators, `n_estimators=500`.

  ```python
  from sklearn.ensemble import RandomForestClassifier

  # Create a random forest classifier
  rf_model = RandomForestClassifier(n_estimators=500, random_state=1)
  ```

**Fit**

* We then fit the model to the scaled training features, `X_train_scaled`, and the training labels `y_train`.

  ```python
  # Fit the data
  rf_model.fit(X_train_scaled, y_train)
  ```

**Predict**

* A set of predicted values is generated using the scaled testing features, `X_test_scaled`.

  ```python
  # Make predictions using the test data
  y_pred = rf_model.predict(X_test_scaled)
  ```

* Looking at the classification report, the model has an overall accuracy score of 0.76 which is not bad. In the next lesson we will look at ways that we can improve this accuracy score.

Clarify any outstanding questions the students may have on the solution, then end class by transitioning into office hours.

---
## Open Office Hours

### Q&A and System Support

This is an opportunity to support students in any way that they require.

* Offer personalized support for students. (**Note:** feel free to pair individual students off with instructional staff as needed.)

* Ensure that everyone's system is functional and that students can access VS Code as well as their terminal instance to run the Python scripts.

* Possible dev environment issues this week might involve Canvas, downloading the activity files, using VS Code, and running Python files with Terminal or GitBash.

* Students might also encounter file structure issues. Suggest keeping activity files in a `FinTech_Workspace` folder organized by module and lesson.

---

© 2021 Trilogy Education Services, a 2U, Inc. brand. All Rights Reserved.
