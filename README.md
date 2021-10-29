# Machine-Learning

## Classification_by_Perceptron_Algorithm

The Perceptron is a binary classification problem linear machine learning method.
It is said to be one of the first and simplest forms of artificial neural networks. It is somewhat "deep" learning, but it is a necessary building component.
It can swiftly learn a linear separation in feature space for two-class classification problems, similar to logistic regression, however unlike logistic regression, it learns using the stochastic gradient descent optimization process and does not forecast calibrated probability.

It is made up of a single node or neuron that takes in a row of data and predicts a class label. This is accomplished by computing the weighted total of the inputs as well as a bias (set to 1). The activation is the weighted sum of the model's input.

Activation = Weights * Inputs + Bias

If the activation is above 0.0, the model will output 1.0; otherwise, it will output 0.0.

Predict 1: If Activation > 0.0

Predict 0: If Activation <= 0.0

The model's coefficients are known as input weights, and they are trained using the stochastic gradient descent optimization process.

Each batch, model weights are updated with a tiny amount of the error, which is regulated by a hyperparameter called the learning rate, which is normally set to a low value. This is done to guarantee that learning does not happen too rapidly, resulting in a potentially lower skill model, which is known as premature convergence of the optimization (search) technique for the model weights.

## Fake_Currency_Detection

Steps used:

1.Loading data

2.Visualization of target distribution and pairplot

3.Splitting data into test and train data

4. Using confusion matrix 

5.Testing data using selecting random value from dataset

Data link: https://raw.githubusercontent.com/Akshatpattiwar512/Datasets/master/banknote_authentication.csv

## Glass_Type_Classification

In this project , visualization classification is done on the basis of type of elements and the type of glass

### Visualizations used: 

1.Boxplot

2.pairplot

3.Univariate plot

4.Heatmap

### Preparation of data: 

1.Data cleaning

2.Removing multiple outliers

### Distribution plotting of types

### Splitting variation dataset

### Transforming data

### Evaluation of algorithm

1.Dimensionality reduction: XGBoost, PCA

2.Algorithm Tuning: Tuning random forest

### Diagnose overfitting by plotting the learning and validation curves

Dataset link: https://www.kaggle.com/uciml/glass

## Iris_dataset_ML

Dataset: Iris.csv
