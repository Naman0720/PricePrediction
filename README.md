# HomePricePredictor
## Description
This project aims to build a data science model that predicts US home prices based on key factors. The model utilizes publicly available data and employs various regression algorithms to analyze the impact of these factors on home prices over the last 20 years. The S&P Case-Schiller Home Price Index is used as a proxy for home prices.
## Table of Contents
#### 1.Installation
#### 2.Usage
#### 3.Data
#### 4.Model Training
#### 5.Evaluation
#### 6.Results
#### 7.Contributing

## Installation
To run this project locally, you need to have Python installed. Clone the repository from GitHub and install the required dependencies using the following command:
Copy

```bash
pip install -r requirements.txt
```
## Usage

To use the US Home Price Prediction Model, follow these steps:
1. Ensure that you have the necessary dataset in CSV format. The dataset should include the key factors that influence home prices, as well as the corresponding home prices based on the S&P Case-Schiller Home Price Index.
2. Preprocess the data by handling missing values and outliers, and perform any necessary feature engineering.
3. Split the data into training and testing sets.
4. Train the regression model using one of the available algorithms, such as Linear Regression, Random Forest, Support Vector Regression (SVR), or Decision Tree Regression.
5. Evaluate the trained model using appropriate metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared.
6. Interpret the model coefficients or feature importance scores to understand the impact of each factor on home prices.
7. Use the trained model to make predictions on new data if needed.

## Data

The dataset used in this project should include the following columns:
TOT_POP: Total population
UNEMP_RATE: Unemployment rate
UND_CONST: Under construction housing units
CONS_SENT: Consumer sentiment index
HOM_PRC: Home prices based on the S&P Case-Schiller Home Price Index

Ensure that the dataset is in CSV format and is properly formatted before using it for training the model.

## Model Training

To train the US Home Price Prediction Model, follow these steps:
Load the dataset using the pd.read_csv() function.
Preprocess the data by handling missing values and outliers, and perform any necessary feature engineering.
Split the data into training and testing sets using the train_test_split() function from the sklearn.model_selection module.
Choose a regression algorithm such as Linear Regression, Random Forest, SVR, or Decision Tree Regression.
Train the chosen model using the training data.

## Evaluation

To evaluate the performance of the US Home Price Prediction Model, follow these steps:
Make predictions on the testing set using the trained model.
Calculate the Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared metrics using the predicted values and the actual values from the testing set.
Interpret the model coefficients or feature importance scores to understand the impact of each factor on home prices.

## Results

The US Home Price Prediction Model was evaluated using various regression algorithms. The results of each algorithm are as follows:

#### Linear Regression:
Mean Squared Error: 57.15221822655363 
Root Mean Squared Error: 7.559908612314942
R-squared: 0.9770764259624048

#### Random Forest:
Mean Squared Error: 3.242576417421159
Root Mean Squared Error: 1.8007155292886101
R-squared: 0.9986994128507373

#### SVR:
Mean Squared Error: 1956.3293323073865
Root Mean Squared Error: 44.23041184871995
R-squared: 0.21532248996362846

#### Decision Tree:
Mean Squared Error: 8.643748035087718
Root Mean Squared Error: 2.9400251759275324
R-squared: 0.9965330199912943

### Based on the evaluation metrics, the Random Forest algorithm performed the best in predicting home prices.

## Contributing
Contributions to this project are welcome. If you have any suggestions or improvements, please open an issue or submit a pull request.
