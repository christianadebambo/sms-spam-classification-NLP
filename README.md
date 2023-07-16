# SMS Spam Classification

This project contains a machine learning pipeline (CountVectorizer + TfidfTransformer + MultinomialNB) for SMS spam classification. The pipeline preprocesses text messages, trains a classifier, and makes predictions on new input messages. It also includes exploratory data analysis and evaluation metrics for the model's performance.

## Code Overview

- **Imports**: The necessary libraries and modules are imported, including pandas, matplotlib, seaborn, nltk, sklearn, pickle, re, and warnings. These libraries are commonly used for data manipulation, visualization, natural language processing (NLP), machine learning, and serialization.
  
- **Dataset Loading**: The code reads a dataset from a CSV file called "SMSSpamCollection" using pandas and stores it in a DataFrame called messages. The dataset contains two columns: "label" (spam or ham) and "messages" (text messages). This dataset is used for training and evaluation.
  
- **Exploratory Data Analysis**: Basic exploratory data analysis is performed on the dataset using pandas. The first few rows of the dataset, descriptive statistics, and group statistics based on the "label" column are printed. Additionally, a new column called "length" is added to the DataFrame, representing the length of each message. These statistics help in understanding the characteristics of the dataset.

- **Message Length Visualization**: The code uses matplotlib to create a histogram plot that visualizes the distribution of message lengths. This provides insights into the length distribution of spam and ham messages.

- **Text Preprocessing Function**: The code defines a function called text_process that preprocesses the text data. It uses regular expressions (re) to remove punctuation and nltk to remove stopwords (common words that do not contribute much to the classification). The function applies these preprocessing steps to each message and returns a list of clean words.

- **Text Processing**: The text_process function is applied to the "messages" column of the messages DataFrame, and the preprocessed messages are stored in a new column called "processed." This step prepares the text data for training the machine learning model.

- **Pipeline Initialization**: The code initializes a machine learning pipeline using sklearn.pipeline.Pipeline. The pipeline consists of three main steps: tokenizing and vectorizing the text using CountVectorizer, transforming the vectorized text using TfidfTransformer, and training a MultinomialNB (Naive Bayes) classifier. These steps are encapsulated in the pipeline to ensure consistent and efficient processing.

- **Data Splitting**: The data is split into training and testing sets using sklearn.model_selection.train_test_split. The messages from the "processed" column are split into a training set (msg_train) and a testing set (msg_test), while the corresponding labels from the "label" column are split into label_train and label_test.

- **Pipeline Training**: The pipeline is trained on the training data using the fit method. This step involves tokenizing, vectorizing, transforming, and training the classifier on the preprocessed messages.

- **Prediction**: Predictions are made on the test set using the predict method of the pipeline. The predictions represent the spam or ham labels assigned by the trained model.

- **Model Evaluation**: The code evaluates the performance of the model by printing a classification report and a confusion matrix. The classification report shows metrics such as precision, recall, F1-score, and support for each class. The confusion matrix displays the number of true positives, true negatives, false positives, and false negatives.

- **Cross-Validation**: Cross-validation is performed using sklearn.model_selection.cross_val_score. This technique provides an estimate of the model's performance on different subsets of the data by splitting the data into multiple folds and evaluating the model on each fold.

- **Pipeline Serialization**: The trained pipeline is saved to a file using the pickle module. Serialization allows the model to be reused or deployed in different environments without the need to retrain it.

- **Pipeline Loading**: The saved pipeline is loaded from the file using pickle.load.

- **User Interaction**: The code prompts the user to enter a message for classification. The user's input is stored in the text_input variable.

- **Message Preprocessing**: The input message is preprocessed using the text_process function to remove punctuation and stopwords. This step ensures consistency in the processing pipeline.

- **Input Transformation**: The preprocessed input message is transformed using the loaded pipeline's vectorizer and tf-idf transformer. These transformations convert the input message into a numerical representation compatible with the trained model.

- **Prediction on Input**: The transformed input message is used to make predictions using the loaded pipeline's classifier. The predicted label for the input message is stored in the prediction variable.

- **Output**: The code prints the predicted label for the input message.

## Installation

To use this code, you need to have Python installed. Clone this repository to your local machine using the following command:

## Requirements

Install the required libraries using the [pip](https://pip.pypa.io/en/stable/) package manager.

For pip version 19.1 or above:

~~~bash
pip install -r requirements.txt --user
~~~

or

~~~bash
pip3 install -r requirements.txt --user
~~~

## Usage

Run the following command:

~~~bash
python main.py
~~~
