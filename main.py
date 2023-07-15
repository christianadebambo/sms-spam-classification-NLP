import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import pickle
import re
import warnings

warnings.filterwarnings('ignore')

sns.set_style("whitegrid")

# Read the dataset
messages = pd.read_csv("smsspamcollection/SMSSpamCollection", sep='\t', names=["label", "messages"])

# Print basic information about the dataset
print(messages.head())
print(messages.describe())

# Exploratory data analysis
print(messages.groupby('label').describe())
messages['length'] = messages['messages'].apply(len)
print(messages.head())

# Visualize message length distribution
messages['length'].plot.hist(bins=70)
plt.show()

# Define optimized text processing function
def text_process(mess):
    # Remove punctuation using regular expressions
    no_punc = re.sub(r'[^\w\s]', '', mess)
    
    # Remove stopwords using NLTK
    stop_words = set(stopwords.words('english'))
    clean_words = [word for word in no_punc.lower().split() if word not in stop_words]
    
    return clean_words

# Apply text processing function to the messages
messages['processed'] = messages['messages'].apply(text_process)

# Initialize the pipeline
def identity_func(x):
    return x

pipeline = Pipeline([
    ('vectorizer', CountVectorizer(analyzer='word', tokenizer=identity_func, preprocessor=identity_func, token_pattern=None)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

# Split the data into training and testing sets
msg_train, msg_test, label_train, label_test = train_test_split(messages['processed'], messages['label'], test_size=0.3)

# Train the pipeline
pipeline.fit(msg_train, label_train)

# Make predictions on the test set
predictions = pipeline.predict(msg_test)

# Evaluate the model
print(classification_report(label_test, predictions))
print(confusion_matrix(label_test, predictions))

# Perform cross-validation
cv_scores = cross_val_score(pipeline, messages['processed'], messages['label'], cv=5)
print("Cross-validation scores:", cv_scores)

# Save the pipeline
with open('models/pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# Load the saved pipeline
loaded_pipeline = pickle.load(open('models/pipeline.pkl', 'rb'))

# Prompt the user for a message
text_input = input("Enter your message: ")

# Preprocess the input message
cleaned_input = text_process(text_input)

# Transform the preprocessed input message
transformed_input = loaded_pipeline.named_steps['vectorizer'].transform([cleaned_input])
preprocessed_input = loaded_pipeline.named_steps['tfidf'].transform(transformed_input)

# Make predictions on the preprocessed input message
prediction = loaded_pipeline.named_steps['classifier'].predict(preprocessed_input)

print("\nThe message is: {}\n".format(prediction[0]))