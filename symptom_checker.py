import pandas as pd
import re
import spacy
from flask import Flask, request, jsonify
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)

# Load the data
data = pd.read_csv('C://Users//Harish//Downloads//symptomssingle.csv')

# Convert data to pandas DataFrame
import pandas as pd
df = pd.DataFrame(data)


# Define a function to separate symptoms and diseases from the text

'''This function takes a text input, extracts symptoms using regular expressions,
removes patterns representing symptoms from the original text, and returns a
tuple containing the joined symptoms and the processed disease text.'''

def separate_symptoms_and_diseases(text):
    symptoms = re.findall(r'{"symptoms":"(.*?)"}', text)
    disease = re.sub(r'(?:{"symptoms":".*?"},?)+', '', text).strip()
    disease = disease.replace('],', '').strip()  # Remove '],' from the disease name
    return ' '.join(symptoms), disease  # Join symptoms into a single string

# Apply the function to the data
data['symptoms_and_diseases'] = data['data'].apply(separate_symptoms_and_diseases)
data[['symptoms', 'disease']] = pd.DataFrame(data['symptoms_and_diseases'].tolist(), index=data.index)
data = data.drop(columns=['data', 'symptoms_and_diseases'])

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Preprocessing function
def preprocess(symptoms):
    processed_symptoms = []
    for symptom in symptoms:
        doc = nlp(symptom)
        processed_symptom = ' '.join(token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha)
        processed_symptoms.append(processed_symptom)
    return ' '.join(processed_symptoms)

# Preprocess the symptoms column
data['symptoms_preprocessed'] = data['symptoms'].apply(preprocess)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data['symptoms'], data['disease'], test_size=0.2, random_state=42)


# Create a pipeline for text classification
pipeline = Pipeline([
    ('count_vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Save the trained model
joblib.dump(pipeline, 'DiseasePredictionBasedonSymptoms.joblib')

# Load the saved model
loaded_pipeline = joblib.load('DiseasePredictionBasedonSymptoms.joblib')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get symptoms from the request
        symptoms = request.json.get('symptoms', [])

        # Join symptoms into a single string
        input_symptoms = ' '.join(symptoms.values())

        # Make a prediction using the loaded pipeline
        prediction = loaded_pipeline.predict([input_symptoms])[0]

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True,port=8086)
