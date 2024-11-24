# Load the saved model and feature columns
import pickle
import pandas as pd

model = pickle.load(
    open('models/binary-classification/binary_classification_model.pkl', 'rb'))
feature_columns = pickle.load(
    open('models/binary-classification/model_columns.pkl', 'rb'))

# Example Input Data (Replace with actual input values)
input_data = {
    'feature_0': 0.5,
    'feature_1': -1.2,
    'feature_2': 3.0,
    'feature_3': 2.1,
    'feature_4': -0.7,
    'feature_5': 0.0,
    'feature_6': 1.5,
    'feature_7': 0.3,
    'feature_8': -1.8,
    'feature_9': 0.9
}

# Convert input to a DataFrame
input_df = pd.DataFrame([input_data])

# Make a prediction
prediction = model.predict(input_df)

# Convert the prediction to "Yes" or "No"
result = "Yes" if prediction[0] == 1 else "No"
print("Prediction:", result)
