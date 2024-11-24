from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


models = [
    {
        'name': 'Linear Regression',
        'environmental-impact': 'Low',
        'accuracy': 'High',
        'interpretability': 'High',
        'complexity': 'Low',
        'cost': 'Low',
        'description': 'Linear regression is a linear approach to modeling the relationship between a scalar response and one or more explanatory variables. The case of one explanatory variable is called simple linear regression.',
        'modelNo': 1
    },
    {
        'name': 'Hollistic Ai',
        'environmental-impact': 'Low',
        'accuracy': 'High',
        'interpretability': 'High',
        'complexity': 'Low',
        'cost': 'Low',
        'description': 'Holistic AI is a form of AI that is designed to mimic the holistic thinking and problem-solving skills of humans. It is a type of AI that is designed to be able to think and reason in a way that is similar to the way that humans do.',
        'modelNo': 2
    },
    {
        'name': 'Logistic Regression',
        'environmental-impact': 'Low',
        'accuracy': 'High',
        'interpretability': 'High',
        'complexity': 'Low',
        'cost': 'Low',
        'description': 'Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist. In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model (a form of binary regression).',
        'modelNo': 3
    },
    {
        'name': 'Random Forest',
        'environmental-impact': 'Low',
        'accuracy': 'High',
        'interpretability': 'High',
        'complexity': 'Low',
        'cost': 'Low',
        'description': 'Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean/average prediction (regression) of the individual trees.',
        'modelNo': 4
    }
]


@ app.route('/2')
def page2():
    return render_template('page2.html', models=models)


@app.route('/3')
def page3():
    modelNo = request.args.get('modelNo')

    # get the model with modelNo of modelNo
    model = [model for model in models if model['modelNo'] == int(modelNo)][0]
    return render_template('page3.html', modelNo=modelNo, model=model)


@app.route('/runModel', methods=['POST'])
def runModel():

    # print the request data body
    if request.json['modelNo'] == '4':

        tableData = (request.json['tableData'])

        # for each row in the tableData, run on runModel4, and add the result in the 'results' key of the row
        for row in tableData:
            row['result'] = runModel4(row)

        return {
            'result': tableData
        }


def runModel4(data):

    import pickle
    import pandas as pd

    # Load the trained Random Forest model
    model = pickle.load(
        open('models/random-forest/random_forest_model.pkl', 'rb'))

    # Load the feature columns (to keep the input data in the right format)
    model_columns = pickle.load(
        open('models/random-forest/model_columns.pkl', 'rb'))

    # Convert input data to DataFrame for consistency
    input_df = pd.DataFrame([data])

    # Apply one-hot encoding to input data to match training columns
    input_df = pd.get_dummies(input_df)

    # Add missing columns (those that were present during training but absent in input data)
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder the columns to match the original training set
    input_df = input_df[model_columns]

    # Make the prediction
    prediction = model.predict(input_df)

    if prediction[0] == 1:
        return 'Yes'
    else:
        return 'No'

# @app.route('/runModel2')
# def runModel2():
#     # Load the saved model and feature columns
#     import pickle
#     import pandas as pd

#     model = pickle.load(
#         open('models/binary-classification/binary_classification_model.pkl', 'rb'))
#     feature_columns = pickle.load(
#         open('models/binary-classification/model_columns.pkl', 'rb'))

#     # Example Input Data (Replace with actual input values)
#     input_data = {
#         'feature_0': 0.5,
#         'feature_1': -1.2,
#         'feature_2': 3.0,
#         'feature_3': 2.1,
#         'feature_4': -0.7,
#         'feature_5': 0.0,
#         'feature_6': 1.5,
#         'feature_7': 0.3,
#         'feature_8': -1.8,
#         'feature_9': 0.9
#     }

#     # Convert input to a DataFrame
#     input_df = pd.DataFrame([input_data])

#     # Add missing columns (those that were present during training but absent in input data)
#     for col in feature_columns:
#         if col not in input_df.columns:
#             input_df[col] = 0

#     # Reorder the columns to match the original training set
#     input_df = input_df[feature_columns]

#     # Make a prediction
#     prediction = model.predict(input_df)


#     # Convert the prediction to "Yes" or "No"
#     result = "Yes" if prediction[0] == 1 else "No"
#     return ("Prediction:", result)
app.run(debug=True)
