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

    print("model")

    # print the request data body
    if request.json['modelNo'] == '2':

        tableData = (request.json['tableData'])

        # for each row in the tableData, run on runModel4, and add the result in the 'results' key of the row
        for row in tableData:
            row['result'] = runModel2(row)

        return {
            'result': tableData
        }

    # print the request data body
    if request.json['modelNo'] == '4':

        tableData = (request.json['tableData'])

        # for each row in the tableData, run on runModel4, and add the result in the 'results' key of the row
        for row in tableData:
            row['result'] = runModel4(row)

        return {
            'result': tableData
        }

     # print the request data body
    if request.json['modelNo'] == '3':

        tableData = (request.json['tableData'])

        # for each row in the tableData, run on runModel4, and add the result in the 'results' key of the row
        for row in tableData:
            row['result'] = runModel3(row)

        return {
            'result': tableData
        }

    if request.json['modelNo'] == '1':

        tableData = (request.json['tableData'])

        # for each row in the tableData, run on runModel4, and add the result in the 'results' key of the row
        for row in tableData:
            row['result'] = runModel1(row)

        return {
            'result': tableData
        }


def runModel1(data):
    import pickle
    import pandas as pd

    """
    Loads the saved model and preprocessing components, processes the input data, and predicts the target.

    Parameters:
    data (dict): Dictionary containing the input data (e.g., new customer data).

    Returns:
    float: The predicted value for y (continuous).
    """
    # Load the model and preprocessing components from the pickle file
    with open('models/linear-regression/linear_regression.pkl', 'rb') as f:
        components = pickle.load(f)

    model = components['model']
    scaler = components['scaler']
    label_encoder = components['label_encoder']

    # Convert the input data to a DataFrame
    df = pd.DataFrame([data])

    # Handle unseen categories for categorical columns (e.g., 'job', 'education', 'contact', etc.)
    def safe_transform(col, encoder):
        """Safely transform a column value, handling unseen categories"""
        if col in encoder.classes_:
            return encoder.transform([col])[0]
        else:
            # Handle unseen label (you can choose another strategy like assigning a default value)
            return -1

    # Preprocess the data like we did for the training dataset
    df['job_numeric'] = df['job'].apply(
        lambda x: safe_transform(x, label_encoder))
    df['education_numeric'] = df['education'].apply(
        lambda x: safe_transform(x, label_encoder))
    df['month_numeric'] = df['month'].apply(
        lambda x: safe_transform(x, label_encoder))
    df['contact_numeric'] = df['contact'].apply(
        lambda x: safe_transform(x, label_encoder))
    df['poutcome_numeric'] = df['poutcome'].apply(
        lambda x: safe_transform(x, label_encoder))
    df['marital_numeric'] = df['marital'].apply(
        lambda x: safe_transform(x, label_encoder))

    # Handle 'yes'/'no' columns: 'default', 'housing', 'loan' (encode them as 1/0)
    df['default'] = df['default'].map({'yes': 1, 'no': 0})
    df['housing'] = df['housing'].map({'yes': 1, 'no': 0})
    df['loan'] = df['loan'].map({'yes': 1, 'no': 0})

    # Select the same features as in the training set (8 features in total)
    df_selected = df[['age', 'job_numeric', 'marital_numeric',
                      'education_numeric', 'default', 'balance', 'housing', 'month_numeric']]

    # Scale the features using the previously trained scaler
    df_scaled = scaler.transform(df_selected)

    # Make the prediction using the trained model
    prediction = model.predict(df_scaled)

    return 'No' if prediction[0] < 0.5 else 'Yes'


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


def runModel3(data):

    import pickle
    import pandas as pd

    """
    Loads the saved model and preprocessing components, processes the input data, and predicts the target.

    Parameters:
    data (dict): Dictionary containing the input data (e.g., new customer data).

    Returns:
    str: The predicted 'y' value ('yes' or 'no').
    """
    # Load the model and preprocessing components from the pickle file
    with open('models/logistic-regression/logistic_regression.pkl', 'rb') as f:
        components = pickle.load(f)

    model = components['model']
    scaler = components['scaler']
    label_encoder = components['label_encoder']

    # Convert the input data to a DataFrame
    df = pd.DataFrame([data])

    # Handle unseen categories for categorical columns (e.g., 'job', 'education', 'contact', etc.)
    def safe_transform(col, encoder):
        """Safely transform a column value, handling unseen categories"""
        if col in encoder.classes_:
            return encoder.transform([col])[0]
        else:
            # Handle unseen label (you can choose another strategy like assigning a default value)
            return -1

    # Preprocess the data like we did for the training dataset
    df['job_numeric'] = df['job'].apply(
        lambda x: safe_transform(x, label_encoder))
    df['education_numeric'] = df['education'].apply(
        lambda x: safe_transform(x, label_encoder))
    df['month_numeric'] = df['month'].apply(
        lambda x: safe_transform(x, label_encoder))
    df['contact_numeric'] = df['contact'].apply(
        lambda x: safe_transform(x, label_encoder))
    df['poutcome_numeric'] = df['poutcome'].apply(
        lambda x: safe_transform(x, label_encoder))

    # Handle 'yes'/'no' columns: 'default', 'housing', 'loan' (encode them as 1/0)
    df['default'] = df['default'].map({'yes': 1, 'no': 0})
    df['housing'] = df['housing'].map({'yes': 1, 'no': 0})
    df['loan'] = df['loan'].map({'yes': 1, 'no': 0})

    # Drop original categorical columns (same as during training)
    df = df.drop(columns=['age', 'job', 'marital',
                 'education', 'month', 'contact', 'poutcome'])

    # Scale the features using the previously trained scaler
    df_scaled = scaler.transform(df)

    # Make the prediction using the trained model
    prediction = model.predict(df_scaled)

    # Convert prediction (1 or 0) back to 'yes' or 'no'
    return 'Yes' if prediction[0] == 1 else 'No'


@app.route('/runModel2')
def runModel2(data):
    import pickle
    import pandas as pd

    model = pickle.load(
        open('models/binary-classification/binary_classification_model.pkl', 'rb'))
    feature_columns = pickle.load(
        open('models/binary-classification/model_columns.pkl', 'rb'))

    # Convert input data to a DataFrame
    input_df = pd.DataFrame([data])

    # Apply one-hot encoding to match training columns
    input_df = pd.get_dummies(input_df)

    # Add missing columns from the training set
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder the columns to match the training set
    input_df = input_df[feature_columns]

    # Make a prediction
    prediction = model.predict(input_df)

    # Convert the prediction to "Yes" or "No"
    result = "Yes" if prediction[0] == 1 else "No"

    return result


app.run(debug=True)
