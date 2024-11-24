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


app.run(debug=True)
