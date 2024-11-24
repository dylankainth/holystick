from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def hello():
     return render_template('index.html', utc_dt="tetet")

app.run(debug=True)
