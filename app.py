import pickle
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route("/")
def hello_world():
    return render_template("index.html") 

@app.route('/prediction')
def hello():
    return render_template("predict.html") 

@app.route('/analysis')
def analysis():
    return render_template("Analysis.html")      
    
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = [float(x) for x in request.form.values()]
    int_features=request.form['Murder'],request.form['Theft'],request.form['Social-Crime']
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('predict.html', prediction_text='Crime rate is :{}'.format(output))     


if __name__=="__main__":
    app.run(debug=True, port=8000)