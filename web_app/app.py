# Importing librairies
from flask import Flask,render_template,url_for,request, json
from utils import prediction

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

	if request.method == 'POST':
		message = request.form['message']
		my_prediction = prediction(message)
	return render_template('home.html',prediction = my_prediction,question = message)

if __name__ == '__main__':
	app.run(debug=True)