import os
from flask import Flask, flash, request, redirect, url_for,render_template
from werkzeug.utils import secure_filename
import pickle
import pandas as pd
import sklearn
app = Flask('Breast Cancer Classification')

ALLOWED_EXTENSIONS = {'csv','xls'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/',methods=['POST','GET'])
def show_predict_stock_form():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # filename = secure_filename(file.filename)
            file.save(os.path.join('./csvs', file.filename))
            return redirect(url_for('results',filename=file.filename))
    return render_template('predictionform.html')


@app.route('/results', methods=['POST','GET'])
def results():
    filename = request.args.get("filename")

    df = pd.read_csv('./csvs/'+filename)
    df = df.dropna(axis=1)
    X = df.iloc[:, 2:31].values
    # write your function that loads the model
    model = pickle.load(open('finalmodel.sav', 'rb'))  # you can use pickle to load the trained model
    malignant_Predicted = model.predict(X)
    return render_template('resultsform.html', predicted=malignant_Predicted)


app.run("localhost", "9999", debug=True)
