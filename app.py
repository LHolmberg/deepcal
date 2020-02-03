import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
import sys
import json
import requests

#For the food database api
APPID = "f1b23fbd"
API_KEY = "7b05c0f405e06d6d6f3af6938505daab"

class Data:
    def __init__(self):
        self.s = "unknown"

    def SetData(self, s):
        self.s = s
        
    def GetData(self):
        return self.s

x = Data()

def Classify(filename):
    x.SetData("unknown")
    image_file = tf.gfile.FastGFile(filename, 'rb')
    data = image_file.read()
    classes = [line.rstrip() for line in tf.gfile.GFile("labels.txt")]

    with tf.gfile.FastGFile("graph.pb", 'rb') as inception_graph:
        definition = tf.GraphDef()
        definition.ParseFromString(inception_graph.read())
        _ = tf.import_graph_def(definition, name='')
    with tf.Session() as session:
        tensor = session.graph.get_tensor_by_name('final_result:0')
        result = session.run(tensor, {'DecodeJpeg/contents:0': data})

        top_results = result[0].argsort()[-len(result[0]):][::-1]
        for type in top_results:
            res = classes[type]
            score = result[0][type]
            if(score > 0.75):
                x.SetData(res)


UPLOAD_FOLDER = 'pics/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def printfile(file):
    print(str(file))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            Classify("pics/" + filename)
            return redirect("/eval")
    return render_template('home.html')

@app.route('/eval')
def eval():
    if x.GetData() != "unknown" and x.GetData() != "Unknown":
        r = requests.get("https://api.edamam.com/api/food-database/parser?nutrition-type=logging&ingr="+ x.GetData() + "&app_id=" + APPID +"&app_key=" + API_KEY)
        nutrients = r.json()["hints"][0]["food"]["nutrients"]
        return render_template('eval.html', data=x.GetData(), protein = int(nutrients["PROCNT"]), fat=int(nutrients["FAT"]), carbs = int(nutrients["CHOCDF"]))
    else:
        return render_template('eval.html', data=x.GetData(), protein = 0, fat = 0, carbs = 0)

if __name__ == "__main__":
    app.run()
