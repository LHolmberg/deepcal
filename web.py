import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
import sys

def Classify(filename):
    temp2 = ""
    temp = 0
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
                temp = score
                temp2 = res

        if temp > 0.75:
            print('################')
            print('\n')
            print(temp2)
            print('\n')
            print('################')
        else:
            print('################')
            print('\n')
            print("unknown")
            print('\n')
            print('################')
    redirect("/b")

UPLOAD_FOLDER = 'pics/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def printfile(file):
    print(str(file))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/a")
def redirect_to_b():
    return redirect("/b")

@app.route("/b")
def handle_b():
    pass

@app.route('/classify', methods=['GET', 'POST'])
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
    return render_template('s.html')

if __name__ == "__main__":
    app.run()