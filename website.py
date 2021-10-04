# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 09:52:08 2021

@author: Wildan A Geraldine
"""

# My two categories
X = 'Covid-19'
Y = 'Normal'
Z = 'Pneumonia'
# Two example image for the website, them in the static directory next
# Where this file is and match the filenames here
sampleX = 'static/covid.png'
sampleY = 'static/normal.png'
sampleZ = 'static/pneumonia.png'

# Where I will keep user uploads
UPLOAD_FOLDER = 'static/infeksi/uploads/'
# Allowed files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Load operating system library
import os

# Website libraries
from flask import render_template
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

# Load math library
import numpy as np

# Load machine learning libraries
from tensorflow.keras.preprocessing import image
from keras.models import load_model
from tensorflow.compat.v1.keras.backend import set_session
import tensorflow as tf
import pickle
from sklearn.linear_model import LogisticRegression

# Create the website object
app = Flask(__name__)

def load_model_from_file():
    # Setup the machine learning session
    mySession = tf.compat.v1.keras.backend.get_session()
    set_session(mySession)
    myModel = load_model('model_pcm_xray.h5')
    myGraph = tf.compat.v1.get_default_graph()
    return (mySession,myModel,myGraph)

# Make sure nothing malicious is uploaded
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', myX=X, myY=Y, myZ=Z, mySampleX=sampleX, mySampleY=sampleY, mySampleZ=sampleZ)

@app.route('/gejala/',methods=['GET', 'POST'])
def gejala():
    if request.method == 'POST':
        gender = request.form["gender"]
        age = request.form['age']
        fever = request.form['fever']
        cough = request.form['cough']
        runny_noise = request.form['runny_noise']
        muscle_soreness = request.form['muscle_soreness']
        pneumonia = request.form['pneumonia']
        diarrhea = request.form['diarrhea']
        lung_infection = request.form['lung_infection']
        travel_history = request.form['travel_history']
        
        if gender == "Pria":
            gender = 1
        else:
            gender = 0
        
        if fever == "Yes_fever":
            fever = 1
        else:
            fever = 0
        
        if cough == "Yes_cough":
            cough = 1
        else:
            cough = 0
        
        if runny_noise == "Yes_runny_noise":
            runny_noise = 1
        else:
            runny_noise = 0
        
        if muscle_soreness == "Yes_muscle_soreness":
            muscle_soreness = 1
        else:
            muscle_soreness = 0
        
        if pneumonia == "Yes_pneumonia":
            pneumonia = 1
        else:
            pneumonia = 0
        
        if diarrhea == "Yes_diarrhea":
            diarrhea = 1
        else:
            diarrhea = 0
        
        if lung_infection == "Yes_lung_infection":
            lung_infection = 1
        else:
            lung_infection = 0
        
        if travel_history == "Yes_travel_history":
            travel_history = 1
        else:
            travel_history = 0
            
        print()
        print("Gender :",gender)
        print("Age :",age)
        print("Fever :",fever)
        print("Cough :",cough)
        print("Runni noise :",runny_noise)
        print("Muscle soreness :",muscle_soreness)
        print("Pneumonia :",pneumonia)
        print("Diarrhea :",diarrhea)
        print("Lung infection :",lung_infection)
        print("Travel history :",travel_history)
        
        return redirect(url_for('hasil_gejala', gender=gender, age=age, fever=fever, cough=cough, runny_noise=runny_noise, 
                                muscle_soreness=muscle_soreness, pneumonia=pneumonia, 
                                diarrhea=diarrhea, lung_infection=lung_infection, travel_history=travel_history))
     
    return render_template('gejala.html')


@app.route('/gejala/hasil/<gender><age><fever><cough><runny_noise><muscle_soreness><pneumonia><diarrhea><lung_infection><travel_history>', methods=['GET'])
def hasil_gejala(gender, age, fever, cough, runny_noise, muscle_soreness, 
                 pneumonia, diarrhea, lung_infection, travel_history):
        
        if request.method == 'GET':
            mySession = app.config['SESSION']
            myModel = app.config['MODEL']
            myGraph = app.config['GRAPH']
    
            with myGraph.as_default():
                set_session(mySession)
                with open('model_covid_symptoms.pkl', 'rb') as file:  
                    myModel = pickle.load(file)
                
                param = [[gender, age, fever, cough, runny_noise, muscle_soreness, 
                          pneumonia, diarrhea, lung_infection, travel_history]]    
                result = myModel.predict(param)
                score = myModel.predict_proba(param)
                results = []
                if result == 1 :
                    accuracy = str(score[0][1]*100)
                    accuracy = str('{0:.5}'.format(accuracy)+'%')
                    answer = "<div class='col text-center'><h5>Kemungkinan "+accuracy+" anda Positif Covid-19</h5></div>"
                else:
                    accuracy = str(score[0][0]*100)
                    accuracy = str('{0:.5}'.format(accuracy)+'%')
                    answer = "<div class='col text-center'><h5>Kemungkinan "+accuracy+" anda Negatif Covid-19</h5></div>"
                results.append(answer)
                return render_template('gejala.html', len=len(results), results=results)


# Define the view for the toop level page
@app.route('/infeksi/', methods=['GET', 'POST'])
def upload_file():
    #initial webpage load
    if request.method == 'GET':
        return render_template('infeksi.html', myX=X, myY=Y, myZ=Z, 
                               mySampleX=sampleX, mySampleY=sampleY, mySampleZ=sampleZ)
    else: # if request method == 'POST'
        if 'file' not in request.files:
            flash('No File part')
            return redirect(request.url)
        file = request.files['file']
        # if user dosen't select file, browser may also
        # submit an empty part without filename
        if file.filename == '':
            flash('Tidak ada foto yang dipilih !')
            return redirect(request.url)
        # if dosen't look like an image file
        if not allowed_file(file.filename):
            flash('Masukan file gambar dengan tipe exetensi '+str(ALLOWED_EXTENSIONS))
            return redirect(request.url)
        # When the user upload a file with good parameter
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))


@app.route('/infeksi/uploads/<filename>')
def uploaded_file(filename):
    test_image = image.load_img(UPLOAD_FOLDER+'/'+filename, target_size=(150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    
    mySession = app.config['SESSION']
    myModel = app.config['MODEL']
    myGraph = app.config['GRAPH']
    with myGraph.as_default():
        set_session(mySession)
        myModel = load_model('model_pcm_xray_v2.h5')
        result = myModel.predict(test_image)
        image_src = '/'+UPLOAD_FOLDER+'/'+filename
        results = []
        if result[0][0] == 1 :
            answer = "<div class='col text-center'><img width=150 height=150 src='"+image_src+"' class=img-'thumbnail'/><h5>Kemungkinan anda positif "+X+" "+str(result[0][0])+"</h5></div>"
        elif result[0][1] == 1 :
            answer = "<div class='col text-center'><img width=150 height=150 src='"+image_src+"' class=img-'thumbnail'/><h5>Paru-paru anda "+Y+" "+str(result[0][1])+"</h5></div>"
        elif result[0][2] == 1 :
            answer = "<div class='col text-center'><img width=150 height=150 src='"+image_src+"' class=img-'thumbnail'/><h5>Kemungkinan anda positif "+Z+" "+str(result[0][2])+"</h5></div>"
        results.append(answer)
        return render_template('infeksi.html', myX=X, myY=Y, myZ=Z, mySampleX=sampleX, mySampleY=sampleY, mySampleZ=sampleZ, len=len(results), results=results)
    
    
def main():
    
    (mySession,myModel,myGraph) = load_model_from_file()
    
    app.config['SECRET_KEY'] = 'super secret key'
    
    app.config['SESSION'] = mySession
    app.config['MODEL'] = myModel
    app.config['GRAPH'] = myGraph
    
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB upload limit
    app.run()
    

# Create a running list of result
results = []

# Launch Everyting
main()

