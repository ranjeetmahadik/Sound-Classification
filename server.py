from flask import Flask,request,render_template,jsonify
import json
import os,sys
from werkzeug.utils import secure_filename
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import urllib.request
from fastai import *

from fastai.vision import *
print("started")

model_path = "https://www.dropbox.com/s/dik331qi0d6zd2k/export.pkl?raw=1"
export_file_name = "export.pkl"
classes = ['air_conditioner','car_horn','children_playing','dog_bark','drilling','engine_idle','gun_shot','jackhammer','siren','street_music']


app = Flask(__name__)
counter = 0

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
#C:\Users\Ranjeet\Downloads\Ranjeet Stuff\Implementation\fastai-v3-master\fastai-v3-master\app
target = os.path.join(APP_ROOT,'audio/')
path = Path(APP_ROOT)
model = os.path.join(APP_ROOT,'models/')


@app.route("/")
def index():
	#print(APP_ROOT)
	#print(target)
	return render_template("index.html")


def downloader(url,dest):
	print("downloading")
	with urllib.request.urlopen(url) as response:
		data = response.read()
	with open(dest,'wb') as f: f.write(data)
	print("downloading done")	
def setup_learner():
	#downloader(model_path,model+export_file_name)
	try:
		learn = load_learner(model,export_file_name)
		return learn
	except:
		print("error")

def extract_features(filename,counter):
    new_filename = filename.split(".")[0]
    samples,sample_rate = librosa.load(filename)
    fig = plt.figure(figsize=[0.72,0.72])
    ax  = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    file_name = path/Path('spectograms/')/Path(filename).name.replace('.wav','.png')
    new_file = path/Path('spectograms/')/Path(file_name).name.replace(new_filename,str(counter))
    S = librosa.feature.melspectrogram(y=samples,sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    plt.savefig(new_file, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()
    print(new_file)
    counter+=1
    return str(file_name)
	
learn = setup_learner()

@app.route('/analyze', methods=['POST'])
def analyze():
	if request.method == "POST":
		filename = request.files['file'].filename
		file = request.files['file']
		path = os.getcwd()
		filename = secure_filename(file.filename)
		#print(filename)
		file.save(filename)
		shutil.move(APP_ROOT+"/"+filename,target)
		image_path = extract_features(target+filename,counter)
		img = open_image(image_path)
		prediction = learn.predict(img)[0]
	print(image_path)
	return jsonify({"result":str(prediction)})
	
if __name__ == "__main__":
	app.run(host='0.0.0.0',port=80)
