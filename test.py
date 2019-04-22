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



APP_ROOT = os.path.dirname(os.path.abspath(__file__))
path = Path(APP_ROOT)
counter = 0
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
	
extract_features('testp.wav',counter)
