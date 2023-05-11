#Importing required libraries
from __future__ import division
from turtle import pu
from django.shortcuts import render
from flask import Flask, render_template, Response, jsonify, send_file
from six.moves.queue import Queue, Empty
import mediapipe as mp
#from model_test_code import *
import time
import cv2
import pyaudio
#from google.cloud import speech
import sys
import re
import os
import requests
import flask
import numpy as np
import joblib
import pickle
from cv2 import imwrite
from autocorrect import Speller
spell = Speller(lang='en')
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms
app = Flask(__name__)
ans = []
count = 0


# Function to find landmarks(Skeleton) on hand using  mediapipe
def data_clean(landmark):
    data = landmark[0]
    try:
        data = str(data)
        data = data.strip().split('\n')
        garbage = [
            'landmark {', '  visibility: 0.0', '  presence: 0.0', '}']
        without_garbage = []
        for i in data:
            if i not in garbage:
                without_garbage.append(i)
        clean = []
        for i in without_garbage:
            i = i.strip()
            clean.append(i[2:])
        for i in range(0, len(clean)):
            clean[i] = float(clean[i])
        return clean
    except:
        return -1


#Final answer output storage variable
result = ''

# Function to generate frames i.e. continous live video recording using web cam
def generate_frames():
    import cv2
    temp = []
    ct = 0
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    # For webcam input:
    hands = mp_hands.Hands(
        min_detection_confidence=0.7, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    space_flag = 0
    while cap.isOpened():
        success, image = cap.read()
        image = cv2.flip(image, 1)
        if not success:
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            space_flag = 1
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cl = data_clean(results.multi_hand_landmarks)
            cleaned_landmark = [cl]
            if cleaned_landmark:
                # Path of file where .pkl file is stored which contains all weights info of Neural Network
                filename = "SVM+Final_Dataset.pkl"
                clf = joblib.load(filename)
                y_pred = clf.predict(cleaned_landmark)
                temp.append(y_pred)
                if len(temp) >= 30:
                    d = dict()
                    for i in temp:
                        
                        if i[0] in d:
                            d[i[0]] += 1
                        else:
                            d[i[0]] = 1
                    maxi = max(d.values())
                    ans = y_pred
                    for g in d:
                        if d[g] == maxi:
                            ans = g
                    # Add the predicted output to result string 
                    global result
                    print('ans',ans)
                    if str(ans)=='O_OR_0':
                        result+='O'
                    elif str(ans)=='V_OR_2':
                        result+='V'
                    elif str(ans)=='W_OR_6':
                        result+='W'
                    else:
                        result+=str(ans)

                    
                    temp = []

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            # print(result)
            yield(b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            # "No Hand detected.." so give normal video  and also add space to result by default (only one space) and just dont show hand for 3-4 seconds and space will be added automatically
            if space_flag == 30:
                space_flag = 0
                result += '  '
                #result = spell(result)
            space_flag += 1
            
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            # print(result)
            yield(b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    print(result)


#Sign to Text Conversion 
@app.route('/Sign_To_Text', methods=['GET', 'POST'])
def index():
    return render_template('Sign_To_Text.html', ans=ans)

# Live video capturing  and showing mediapipe results.
@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
# Live video capturing  and showing mediapipe results.

@app.route('/mediapipedemo')
def mediapipedemo():
    return render_template('MediapipeDemo.html')

# Gets the result 
@app.route('/answer', methods=['GET'])
def answer():
    print('Result is', result)
    return jsonify(result)

# Clears the output to empty string 
@app.route('/clearoutput', methods=['GET', 'POST'])
def clearoutput():
    global result
    result = ' '
    return jsonify(result)

# Gives information about project 
@app.route('/about',methods=['GET'])
def about():
    return render_template('about.html')

# Gives information about Team mates.
@app.route('/contactus',methods=['GET'])
def contactus():
    return render_template('ContactUs.html')

# Text to Sign conversion
@app.route('/Text_To_Sign', methods=['GET'])
def serve_page():
    return flask.render_template('Text_To_Sign.html')


# Dataset
@app.route('/dataset', methods=['GET'])
def dataset():
    return flask.render_template('Dataset.html')


# Home Page
@app.route("/",methods=['GET'])
def home():
    return flask.render_template('home.html')

# gives the corresponding lables for each text
@app.route('/process_query', methods=['POST'])
def process_query():
    data = flask.request.form  # is a dictionary
    input = data['user_input']
    #input+=' A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'
    input_in_list = input.split(' ')
    #print(input_in_list)
    extra=[]
    words=["baby","five","house","brother","dont_like","eight","example","five","four","friend","help","like","love","make","more","name","nine","no","nothing","one","pay","play","seven","stop","three","two","with","yes"]
    for i in input_in_list:
        if i.lower() in words:
            extra.append(i.lower())
        else:
            for p in i:
                extra.append(p)
        extra+="   "

    #print(extra)
    
    return flask.render_template('Text_To_Sign.html', same=processInput(extra), og=input)

# Function which sees the matching lable for the given text 
def processInput(input_in_list):
    for s, i in enumerate(input_in_list):
        print(s,i)
        if "baby" == i.lower(): 
            input_in_list[s] = "static/Baby.jpg"
        elif "brother" == i.lower():
            input_in_list[s] = "static/Brother.jpg"
        elif "dont_like"==i.lower():
            input_in_list[s]="static/Dont_like.jpg"
        elif "eight" == i.lower():
            input_in_list[s] = "static/EIGHT.jpg"
        elif "example"==i.lower():
            input_in_list[s]="static/Example.png"
        elif "five" == i.lower():
            input_in_list[s] = "static/FIVE.jpg"
        elif "four"==i.lower():
            input_in_list[s]="static/FOUR.jpg"
        elif "friend"==i.lower():
            input_in_list[s]="static/Friend.jpg"
        elif "help" == i.lower():
            input_in_list[s] = "static/Help.jpg"
        elif "house"==i.lower():
            input_in_list[s]="static/House.jpg"
        elif "like" == i.lower():
            input_in_list[s] = "static/Like.jpg"
        elif "love"==i.lower():
            input_in_list[s]="static/Love.jpg"
        elif "make" == i.lower():
            input_in_list[s] = "static/Make.jpg"
        elif "more"==i.lower():
            input_in_list[s]="static/More.jpg"
        elif "name" == i.lower():
            input_in_list[s] = "static/Name.jpg"
        elif "nine"==i.lower():
            input_in_list[s]="static/NINE.jpg" 
        elif "no" == i.lower():
            input_in_list[s] = "static/No.jpg"
        elif "nothing"==i.lower():
            input_in_list[s]="static/nothing.jpg"
        elif "one" == i.lower():
            input_in_list[s] = "static/ONE.jpg"
        elif "pay"==i.lower():
            input_in_list[s]="static/Pay.jpg"
        elif "play" == i.lower():
            input_in_list[s] = "static/Play.jpg"
        elif "seven"==i.lower():
            input_in_list[s]="static/SEVEN.jpg"
        elif "stop" == i.lower():
            input_in_list[s] = "static/Stop.jpg"
        elif "three"==i.lower():
            input_in_list[s]="static/THREE.jpg"
        elif "two" == i.lower():
            input_in_list[s] = "static/TWO.jpg"
        elif "with"==i.lower():
            input_in_list[s]="static/With.jpg"
        elif "yes" == i.lower():
            input_in_list[s] = "static/Yes.jpg"
        elif "a"==i.lower():
            input_in_list[s]="static/A.jpg"
        elif "b" ==i.lower():
            input_in_list[s]="static/B.jpg"
        elif "c"==i.lower():
            input_in_list[s]="static/C.jpg"
        elif "d" ==i.lower():
            input_in_list[s]="static/D.jpg"
        elif "e"==i.lower():
            input_in_list[s]="static/E.jpg"
        elif "f" ==i.lower():
            input_in_list[s]="static/F.jpg"
        elif "g"==i.lower():
            input_in_list[s]="static/G.jpg"
        elif "h" ==i.lower():
            input_in_list[s]="static/H.jpg"
        elif "i"==i.lower():
            input_in_list[s]="static/I.jpg"
        elif "j" ==i.lower():
            input_in_list[s]="static/J.jpg"
        elif "k"==i.lower():
            input_in_list[s]="static/K.jpg"
        elif "l" ==i.lower():
            input_in_list[s]="static/L.jpg"
        elif "m"==i.lower():
            input_in_list[s]="static/M.jpg"
        elif "n" ==i.lower():
            input_in_list[s]="static/N.jpg"
        elif "o"==i.lower():
            input_in_list[s]="static/O.jpg"
        elif "zero"==i.lower():
            input_in_list[s]="static/O.jpg"
        elif "p" ==i.lower():
            input_in_list[s]="static/P.jpg" 
        elif "q"==i.lower():
            input_in_list[s]="static/Q.jpg"
        elif "r" ==i.lower():
            input_in_list[s]="static/R.jpg"
        elif "s"==i.lower():
            input_in_list[s]="static/S.jpg"
        elif "t" ==i.lower():
            input_in_list[s]="static/T.jpg" 
        elif "u"==i.lower():
            input_in_list[s]="static/U.jpg"
        elif "v" == i.lower():
            input_in_list[s] = "static/TWO.jpg"
        elif "w" ==i.lower():
            input_in_list[s]="static/W.jpg"
        elif "six"==i.lower():
            input_in_list[s]="static/W.jpg"
        elif "x"==i.lower():
            input_in_list[s]="static/X.jpg"
        elif "y" ==i.lower():
            input_in_list[s]="static/Y.jpg"
        elif "z"==i.lower():
            input_in_list[s]="static/Z.jpg"
        else:
            input_in_list[s]="  "
        
    return input_in_list


if __name__ == "__main__":  
    app.run(debug=True, use_reloader=False)
