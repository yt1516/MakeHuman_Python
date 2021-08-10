import os
import cv2
import dlib
import numpy as np
import math

import array
import random

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

def readFace(img_name,detector,predictor):
    
    img = cv2.imread(img_name)
    gray=cv2.cvtColor(src=img,code=cv2.COLOR_BGR2GRAY)
    faces=detector(gray)
    pointsArray = [];
    
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        landmarks=predictor(image=gray,box=face)
        
        for n in range(0,68):
            x=landmarks.part(n).x
            y=landmarks.part(n).y
            pointsArray.append(int(x))
            pointsArray.append(int(y))
            
    return pointsArray

def readPoints(filename):
    content=[]
    with open(filename) as file:
        for line in file:
            x,y=line.split()
            content.append(int(x))
            content.append(int(y))
    return(content[0:68*2])

def minError(individual):
    # Make the changes
    for i in range(len(individual)):
        MHScript.applyTarget(params[i],individual[i])
    # Screenshot it
    MHScript.screenShot('initial.png')
    
    # Get points
    current_points=readFace('initial.png',detector,predictor)
    
    errors=[]
    for i in range(len(ref_points)):
        errors.append(abs(ref_points[i]-current_points[i]))
    return sum(errors),

# Load reference points

global ref_points
ref_points=readPoints(filename)

# Load parameters

global params
params=[]
with open("weights_1d.txt",'r') as file:
    for line in file:
        params.append(line.strip('\n'))
        
# Load detector and predictor

global detector
global predictor
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
# Parameter size

IND_SIZE = len(params)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_float", random.random) # Each value is 0 to 1

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


























# Append all the parameters to be changed
parameters=[]
with open("weights_1d.txt",'r') as file:
    for line in file:
        parameters.append(line.strip('\n'))

# Load reference points
Man_img='White_Men.jpg.txt'
Man_landmarks=readPoints(Man_img)


# Initialise the model to match the two anchor points

loop=30
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
error_full=[]

MH_img='initial.png'
MH_landmarks=txt_gen(MH_img,detector,predictor)

# Position anchor points and set camera
error_l=difference_new(MH_landmarks[1],Man_landmarks[1])

error_r=difference_new(MH_landmarks[15],Man_landmarks[15])

error_track=[]
error_track.append([error_l,error_r])

zoom=12
move=0.83
#zoom=13
#move=0.83
zoom_c=1
move_c=1

zoom_step=0.01
move_step=0.001

error_sum=abs(error_l[0])+abs(error_l[1])+abs(error_r[0])+abs(error_r[1])

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append([zoom,move])

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))

    if error_sum >= 20:

        G.app.modelCamera._horizontalRotation=0
        G.app.modelCamera._verticalInclination=0

        if error_l[0]>0 and error_r[0]<0:
            zoom_c=1
        else:
            zoom_c=-1

        if error_l[1]<0 and error_r[1]<0:
            move_c=1
        else:
            move_c=-1

        zoom = zoom + zoom_c*zoom_step
        move = move + move_c*move_step

        G.app.modelCamera.translation=[0,move,0]
        G.app.modelCamera.zoomFactor=zoom
        G.app.modelCamera._upY=0.2
        G.app.modelCamera._eyeX=0
        G.app.modelCamera._eyeZ=0

        MHScript.screenShot('initial.png')

        MH_landmarks=txt_gen(MH_img,detector,predictor)

        error_l=difference_new(MH_landmarks[1],Man_landmarks[1])

        error_r=difference_new(MH_landmarks[15],Man_landmarks[15])

        error_sum=abs(error_l[0])+abs(error_l[1])+abs(error_r[0])+abs(error_r[1])

        # log.debug("positioning:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append([zoom,move])

if minim<error_sum:
    [zoom,move]=lowest_val[minim_idx]
    G.app.modelCamera.translation=[0,move,0]
    G.app.modelCamera.zoomFactor=zoom
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass