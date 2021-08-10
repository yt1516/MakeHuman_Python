#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

"""
**Project Name:**      MakeHuman

**Product Home Page:** http://www.makehuman.org/

**Code Home Page:**    https://bitbucket.org/MakeHuman/makehuman/

**Authors:**           Joel Palmius, Marc Flerackers

**Copyright(c):**      MakeHuman Team 2001-2017

**Licensing:**         AGPL3

    This file is part of MakeHuman (www.makehuman.org).

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.


Abstract
--------

TODO
"""

# We need this for gui controls
import camera
from OpenGL.GL import *
import numpy as np

import datetime
import time
import gui3d
import mh
import gui
import log
import getpath
import os
import humanmodifier
import modifierslider
import io
# import algos3d
import json
from collections import OrderedDict
# import scene
# import projection
import glmodule
import image
from image import Image
import image_operations as imgop
import cv2
import dlib
import math
import random
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
# import guirender
from core import G
from progress import Progress

class ExampleTaskView(gui3d.TaskView):

    def __init__(self, category, appFacs):
        gui3d.TaskView.__init__(self, category, 'Face Matching')
        self.facs_human = appFacs.selectedHuman
        camera = G.app.modelCamera
        self.human=G.app.selectedHuman
        self.app = appFacs
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.ref_points=self.readPoints("White_Men.jpg.txt")
        self.filename="White_Men.jpg.txt"
        self.error_full=[]

        self.h_tar=self.readTargets('height_target.txt')
        self.h_idx=self.readDoublePoints('height_landmarks.txt')

        self.w_tar=self.readTargets('width_target.txt')
        self.w_idx=self.readDoublePoints('width_landmarks.txt')

        self.y_tar=self.readTargets('y_target.txt')
        self.y_idx=self.readSinglePoints('y_landmarks.txt')

        self.x_tar=self.readTargets('x_target.txt')
        self.x_idx=self.readSinglePoints('x_landmarks.txt')

        self.params=self.readTargets('weights_1d.txt')

        self.IND_SIZE = len(self.params)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()
        # Attribute generator
        self.toolbox.register("attr_float", random.uniform,-0.5,0.5)
        # Structure initializers
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=self.IND_SIZE)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.minError)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)


        box=self.addLeftWidget(gui.GroupBox('Settings'))
        self.slider_val=30
        self.calibrateButton=box.addWidget(gui.Button('Calibrate Camera'))
        self.meshSlider = box.addWidget(gui.Slider(value=self.slider_val,min=0, max=500, label=['Number of Loops',' %d']))
        self.GAButton = box.addWidget(gui.Button('Execute'))

        box2 = self.addLeftWidget(gui.GroupBox('Set canvas size'))
        self.Set1500 = box2.addWidget(gui.Button('1500x1500'))
        self.Set1000 = box2.addWidget(gui.Button('1000x1000'))
        self.dotImage = box2.addWidget(gui.Button('Generate Dotted Output'))

        @self.calibrateButton.mhEvent
        def onClicked(event):
            self.calibrateCamera(self.filename)
            for i in range(0,len(self.w_tar)):
                self.MatchWidth(40,10,0.5,0.02,self.w_tar[i],self.w_idx[i],i)

            for i in range(0,len(self.x_tar)):
                self.MatchHoriz(40,10,0.5,0.02,self.x_tar[i],self.x_idx[i],i)

            for i in range(0,len(self.h_tar)):
                self.MatchHeight(40,10,0.5,0.02,self.h_tar[i],self.h_idx[i],i) # loop,threshold,val,step,target,landmark

            for i in range(0,len(self.y_tar)):
                self.MatchVert(40,10,0.5,0.02,self.y_tar[i],self.y_idx[i],i)

            # for i in range(0,len(self.w_tar)):
            #     self.MatchWidth(40,10,0.5,0.01,self.w_tar[i],self.w_idx[i])
            #
            # for i in range(0,len(self.x_tar)):
            #     self.MatchHoriz(40,10,0.5,0.01,self.x_tar[i],self.x_idx[i])
            #
            # for i in range(0,len(self.h_tar)):
            #     self.MatchHeight(40,10,0.5,0.01,self.h_tar[i],self.h_idx[i]) # loop,threshold,val,step,target,landmark
            #
            # for i in range(0,len(self.y_tar)):
            #     self.MatchVert(40,10,0.5,0.01,self.y_tar[i],self.y_idx[i])

            G.app.mhapi.modifiers.applySymmetryLeft()
            self.screenShot('initial.png')

            # self.MatchWidth(10,15,0.5,0.01,'head/head-fat-decr',[5,11])
            # log.debug(len(self.h_tar))
            # log.debug(len(self.h_idx))
            log.debug(self.error_full)

        @self.dotImage.mhEvent
        def onClicked(event):
            self.screenShot('initial.png')
            file_name='initial.png'+'.txt'
            MH_landmarks=self.readDoubleFace('initial.png',self.detector,self.predictor)
            np.savetxt(file_name,MH_landmarks,fmt="%d")
            output=cv2.imread('initial.png')
            for p in MH_landmarks:
                self.drawPoint(output, (int(p[0]),int(p[1])), (0,0,255))
            cv2.imwrite('initial_dotted.png',output)

        @self.meshSlider.mhEvent
        def onChanging(value):
            self.slider_val=value
            human = gui3d.app.selectedHuman

        @self.GAButton.mhEvent
        def onClicked(event):
            self.main(self.slider_val)

        @self.Set1500.mhEvent
        def onClicked(event):
            Width=1503
            Height=1503
            qmainwin = G.app.mainwin
            central = qmainwin.centralWidget()
            cWidth = central.frameSize().width()
            cHeight = central.frameSize().height()
            width = G.windowWidth
            height = G.windowHeight
            xdiff = Width - width
            ydiff = Height - height

            cWidth = cWidth + xdiff
            cHeight = cHeight + ydiff

            central.setFixedSize(cWidth,cHeight)
            qmainwin.adjustSize()

        @self.Set1000.mhEvent
        def onClicked(event):
            Width=1003
            Height=1003
            qmainwin = G.app.mainwin
            central = qmainwin.centralWidget()
            cWidth = central.frameSize().width()
            cHeight = central.frameSize().height()
            width = G.windowWidth
            height = G.windowHeight
            xdiff = Width - width
            ydiff = Height - height

            cWidth = cWidth + xdiff
            cHeight = cHeight + ydiff

            central.setFixedSize(cWidth,cHeight)
            qmainwin.adjustSize()

    def readTargets(self,filename):
        target_names=[]
        with open(filename,'r') as file:
            for line in file:
                target_names.append(line.strip('\n'))
        return target_names

    def readFace(self,img_name,detector,predictor):

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

    def readDoubleFace(self,img_name,detector,predictor):
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
                points = (int(x), int(y))
                pointsArray.append(points)
        return pointsArray

    def readDoublePoints(self,filename):
        points=[]
        with open(filename) as file:
            for line in file :
                x, y = line.split()
                points.append((int(x), int(y)))
        return points

    def readSinglePoints(self,filename):
        points=[]
        with open(filename) as file:
            for line in file :
                points.append(int(line))
        return points

    def readPoints(self,filename):
        content=[]
        with open(filename) as file:
            for line in file:
                x,y=line.split()
                content.append(int(x))
                content.append(int(y))
        return(content[0:68*2])

    def drawPoint(self,img, p, color ) :
        cv2.circle( img, p, 5, color, -1, cv2.LINE_AA, 0 )

    def difference_new(self,point1,point2):
        (x1,y1)=point1
        (x2,y2)=point2
        x_diff=x1-x2
        y_diff=y1-y2
        return(x_diff,y_diff)

    def screenShot(self,fileName):
        log.message("SCRIPT: screenShot(" + fileName + ")")
        width = G.windowWidth
        height = G.windowHeight
        width = width - 3
        height = height - 3
        mh.grabScreen(1,1,width,height,fileName)

    def minError(self,individual):
        # Make the changes
        for i in range(len(individual)):
            G.app.mhapi.modifiers.applyTarget(self.params[i],individual[i])
        # Screenshot it
        self.screenShot('initial.png')

        # Get points
        current_points=self.readFace('initial.png',self.detector,self.predictor)

        errors=[]
        for i in range(len(self.ref_points)):
            errors.append(abs(self.ref_points[i]-current_points[i]))
        return sum(errors),


    # Assumptions:
    # 1. MH width is always greater than Man width. Positive coef to go thinner
    # 2. MH height is always greater than Man height. Positive coef to go shorter
    # 3. MH horizontal pos is always on the right of Man. Positive coef to decrease
    # 4. MH vertical pos is always below Man. Positive coef to increase

    def MatchHeight(self,loop,threshold,val,step,target,landmark,j):
        coef=1
        MH_landmarks=self.readDoubleFace('initial.png',self.detector,self.predictor)
        Man_landmarks=self.readDoublePoints(self.filename)
        error_all=[]
        lowest_val=[]
        min_idx=0

        error_mh=self.difference_new(MH_landmarks[landmark[0]],MH_landmarks[landmark[1]])
        error_man=self.difference_new(Man_landmarks[landmark[0]],Man_landmarks[landmark[1]])
        error_last=error_mh[1]-error_man[1]

        error_all.append(abs(error_last))
        lowest_val.append(val)

        for i in range(loop):
            min_val=min(error_all)
            min_idx=error_all.index(min(error_all))
            if abs(error_last)>=threshold:
                if error_last>0:
                    coef=1
                else:
                    coef=-1
                val=val+coef*step
                G.app.mhapi.modifiers.applyTarget(target,val)
                self.screenShot('initial.png')
                self.screenShot(str(j)+'_'+str(i)+'_'+'height.png')
                MH_landmarks=self.readDoubleFace('initial.png',self.detector,self.predictor)
                error_mh=self.difference_new(MH_landmarks[landmark[0]],MH_landmarks[landmark[1]])
                error_last=error_mh[1]-error_man[1]
                error_all.append(abs(error_last))
                lowest_val.append(val)

        if min_val<abs(error_last):
            val=lowest_val[min_idx]
            G.app.mhapi.modifiers.applyTarget(target,val)
            self.screenShot('initial.png')
            log.debug("used min")
            self.error_full.append(min_val)
        else:
            log.debug("Minimum Error:")
            log.debug(error_last)
            self.error_full.append(abs(error_last))

    def MatchWidth(self,loop,threshold,val,step,target,landmark,j):
        coef=1
        MH_landmarks=self.readDoubleFace('initial.png',self.detector,self.predictor)
        Man_landmarks=self.readDoublePoints(self.filename)
        error_all=[]
        lowest_val=[]
        min_idx=0

        error_mh=self.difference_new(MH_landmarks[landmark[0]],MH_landmarks[landmark[1]])
        error_man=self.difference_new(Man_landmarks[landmark[0]],Man_landmarks[landmark[1]])
        error_last=error_mh[0]-error_man[0]

        error_all.append(abs(error_last))
        lowest_val.append(val)

        for i in range(loop):
            min_val=min(error_all)
            min_idx=error_all.index(min(error_all))
            if abs(error_last)>=threshold:
                if error_last>0:
                    coef=1
                else:
                    coef=-1
                val=val+coef*step
                G.app.mhapi.modifiers.applyTarget(target,val)
                self.screenShot('initial.png')
                self.screenShot(str(j)+'_'+str(i)+'_'+'width.png')
                MH_landmarks=self.readDoubleFace('initial.png',self.detector,self.predictor)
                error_mh=self.difference_new(MH_landmarks[landmark[0]],MH_landmarks[landmark[1]])
                error_last=error_mh[0]-error_man[0]
                error_all.append(abs(error_last))
                lowest_val.append(val)

        if min_val<abs(error_last):
            val=lowest_val[min_idx]
            G.app.mhapi.modifiers.applyTarget(target,val)
            self.screenShot('initial.png')
            log.debug("used min")
            self.error_full.append(min_val)
        else:
            log.debug("Minimum Error:")
            log.debug(error_last)
            self.error_full.append(abs(error_last))

    def MatchHoriz(self,loop,threshold,val,step,target,landmark,j):
        coef=1
        MH_landmarks=self.readDoubleFace('initial.png',self.detector,self.predictor)
        Man_landmarks=self.readDoublePoints(self.filename)
        error_all=[]
        lowest_val=[]
        min_idx=0

        error=self.difference_new(MH_landmarks[landmark],Man_landmarks[landmark])
        error_last=error[0]

        error_all.append(abs(error_last))
        lowest_val.append(val)

        for i in range(loop):
            min_val=min(error_all)
            min_idx=error_all.index(min(error_all))
            if abs(error_last)>=threshold:
                if error_last>=0:
                    coef=1
                else:
                    coef=-1
                val=val+coef*step
                G.app.mhapi.modifiers.applyTarget(target,val)
                self.screenShot('initial.png')
                self.screenShot(str(j)+'_'+str(i)+'_'+'horiz.png')
                MH_landmarks=self.readDoubleFace('initial.png',self.detector,self.predictor)
                error=self.difference_new(MH_landmarks[landmark],Man_landmarks[landmark])
                error_last=error[0]
                error_all.append(abs(error_last))
                lowest_val.append(val)

        if min_val<abs(error_last):
            val=lowest_val[min_idx]
            G.app.mhapi.modifiers.applyTarget(target,val)
            self.screenShot('initial.png')
            log.debug("used min")
            self.error_full.append(min_val)
        else:
            log.debug("Minimum Error:")
            log.debug(error_last)
            self.error_full.append(abs(error_last))

    def MatchVert(self,loop,threshold,val,step,target,landmark,j):
        log.debug(target)
        coef=1
        MH_landmarks=self.readDoubleFace('initial.png',self.detector,self.predictor)
        Man_landmarks=self.readDoublePoints(self.filename)
        error_all=[]
        lowest_val=[]
        min_idx=0

        error=self.difference_new(MH_landmarks[landmark],Man_landmarks[landmark])
        error_last=error[1]

        error_all.append(abs(error_last))
        lowest_val.append(val)

        for i in range(loop):
            log.debug(error_last)
            min_val=min(error_all)
            min_idx=error_all.index(min(error_all))
            if abs(error_last)>=threshold:
                if error_last>=0:
                    coef=1
                else:
                    coef=-1
                val=val+coef*step
                G.app.mhapi.modifiers.applyTarget(target,val)
                self.screenShot('initial.png')
                self.screenShot(str(j)+'_'+str(i)+'_'+'vert.png')
                MH_landmarks=self.readDoubleFace('initial.png',self.detector,self.predictor)
                error=self.difference_new(MH_landmarks[landmark],Man_landmarks[landmark])
                error_last=error[1]
                error_all.append(abs(error_last))
                lowest_val.append(val)

        if min_val<abs(error_last):
            val=lowest_val[min_idx]
            G.app.mhapi.modifiers.applyTarget(target,val)
            self.screenShot('initial.png')
            log.debug("used min")
            self.error_full.append(min_val)
        else:
            log.debug("Minimum Error:")
            log.debug(error_last)
            self.error_full.append(abs(error_last))

    def calibrateCamera(self,filename):
        loop=40
        zoom=12
        move=0.80
        error_full=[]
        G.app.modelCamera.translation=[0,move,0]
        G.app.modelCamera.zoomFactor=zoom
        self.screenShot('initial.png')
        MH_landmarks=self.readDoubleFace('initial.png',self.detector,self.predictor)
        Man_landmarks=self.readDoublePoints(filename)
        error_l=self.difference_new(MH_landmarks[1],Man_landmarks[1])
        error_r=self.difference_new(MH_landmarks[15],Man_landmarks[15])
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

                self.screenShot('initial.png')

                MH_landmarks=self.readDoubleFace('initial.png',self.detector,self.predictor)

                error_l=self.difference_new(MH_landmarks[1],Man_landmarks[1])

                error_r=self.difference_new(MH_landmarks[15],Man_landmarks[15])

                error_sum=abs(error_l[0])+abs(error_l[1])+abs(error_r[0])+abs(error_r[1])

                error_all.append(error_sum)
                lowest_val.append([zoom,move])

        if minim<error_sum:
            [zoom,move]=lowest_val[minim_idx]
            G.app.modelCamera.translation=[0,move,0]
            G.app.modelCamera.zoomFactor=zoom
            self.screenShot('initial.png')
            error_full.append(minim)
        else:
            error_full.append(error_sum)
            pass

        log.debug("Calibrate Done, Error:")
        log.debug(error_full)
        log.debug("Proceeding..........")

    def main(self,sliderval):
        pop = self.toolbox.population(n=200)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        pop, log = algorithms.eaSimple(pop, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=sliderval,
                                       stats=stats, halloffame=hof, verbose=True)


category = None
taskview = None

# This method is called when the plugin is loaded into makehuman
# The app reference is passed so that a plugin can attach a new category, task, or other GUI elements
def load(app):
    category = app.getCategory('Modelling')
    taskview = category.addTask(ExampleTaskView(category, app))


# This method is called when the plugin is unloaded from makehuman
# At the moment this is not used, but in the future it will remove the added GUI elements


def unload(app):
    pass
