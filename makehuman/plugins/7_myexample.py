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
# import guirender
from core import G
from progress import Progress

class ExampleTaskView(gui3d.TaskView):

    def __init__(self, category, appFacs):
        gui3d.TaskView.__init__(self, category, 'Face Dimensions')
        self.facs_human = appFacs.selectedHuman
        camera = G.app.modelCamera
        self.app = appFacs

        self.face_feature_path=getpath.getPath('data')
        self.face_feature_file=self.face_feature_path + 'modeling_modifiers.json'

        data = json.load(io.open(getpath.getSysDataPath('modifiers/modeling_modifiers.json'),'r',encoding='utf-8'),object_pairs_hook=OrderedDict)

        groupNames=[]
        count = 1
        for modifierGroup in data:
            if count <=10:
                groupName = modifierGroup['group']
                groupNames.append(groupName)
                count += 1
        log.debug(groupNames)

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
