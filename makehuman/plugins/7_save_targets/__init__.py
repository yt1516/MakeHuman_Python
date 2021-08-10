#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
**Project Name:**      MakeHuman

**Product Home Page:** http://www.makehumancommunity.org/

**Github Code Home Page:**    https://github.com/makehumancommunity/

**Authors:**           Joel Palmius, Aranuvir

**Copyright(c):**      MakeHuman Team 2001-2020

**Licensing:**         AGPL3

    This file is part of MakeHuman (www.makehumancommunity.org).

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
import os
import log
from .savetargets import SaveTargetsTaskView


def load(app):
    category = app.getCategory('Utilities')
    taskview = category.addTask(SaveTargetsTaskView(category))


def unload(app):

    category = app.getCategory('Utilities')
    taskview = category.getTaskByName('Save Targets')


    if os.path.isfile(taskview.metaFile):
        try:
            os.remove(taskview.metaFile)
        except OSError as e:
            log.error('Cannot delete meta target file: %s',  format(taskview.metaFile))

    if os.path.isdir(taskview.metaFilePath):
        try:
            os.rmdir(taskview.metaFilePath)
        except OSError as e:
            if e.errno == 39:
                log.warning('Cannot delete save targets cache: %s. Folder still in use...', format(taskview.metaFilePath))
            else:
                log.error('Cannot delete save targets cache: %s', format(taskview.metaFilePath))
