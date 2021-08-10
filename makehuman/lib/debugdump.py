#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
**Project Name:**      MakeHuman

**Product Home Page:** http://www.makehumancommunity.org/

**Github Code Home Page:**    https://github.com/makehumancommunity/

**Authors:**           Joel Palmius

**Copyright(c):**      MakeHuman Team 2001-2020

**Licensing:**         AGPL3

    This file is part of MakeHuman Community (www.makehumancommunity.org).

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

This module dumps important debug information to a text file in the user's home directory
"""

import sys
import os
import re
import platform
import locale
if sys.platform.startswith('win'):
    import winreg
import log
import getpath
from mhversion import MHVersion
from core import G

class DependencyError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class DebugDump(object):

    """
    A class that dumps relevant information to a text file in the user's home directory
    """
    def __init__(self):
        self.debugpath = None

    def open(self):
        import io
        if self.debugpath is None:
            self.debugpath = getpath.getPath()

            if not os.path.exists(self.debugpath):
                os.makedirs(self.debugpath)

            self.debugpath = os.path.join(self.debugpath, "makehuman-debug.txt")
            self.debug = io.open(self.debugpath, "w", encoding="utf-8")
        else:
            self.debug = io.open(self.debugpath, "a", encoding="utf-8")

    def write(self, msg, *args):
        try:
            log.debug(msg, *args)
            self.debug.write((msg % args) + "\n")
        except UnicodeDecodeError:
            #encs = [sys.stdout.encoding,sys.getfilesystemencoding(),sys.getdefaultencoding(),'utf-8']
            #msg = getpath.stringToUnicode(msg,encs)
            #uargs = []
            #for i in args:
            #    if isinstance(i,str):
            #        uargs.append(getpath.stringToUnicode(i,encs))
            #    else:
            #        uargs.append(i)
            #
            #log.debug(msg, *uargs)
            #self.debug.write((msg % uargs) + "\n")
            pass

    def close(self):
        self.debug.close()
        self.debug = None

    def reset(self):
        self.open()

        mhv = MHVersion()

        self.write("VERSION: %s", os.environ['MH_VERSION'])
        self.write("SHORT VERSION: %s", os.environ['MH_SHORT_VERSION'])
        self.write("LONG VERSION: %s", mhv.getFullVersionStr())
        self.write("BASEMESH VERSION: %s", os.environ['MH_MESH_VERSION'])
        self.write("IS BUILT (FROZEN): %s", os.environ['MH_FROZEN'])
        self.write("IS RELEASE VERSION: %s", os.environ['MH_RELEASE'])
        self.write("DEFAULT ENCODING: %s", sys.getdefaultencoding())
        self.write("FILESYSTEM ENCODING: %s", sys.getfilesystemencoding())
        self.write("STDOUT ENCODING: %s", sys.stdout.encoding)
        self.write("LOCALE PREFERRED ENCODING: %s", locale.getpreferredencoding(False))
        self.write("WORKING DIRECTORY: %s", getpath.pathToUnicode(os.getcwd()))
        self.write("HOME LOCATION: %s", getpath.pathToUnicode(getpath.getHomePath()))
        syspath = os.path.pathsep.join( [getpath.pathToUnicode(p) for p in sys.path] )
        self.write("PYTHON PATH: %s", syspath)
        self.write("DLL PATH: %s", getpath.pathToUnicode(os.environ['PATH']))
        version = re.sub(r"[\r\n]"," ", sys.version)
        self.write("SYS.VERSION: %s", version)
        self.write("SYS.PLATFORM: %s", sys.platform)
        self.write("SYS.EXECUTABLE: %s", sys.executable)
        self.write("PLATFORM.MACHINE: %s", platform.machine())
        self.write("PLATFORM.PROCESSOR: %s", platform.processor())
        self.write("PLATFORM.UNAME.RELEASE: %s", platform.uname()[2])

        if sys.platform.startswith('linux'):
            try:
                self.write("PLATFORM.LINUX_DISTRIBUTION: %s", ' '.join(platform.linux_distribution()))
            except AttributeError:
                try:
                    import distro
                    self.write("PLATFORM.LINUX_DISTRIBUTION: %s", ' '.join(distro.linux_distribution()))
                except ImportError:
                    self.write("PLATFORM.LINUX_DISTRIBUTION: %s", 'Unknown')
            
        if sys.platform.startswith('darwin'):
            self.write("PLATFORM.MAC_VER: %s", platform.mac_ver()[0])
            
        if sys.platform.startswith('win'):
            self.write("PLATFORM.WIN32_VER: %s", " ".join(platform.win32_ver()))

        import numpy
        self.write("NUMPY.VERSION: %s", numpy.__version__)
        numpyVer = numpy.__version__.split('.')
        if int(numpyVer[0]) <= 1 and int(numpyVer[1]) < 6:
            raise DependencyError('MakeHuman requires at least numpy version 1.6')

        self.close()

    def appendGL(self):
        import OpenGL
        self.open()
        self.write("PYOPENGL.VERSION: %s", OpenGL.__version__)
        
        noshaders = "not set"
        if G.args.get('noshaders', False):
            noshaders = "set via command line"
        if G.preStartupSettings["noShaders"]:
            noshaders = "set via setting"
        self.write("NOSHADERS: %s", noshaders)
        
        nosamplebuffers = "not set"
        if G.preStartupSettings["noSampleBuffers"]:
            nosamplebuffers = "set via setting"
        self.write("NOSAMPLEBUFFERS: %s", nosamplebuffers)

        self.close()

    def appendQt(self):
        import qtui
        self.open()
        self.write("QT.VERSION: %s", qtui.getQtVersionString())
        self.write("QT.JPG_SUPPORT: %s", "supported" if qtui.supportsJPG() else "not supported")
        self.write("QT.SVG_SUPPORT: %s", "supported" if qtui.supportsSVG() else "not supported")
        py_plugin_path = os.path.pathsep.join( [getpath.pathToUnicode(p) for p in qtui.QtCore.QCoreApplication.libraryPaths()] )
        self.write("QT.PLUGIN_PATH: %s" % py_plugin_path)
        qt_plugin_path_env = os.environ['QT_PLUGIN_PATH'] if 'QT_PLUGIN_PATH' in os.environ else ""
        self.write("QT.PLUGIN_PATH_ENV: %s" % getpath.pathToUnicode(qt_plugin_path_env))
        qt_conf_present = os.path.isfile(getpath.getSysPath('qt.conf'))
        if qt_conf_present:
            import io
            f = io.open(getpath.getSysPath('qt.conf'), "r", encoding="utf-8", errors="replace")
            qt_conf_content = f.read()
            qt_conf_content = qt_conf_content.replace('\n', '\n'+(' '*len('QT.CONF: '))).strip()
            f.close()
            self.write("QT.CONF: %s" % qt_conf_content)
        else:
            self.write("QT.CONF: NOT PRESENT")
        self.close()

    def appendMessage(self,message):
        self.open()
        self.write(message)
        self.close()

dump = DebugDump()
