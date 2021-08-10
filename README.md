# MakeHuman

This is the main source code for the MakeHuman application as such. See "Getting started" below for instructions on how to get MakeHuman up and running. Mac users
_should_ be able to use the same instructions as windows users, although this has not been thoroughly tested.

## Current status

At the point of writing this, the source code is almost ready for a stable release. 

## Support requests

If you have any questions about the software and its usage, please make a request in our forum: http://www.makehumancommunity.org/forum.

A quick look through at least the top questions in the FAQ might be a good idea too: http://www.makehumancommunity.org/wiki/FAQ:Index

Please do not use the issue tracker for general tech support. For such questions, please use the forums.

## Testing and reporting bugs

The testing vision for this code is to build a community release that includes main application and often-used, user-contributed 
plug-ins. We hope that the utility of this integrated functionality is sufficient to entice a larger cohort of testers who get
value-added in exchange for the possibility of uncovering deficiencies in our application.

If you find a bug, please report it in the issues section here on github. In order to make a good bug report, please also include
the logs: http://www.makehumancommunity.org/wiki/FAQ:How\_to\_provide\_a\_makehuman\_log\_for\_a\_good\_bug\_report%3F

## Getting started

Builds for Windows platforms can be downloaded from http://www.makehumancommunity.org/content/downloads.html

If you rather run the code from source:

* Install git (https://git-scm.com/) with LFS support (https://git-lfs.github.com/). Modern git clients have LFS support included per default. 
* Make sure the command "git" is available via the PATH variable.
* Install python 3.6.x or later from https://www.python.org/ (or via your system's package management). On windows you **MUST** use 64-bit python. 32-bit python will not work.
* Install python dependencies (see below)
* Use git to clone https://github.com/makehumancommunity/makehuman.git (or download the source as a zip)
* Run the "download\_assets\_git.py" script in the "makehuman" subdirectory of the source code.
* Optionally also run:
  * compile\_models.py
  * compile\_proxies.py
  * compile\_targets.py
 
### Installing python dependencies on debian, ubuntu, mint and similar systems

All that you need should be available via apt. On a console prompt, run:

* apt-get install python3-numpy, python3-opengl, python3-pyqt5, python3-pyqt5.qtopengl, python3-pyqt5.qtsvg

### Installing python dependencies on windows

You should be able to start the command "pip" by opening a console prompt ("run" -> "cmd.exe") and writing "pip". If not, 
figure out how to run "pip": https://pip.pypa.io/en/stable/ (it should have been installed automatically by python)

Use "pip" to install dependencies. Run the following commands:

* pip install numpy
* pip install PyQt5==5.12.2
* pip install PyOpenGL

### Installing plugins

If you want to use community plugins like the asset downloader - download them, put in the plugins directory, enable in settings and restart app:

* https://github.com/makehumancommunity/community-plugins-mhapi
* https://github.com/makehumancommunity/community-plugins-assetdownload
* https://github.com/makehumancommunity/community-plugins-socket
* https://github.com/makehumancommunity/makehuman-plugin-for-blender

### Starting MakeHuman

Having done this, you can now start MakeHuman by running the makehuman.py script. On a prompt run 

* python makehuman.py (on windows)
* python3 makehuman.py (on debian, ubuntu, mint...)

## Branches

There are three standard branches and some additional developer working branches:

* master: This is where you will find the latest version of MakeHuman.

Read-only reference branches

* bitbucket-stable: This is the code as it looks in the "stable" branch at bitbucket. This is the ancestor of what is now the "master" branch.
* bitbucket-default: This is the code as it looks in the "default" branch at bitbucket.

In addition you may from time to time see feature branches (usually named \_feature...), which are removed after having been merged to the master branch. 

