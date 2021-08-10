import os
import cv2
import dlib
import numpy as np
import math

global np
global cv2
global dlib
global math


def txt_gen(img_name,detector,predictor):
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

def readPoints(filename):
    points=[]
    with open(filename) as file:
        for line in file :
            x, y = line.split()
            points.append((int(x), int(y)))

    return points

def difference_new(point1,point2):
    (x1,y1)=point1
    (x2,y2)=point2
    x_diff=x1-x2
    y_diff=y1-y2

    return(x_diff,y_diff)

def draw_point(img, p, color ) :
    cv2.circle( img, p, 5, color, -1, cv2.LINE_AA, 0 )

loop=30
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
error_full=[]

MH_img='initial.png'
MH_landmarks=txt_gen(MH_img,detector,predictor)

Man_img='White_Men.jpg.txt'
Man_landmarks=readPoints(Man_img)

# Position anchor points and set camera
error_l=difference_new(MH_landmarks[0],Man_landmarks[0])

error_r=difference_new(MH_landmarks[16],Man_landmarks[16])

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

        error_l=difference_new(MH_landmarks[0],Man_landmarks[0])

        error_r=difference_new(MH_landmarks[16],Man_landmarks[16])

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

# head length

head_e=difference_new(MH_landmarks[8],Man_landmarks[8])

val=0
step=0.01
coef=1

error_sum=abs(head_e[1])

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=15:
        if head_e[1]>=0:
            coef=1
        else:
            coef=-1
        val=val+coef*step
        MHScript.applyTarget('head/head-scale-vert-decr',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        head_e=difference_new(MH_landmarks[8],Man_landmarks[8])

        error_sum=abs(head_e[1])

        # log.debug("head length:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('head/head-scale-vert-decr',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass

# Head fat

eye_h_MH=difference_new(MH_landmarks[3],MH_landmarks[11])
eye_h_MAN=difference_new(Man_landmarks[3],Man_landmarks[11])
eye_h1=eye_h_MH[1]-eye_h_MAN[1]

val=0
step=0.01
coef=1

error_sum=abs(eye_h1)

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=1:
        if eye_h1>=0:
            coef=1
        else:
            coef=-1
        val=val+coef*step
        MHScript.applyTarget('head/head-fat-incr',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eye_h_MH=difference_new(MH_landmarks[3],MH_landmarks[11])
        eye_h_MAN=difference_new(Man_landmarks[3],Man_landmarks[11])
        eye_h1=eye_h_MH[1]-eye_h_MAN[1]

        error_sum=abs(eye_h1)

        # log.debug("Upper lip horizontal size:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('head/head-fat-incr',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass

# eyebrow height

eyebrow_h=difference_new(MH_landmarks[19],Man_landmarks[19])

val=0
step=0.01
coef=1

error_sum=abs(eyebrow_h[1])

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=15:
        if eyebrow_h[1]>=0:
            coef=1
        else:
            coef=-1
        val=val+coef*step
        MHScript.applyTarget('eyebrows/eyebrows-trans-up',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eyebrow_h=difference_new(MH_landmarks[19],Man_landmarks[19])

        error_sum=abs(eyebrow_h[1])

        # log.debug("Eyebrow height:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('eyebrows/eyebrows-trans-up',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass

# eyebrow angle

eyebrow_a=difference_new(MH_landmarks[17],Man_landmarks[17])

val=0
step=0.01
coef=1

error_sum=abs(eyebrow_a[1])

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=15:
        if eyebrow_a[1]>=0:
            coef=1
        else:
            coef=-1
        val=val+coef*step
        MHScript.applyTarget('eyebrows/eyebrows-angle-up',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eyebrow_a=difference_new(MH_landmarks[17],Man_landmarks[17])

        error_sum=abs(eyebrow_a[1])

        # log.debug("eyebrow angle:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('eyebrows/eyebrows-angle-up',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass

# Right Eye inner height

eye_h_MH=difference_new(MH_landmarks[40],MH_landmarks[38])
eye_h_MAN=difference_new(Man_landmarks[40],Man_landmarks[38])
eye_h1=eye_h_MH[1]-eye_h_MAN[1]

val=0
step=0.01
coef=1

error_sum=abs(eye_h1)

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=1:
        if eye_h1>=0:
            coef=-1
        else:
            coef=+1
        val=val+coef*step
        MHScript.applyTarget('eyes/r-eye-height1-incr',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eye_h_MH=difference_new(MH_landmarks[40],MH_landmarks[38])
        eye_h_MAN=difference_new(Man_landmarks[40],Man_landmarks[38])
        eye_h1=eye_h_MH[1]-eye_h_MAN[1]

        error_sum=abs(eye_h1)

        # log.debug("right eye inner height:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('eyes/r-eye-height1-incr',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass

# Right Eye outer height

eye_h_MH=difference_new(MH_landmarks[41],MH_landmarks[37])
eye_h_MAN=difference_new(Man_landmarks[41],Man_landmarks[37])
eye_h1=eye_h_MH[1]-eye_h_MAN[1]

val=0
step=0.01
coef=1

error_sum=abs(eye_h1)

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=1:
        if eye_h1>=0:
            coef=-1
        else:
            coef=+1
        val=val+coef*step
        MHScript.applyTarget('eyes/r-eye-height3-incr',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eye_h_MH=difference_new(MH_landmarks[41],MH_landmarks[37])
        eye_h_MAN=difference_new(Man_landmarks[41],Man_landmarks[37])
        eye_h1=eye_h_MH[1]-eye_h_MAN[1]

        error_sum=abs(eye_h1)

        # log.debug("right eye outer height:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('eyes/r-eye-height3-incr',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass

# Right Eye horizontal location

eye_h1=difference_new(MH_landmarks[38],Man_landmarks[38])

val=0
step=0.01
coef=1

error_sum=abs(eye_h1[0])

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=1:
        if eye_h1[0]>=0:
            coef=-1
        else:
            coef=+1
        val=val+coef*step
        MHScript.applyTarget('eyes/r-eye-trans-in',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eye_h1=difference_new(MH_landmarks[38],Man_landmarks[38])

        error_sum=abs(eye_h1[0])

        # log.debug("right eye horizontal loc:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('eyes/r-eye-trans-in',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass

# Right Eye vertical location

eye_h1=difference_new(MH_landmarks[38],Man_landmarks[38])

val=0
step=0.01
coef=1

error_sum=abs(eye_h1[1])

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=1:
        if eye_h1[1]>=0:
            coef=1
        else:
            coef=-1
        val=val+coef*step
        MHScript.applyTarget('eyes/r-eye-trans-up',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eye_h1=difference_new(MH_landmarks[38],Man_landmarks[38])

        error_sum=abs(eye_h1[0])

        # log.debug("right eye vertical loc:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('eyes/r-eye-trans-up',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass

# Right Eye outer corner horizontal

eye_h1=difference_new(MH_landmarks[36],Man_landmarks[36])

val=0
step=0.01
coef=1

error_sum=abs(eye_h1[0])

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=1:
        if eye_h1[0]>=0:
            coef=-1
        else:
            coef=+1
        val=val+coef*step
        MHScript.applyTarget('eyes/r-eye-push1-in',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eye_h1=difference_new(MH_landmarks[36],Man_landmarks[36])

        error_sum=abs(eye_h1[0])

        # log.debug("right eye outer corner horizontal:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('eyes/r-eye-push1-in',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass

# Right Eye outer corner vertical

eye_h1=difference_new(MH_landmarks[36],Man_landmarks[36])

val=0
step=0.01
coef=1

error_sum=abs(eye_h1[1])

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=1:
        if eye_h1[1]>=0:
            coef=1
        else:
            coef=-1
        val=val+coef*step
        MHScript.applyTarget('eyes/r-eye-corner1-up',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eye_h1=difference_new(MH_landmarks[36],Man_landmarks[36])

        error_sum=abs(eye_h1[0])

        # log.debug("right eye outer corner vertical:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('eyes/r-eye-corner1-up',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass

# Right Eye inner corner horizontal

eye_h1=difference_new(MH_landmarks[39],Man_landmarks[39])

val=0
step=0.01
coef=1

error_sum=abs(eye_h1[0])

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=1:
        if eye_h1[0]>=0:
            coef=-1
        else:
            coef=+1
        val=val+coef*step
        MHScript.applyTarget('eyes/r-eye-push2-in',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eye_h1=difference_new(MH_landmarks[39],Man_landmarks[39])

        error_sum=abs(eye_h1[0])

        # log.debug("right eye inner corner horizontal:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('eyes/r-eye-push2-in',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass

# Right Eye outer corner vertical

eye_h1=difference_new(MH_landmarks[39],Man_landmarks[39])

val=0
step=0.01
coef=1

error_sum=abs(eye_h1[1])

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=1:
        if eye_h1[1]>=0:
            coef=1
        else:
            coef=-1
        val=val+coef*step
        MHScript.applyTarget('eyes/r-eye-corner2-up',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eye_h1=difference_new(MH_landmarks[39],Man_landmarks[39])

        error_sum=abs(eye_h1[0])

        # log.debug("right eye inner corner vertical:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('eyes/r-eye-corner2-up',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass

# left Eye inner height

eye_h_MH=difference_new(MH_landmarks[47],MH_landmarks[43])
eye_h_MAN=difference_new(Man_landmarks[47],Man_landmarks[43])
eye_h1=eye_h_MH[1]-eye_h_MAN[1]

val=0
step=0.01
coef=1

error_sum=abs(eye_h1)

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=1:
        if eye_h1>=0:
            coef=-1
        else:
            coef=+1
        val=val+coef*step
        MHScript.applyTarget('eyes/l-eye-height1-incr',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eye_h_MH=difference_new(MH_landmarks[47],MH_landmarks[43])
        eye_h_MAN=difference_new(Man_landmarks[47],Man_landmarks[43])
        eye_h1=eye_h_MH[1]-eye_h_MAN[1]

        error_sum=abs(eye_h1)

        # log.debug("left eye inner height:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('eyes/l-eye-height1-incr',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass

# left Eye outer height

eye_h_MH=difference_new(MH_landmarks[46],MH_landmarks[44])
eye_h_MAN=difference_new(Man_landmarks[46],Man_landmarks[44])
eye_h1=eye_h_MH[1]-eye_h_MAN[1]

val=0
step=0.01
coef=1

error_sum=abs(eye_h1)

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=1:
        if eye_h1>=0:
            coef=-1
        else:
            coef=+1
        val=val+coef*step
        MHScript.applyTarget('eyes/l-eye-height3-incr',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eye_h_MH=difference_new(MH_landmarks[46],MH_landmarks[44])
        eye_h_MAN=difference_new(Man_landmarks[46],Man_landmarks[44])
        eye_h1=eye_h_MH[1]-eye_h_MAN[1]

        error_sum=abs(eye_h1)

        # log.debug("left eye outer height:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('eyes/l-eye-height3-incr',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass
# left Eye horizontal location

eye_h1=difference_new(MH_landmarks[43],Man_landmarks[43])

val=0
step=0.01
coef=1

error_sum=abs(eye_h1[0])

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=1:
        if eye_h1[0]>=0:
            coef=1
        else:
            coef=-1
        val=val+coef*step
        MHScript.applyTarget('eyes/l-eye-trans-in',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eye_h1=difference_new(MH_landmarks[43],Man_landmarks[43])

        error_sum=abs(eye_h1[0])

        # log.debug("left eye horizontal loc:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('eyes/l-eye-trans-in',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass

# left Eye vertical location

eye_h1=difference_new(MH_landmarks[43],Man_landmarks[43])

val=0
step=0.01
coef=1

error_sum=abs(eye_h1[1])

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=1:
        if eye_h1[1]>=0:
            coef=1
        else:
            coef=-1
        val=val+coef*step
        MHScript.applyTarget('eyes/l-eye-trans-up',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eye_h1=difference_new(MH_landmarks[43],Man_landmarks[43])

        error_sum=abs(eye_h1[0])

        # log.debug("left eye vertical loc:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('eyes/l-eye-trans-up',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass

# left Eye outer corner horizontal

eye_h1=difference_new(MH_landmarks[45],Man_landmarks[45])

val=0
step=0.01
coef=1

error_sum=abs(eye_h1[0])

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=1:
        if eye_h1[0]>=0:
            coef=1
        else:
            coef=-1
        val=val+coef*step
        MHScript.applyTarget('eyes/l-eye-push1-in',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eye_h1=difference_new(MH_landmarks[45],Man_landmarks[45])

        error_sum=abs(eye_h1[0])

        # log.debug("left eye outer corner horizontal:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('eyes/l-eye-push1-in',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass
# left Eye outer corner vertical

eye_h1=difference_new(MH_landmarks[45],Man_landmarks[45])

val=0
step=0.01
coef=1

error_sum=abs(eye_h1[1])

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=1:
        if eye_h1[1]>=0:
            coef=1
        else:
            coef=-1
        val=val+coef*step
        MHScript.applyTarget('eyes/l-eye-corner1-up',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eye_h1=difference_new(MH_landmarks[45],Man_landmarks[45])

        error_sum=abs(eye_h1[0])

        # log.debug("left eye outer corner vertical:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('eyes/l-eye-corner1-up',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass

# left Eye inner corner horizontal

eye_h1=difference_new(MH_landmarks[42],Man_landmarks[42])

val=0
step=0.01
coef=1

error_sum=abs(eye_h1[0])

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=1:
        if eye_h1[0]>=0:
            coef=1
        else:
            coef=-1
        val=val+coef*step
        MHScript.applyTarget('eyes/l-eye-push2-in',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eye_h1=difference_new(MH_landmarks[42],Man_landmarks[42])

        error_sum=abs(eye_h1[0])

        # log.debug("left eye inner corner horizontal:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('eyes/l-eye-push2-in',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass

# left Eye outer corner vertical

eye_h1=difference_new(MH_landmarks[42],Man_landmarks[42])

val=0
step=0.01
coef=1

error_sum=abs(eye_h1[1])

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=1:
        if eye_h1[1]>=0:
            coef=1
        else:
            coef=-1
        val=val+coef*step
        MHScript.applyTarget('eyes/l-eye-corner2-up',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eye_h1=difference_new(MH_landmarks[42],Man_landmarks[42])

        error_sum=abs(eye_h1[0])

        # log.debug("left eye inner corner vertical:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('eyes/l-eye-corner2-up',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass

# Nose vertical position

eye_h1=difference_new(MH_landmarks[27],Man_landmarks[27])

val=0
step=0.01
coef=1

error_sum=abs(eye_h1[1])

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=1:
        if eye_h1[1]>=0:
            coef=1
        else:
            coef=-1
        val=val+coef*step
        MHScript.applyTarget('nose/nose-trans-up',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eye_h1=difference_new(MH_landmarks[27],Man_landmarks[27])

        error_sum=abs(eye_h1[0])

        # log.debug("Nose vertical position:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('nose/nose-trans-up',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass

# Nose vertical size

eye_h1=difference_new(MH_landmarks[30],Man_landmarks[30])

val=0
step=0.01
coef=1

error_sum=abs(eye_h1[1])

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=1:
        if eye_h1[1]>=0:
            coef=1
        else:
            coef=-1
        val=val+coef*step
        MHScript.applyTarget('nose/nose-scale-vert-incr',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eye_h1=difference_new(MH_landmarks[30],Man_landmarks[30])

        error_sum=abs(eye_h1[0])

        # log.debug("Nose vertical size:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('nose/nose-scale-vert-incr',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass

# Nose horizontal size

eye_h_MH=difference_new(MH_landmarks[35],MH_landmarks[31])
eye_h_MAN=difference_new(Man_landmarks[35],Man_landmarks[31])
eye_h1=eye_h_MH[1]-eye_h_MAN[1]

val=0
step=0.01
coef=1

error_sum=abs(eye_h1)

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=1:
        if eye_h1>=0:
            coef=1
        else:
            coef=-1
        val=val+coef*step
        MHScript.applyTarget('nose/nose-scale-horiz-incr',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eye_h_MH=difference_new(MH_landmarks[35],MH_landmarks[31])
        eye_h_MAN=difference_new(Man_landmarks[35],Man_landmarks[31])
        eye_h1=eye_h_MH[1]-eye_h_MAN[1]

        error_sum=abs(eye_h1)

        # log.debug("nose horizontal size:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('nose/nose-scale-horiz-incr',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass

































# Mouth horizontal size

eye_h_MH=difference_new(MH_landmarks[54],MH_landmarks[48])
eye_h_MAN=difference_new(Man_landmarks[54],Man_landmarks[48])
eye_h1=eye_h_MH[1]-eye_h_MAN[1]

val=0
step=0.01
coef=1

error_sum=abs(eye_h1)

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=1:
        if eye_h1>=0:
            coef=1
        else:
            coef=-1
        val=val+coef*step
        MHScript.applyTarget('mouth/mouth-scale-horiz-incr',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eye_h_MH=difference_new(MH_landmarks[54],MH_landmarks[48])
        eye_h_MAN=difference_new(Man_landmarks[54],Man_landmarks[48])
        eye_h1=eye_h_MH[1]-eye_h_MAN[1]

        error_sum=abs(eye_h1)

        # log.debug("mouth horizontal size:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('mouth/mouth-scale-horiz-incr',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass

# Mouth vertical size

eye_h_MH=difference_new(MH_landmarks[57],MH_landmarks[51])
eye_h_MAN=difference_new(Man_landmarks[57],Man_landmarks[51])
eye_h1=eye_h_MH[1]-eye_h_MAN[1]

val=0
step=0.01
coef=1

error_sum=abs(eye_h1)

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=1:
        if eye_h1>=0:
            coef=-1
        else:
            coef=1
        val=val+coef*step
        MHScript.applyTarget('mouth/mouth-scale-vert-incr',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eye_h_MH=difference_new(MH_landmarks[57],MH_landmarks[51])
        eye_h_MAN=difference_new(Man_landmarks[57],Man_landmarks[51])
        eye_h1=eye_h_MH[1]-eye_h_MAN[1]

        error_sum=abs(eye_h1)

        # log.debug("mouth horizontal size:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('mouth/mouth-scale-horiz-incr',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass

# mouth vertical position

eye_h1=difference_new(MH_landmarks[62],Man_landmarks[62])

val=0
step=0.01
coef=1

error_sum=abs(eye_h1[1])

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=1:
        if eye_h1[1]>=0:
            coef=1
        else:
            coef=-1
        val=val+coef*step
        MHScript.applyTarget('mouth/mouth-trans-up',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eye_h1=difference_new(MH_landmarks[62],Man_landmarks[62])

        error_sum=abs(eye_h1[0])

        # log.debug("mouth vertical position:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('mouth/mouth-trans-up',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass

# Lower lip vertical position

eye_h1=difference_new(MH_landmarks[58],Man_landmarks[58])

val=0
step=0.01
coef=1

error_sum=abs(eye_h1[1])

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=1:
        if eye_h1[1]>=0:
            coef=1
        else:
            coef=-1
        val=val+coef*step
        MHScript.applyTarget('mouth/mouth-lowerlip-height-incr',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eye_h1=difference_new(MH_landmarks[58],Man_landmarks[58])

        error_sum=abs(eye_h1[0])

        # log.debug("Lower lip vertical position:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('mouth/mouth-lowerlip-height-incr',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass

# Lower lip horizontal size

eye_h_MH=difference_new(MH_landmarks[55],MH_landmarks[59])
eye_h_MAN=difference_new(Man_landmarks[55],Man_landmarks[59])
eye_h1=eye_h_MH[1]-eye_h_MAN[1]

val=0
step=0.01
coef=1

error_sum=abs(eye_h1)

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=1:
        if eye_h1>=0:
            coef=1
        else:
            coef=-1
        val=val+coef*step
        MHScript.applyTarget('mouth/mouth-lowerlip-width-incr',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eye_h_MH=difference_new(MH_landmarks[55],MH_landmarks[59])
        eye_h_MAN=difference_new(Man_landmarks[55],Man_landmarks[59])
        eye_h1=eye_h_MH[1]-eye_h_MAN[1]

        error_sum=abs(eye_h1)

        # log.debug("lower lip horizontal size:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('mouth/mouth-lowerlip-width-incr',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass

# Upper lip vertical position

eye_h1=difference_new(MH_landmarks[51],Man_landmarks[51])

val=0
step=0.01
coef=1

error_sum=abs(eye_h1[1])

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=1:
        if eye_h1[1]>=0:
            coef=1
        else:
            coef=-1
        val=val+coef*step
        MHScript.applyTarget('mouth/mouth-upperlip-height-incr',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eye_h1=difference_new(MH_landmarks[51],Man_landmarks[51])

        error_sum=abs(eye_h1[0])

        # log.debug("Upper lip vertical position:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('mouth/mouth-upperlip-height-incr',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass

# Upper lip horizontal size

eye_h_MH=difference_new(MH_landmarks[53],MH_landmarks[49])
eye_h_MAN=difference_new(Man_landmarks[53],Man_landmarks[49])
eye_h1=eye_h_MH[1]-eye_h_MAN[1]

val=0
step=0.01
coef=1

error_sum=abs(eye_h1)

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=1:
        if eye_h1>=0:
            coef=1
        else:
            coef=-1
        val=val+coef*step
        MHScript.applyTarget('mouth/mouth-upperlip-width-incr',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eye_h_MH=difference_new(MH_landmarks[53],MH_landmarks[49])
        eye_h_MAN=difference_new(Man_landmarks[53],Man_landmarks[49])
        eye_h1=eye_h_MH[1]-eye_h_MAN[1]

        error_sum=abs(eye_h1)

        # log.debug("Upper lip horizontal size:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('mouth/mouth-upperlip-width-incr',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass

# Mouth corner vertical position

eye_h1=difference_new(MH_landmarks[48],Man_landmarks[48])

val=0
step=0.01
coef=1

error_sum=abs(eye_h1[1])

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=1:
        if eye_h1[1]>=0:
            coef=1
        else:
            coef=-1
        val=val+coef*step
        MHScript.applyTarget('mouth/mouth-angles-up',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eye_h1=difference_new(MH_landmarks[48],Man_landmarks[48])

        error_sum=abs(eye_h1[0])

        # log.debug("Upper lip vertical position:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('mouth/mouth-angles-up',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass

# Mouth lowerlip middle vertical position

eye_h1=difference_new(MH_landmarks[66],Man_landmarks[66])

val=0
step=0.01
coef=1

error_sum=abs(eye_h1[1])

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=1:
        if eye_h1[1]>=0:
            coef=1
        else:
            coef=-1
        val=val+coef*step
        MHScript.applyTarget('mouth/mouth-lowerlip-middle-up',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eye_h1=difference_new(MH_landmarks[66],Man_landmarks[66])

        error_sum=abs(eye_h1[0])

        # log.debug("Lower lip middle vertical position:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('mouth/mouth-lowerlip-middle-up',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass

# Mouth upperlip middle vertical position

eye_h1=difference_new(MH_landmarks[62],Man_landmarks[62])

val=0
step=0.01
coef=1

error_sum=abs(eye_h1[1])

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=1:
        if eye_h1[1]>=0:
            coef=1
        else:
            coef=-1
        val=val+coef*step
        MHScript.applyTarget('mouth/mouth-upperlip-middle-up',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eye_h1=difference_new(MH_landmarks[62],Man_landmarks[62])

        error_sum=abs(eye_h1[0])

        # log.debug("Upper lip middle vertical position:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('mouth/mouth-upperlip-middle-up',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass


# Chin vertical position

eye_h1=difference_new(MH_landmarks[8],Man_landmarks[8])

val=0
step=0.01
coef=1

error_sum=abs(eye_h1[1])

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=1:
        if eye_h1[1]>=0:
            coef=1
        else:
            coef=-1
        val=val+coef*step
        MHScript.applyTarget('chin/chin-height-incr',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eye_h1=difference_new(MH_landmarks[8],Man_landmarks[8])

        error_sum=abs(eye_h1[0])

        # log.debug("chin vertical position:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('chin/chin-height-incr',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass

# Chin horizontal size

eye_h_MH=difference_new(MH_landmarks[5],MH_landmarks[13])
eye_h_MAN=difference_new(Man_landmarks[5],Man_landmarks[13])
eye_h1=eye_h_MH[1]-eye_h_MAN[1]

val=0
step=0.01
coef=1

error_sum=abs(eye_h1)

error_all=[]
lowest_val=[]
minim_idx=0

error_all.append(error_sum)
lowest_val.append(val)

for count in range(loop):
    minim=min(error_all)
    minim_idx=error_all.index(min(error_all))
    if error_sum>=1:
        if eye_h1>=0:
            coef=1
        else:
            coef=-1
        val=val+coef*step
        MHScript.applyTarget('chin/chin-bones-incr',val)

        MHScript.screenShot('initial.png')

        MH_img='initial.png'
        MH_landmarks=txt_gen(MH_img,detector,predictor)

        eye_h_MH=difference_new(MH_landmarks[5],MH_landmarks[13])
        eye_h_MAN=difference_new(Man_landmarks[5],Man_landmarks[13])
        eye_h1=eye_h_MH[1]-eye_h_MAN[1]

        error_sum=abs(eye_h1)

        # log.debug("Upper lip horizontal size:")
        # log.debug(error_sum)

        error_all.append(error_sum)
        lowest_val.append(val)

if minim<error_sum:
    val=lowest_val[minim_idx]
    MHScript.applyTarget('chin/chin-bones-incr',val)
    MHScript.screenShot('initial.png')
    MH_landmarks=txt_gen(MH_img,detector,predictor)
    log.debug("used min")
    error_full.append(minim)
else:
    log.debug("Minimum Error:")
    log.debug(error_sum)
    error_full.append(error_sum)
    pass

log.debug(error_full)

file_name=MH_img+'.txt'
np.savetxt(file_name,MH_landmarks,fmt="%d")
output=cv2.imread('initial.png')
for p in MH_landmarks:
    draw_point(output, (int(p[0]),int(p[1])), (0,0,255))

cv2.imwrite('initial_dotted.png',output)
