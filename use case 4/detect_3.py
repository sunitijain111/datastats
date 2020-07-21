#on a single image, lipstick, liner, lashes, brows thickem
# import the necessary packages
from imutils import face_utils
import numpy as np
import dlib
import cv2
import sys

shape_predictor= "shape_predictor_68_face_landmarks.dat"
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)
  
def apply_lipstick(img)  :
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray, 1)
    
    # loop over the face detections
    a=[]
    for (i, rect) in enumerate(rects):
    	shape = predictor(gray, rect)
    	shape = face_utils.shape_to_np(shape)
    	a=list(face_utils.FACIAL_LANDMARKS_IDXS.items())
   
    if(len(a)):
        name, (i, j)=a[0]
        points= []
        
        for (x, y) in shape[i:j]:
            points.append([x,y])
        points= np.reshape(points, (-1, 1, 2))    
        cv2.fillPoly(img, [points], (84,44, 150), 8)        
    
    return img

def apply_lashes(img)  : 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray, 1)
    
    # loop over the face detections
    a=[]
    shape=[]
    for (i, rect) in enumerate(rects):
    	shape = predictor(gray, rect)
    	shape = face_utils.shape_to_np(shape)
    	a=list(face_utils.FACIAL_LANDMARKS_IDXS.items())
   
    if(len(a)):
      name,(i,j)  = a[4]   #left eye
    
    points= []
    cnt=0  
    
    if len(shape):
     for (x, y) in shape[i:j]:
            points.append([x,y]) 
            cnt+=1
            if cnt==4:
                break
            
    if len(points):  #liner
        left= points[0]
        left= (left[0], left[1])
        right= points[1]
        right= (right[0], right[1])
        pt= (0,0,0)
        cv2.line(img,left, right, pt,1)
        left= points[1]
        left= (left[0], left[1])
        right= points[2]
        right= (right[0], right[1])
        cv2.line(img,left, right, pt,1)
        left= points[2]
        left= (left[0], left[1])
        right= points[3]
        right= (right[0], right[1])
        cv2.line(img,left, right, pt,1)
        
        #extreme left el, er
        el= points[0]         
        er= points[2]
        

        
        #diff in y aka heihgt
        x_left, y_left, x_r, y_r= el[0] , el[1], er[0], er[1]
        weye= abs(x_r- x_left)*2
        
        
        lash= cv2.imread("one_lash.jpeg",-1)
        if lash is None:
            print(" not opened")
            return img
        lash= cv2.resize(lash,(weye, int(weye/4)))
        #print(lash.shape)
        
        roi= img[ y_left-lash.shape[0]:y_left, x_r-lash.shape[1]:x_r] 
        
        for x in range(0, lash.shape[0]):
            for y in range(0,lash.shape[1]):
                a,b,c= lash[x,y]
                if not(a>=150 or b>=150 or c>=150):
                 roi[x,y]= (a,b,c)
                    
                    
        img[ y_left-lash.shape[0]:y_left, x_r-lash.shape[1]:x_r]  = roi
        
        ############right one
    a=[]
    shape=[]
    for (i, rect) in enumerate(rects):
    	shape = predictor(gray, rect)
    	shape = face_utils.shape_to_np(shape)
    	a=list(face_utils.FACIAL_LANDMARKS_IDXS.items())    
    if(len(a)):
      name,(i,j)  = a[5]   #right eye,    
    points=[]
    cnt=0    
    if len(shape):
         for (x, y) in shape[i:j]:
                points.append([x,y]) 
                cnt+=1
                if cnt==4:
                    break
            
    if len(points):  #liner
        left= points[0]
        left= (left[0], left[1])
        right= points[1]
        right= (right[0], right[1])
        pt= (0,0,0)
        cv2.line(img,left, right, pt,1)
        left= points[1]
        left= (left[0], left[1])
        right= points[2]
        right= (right[0], right[1])
        cv2.line(img,left, right, pt,1)
        left= points[2]
        left= (left[0], left[1])
        right= points[3]
        right= (right[0], right[1])
        cv2.line(img,left, right, pt,1)
        
        el= points[1]         
        er= points[3]
        
            
        x_left, y_left, x_r, y_r= el[0] , el[1], er[0], er[1]
        weye= abs(x_r- x_left)*2
        
        lash= cv2.imread("one_lash_r.jpeg",-1)
        if lash is None:
            print(" not opened")
            return img
        lash= cv2.resize(lash,(weye, int(weye/4)))
        #print(lash.shape)
        
        roi= img[ y_left-lash.shape[0]:y_left, x_left:x_left+lash.shape[1] ]
        
        for x in range(0, lash.shape[0]):
            for y in range(0,lash.shape[1]):
                a,b,c= lash[x,y]
                if not(a>=150 and b>=150 and c>=150):
                 roi[x,y]= (a,b,c)
                    
                    
        img[ y_left-lash.shape[0]:y_left, x_left:x_left+lash.shape[1] ] = roi
        return img 
def eye_brow(img)        :
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray, 1)
    
    # loop over the face detections
    a=[]
    shape=[]
    for (i, rect) in enumerate(rects):
    	shape = predictor(gray, rect)
    	shape = face_utils.shape_to_np(shape)
    	a=list(face_utils.FACIAL_LANDMARKS_IDXS.items())
    
    
    if(len(a)):
      name,(i,j)  = a[2]   #left brows   ,  5 pts 
    points=[]   
    if len(shape):
         for (x, y) in shape[i:j]:
                points.append([x,y]) 
         cnt=0       
         for (x, y) in shape[i:j]:
                cnt+=1
                if cnt>=3:
                 points.insert(0,[x,y+1])
         points= np.reshape(points, (-1, 1, 2))        
         cv2.fillPoly(img, [points], (0,0,0), 4)            
    
        
        
    ### eye brow, right
    if(len(a)):
      name,(i,j)  = a[3]   #right brows   ,  5 pts 
    points=[]   
    if len(shape):
         for (x, y) in shape[i:j]:
                points.append([x,y]) 
         cnt=0       
         for (x, y) in shape[i:j]:
                cnt+=1
                if cnt<=3:
                 points.insert(0,[x,y+1])
         points= np.reshape(points, (-1, 1, 2))        
         cv2.fillPoly(img, [points], (0,0,0), 4)                  
    return img
c=0
def view(img):
    if img is None or img.shape[0]==0 or img.shape[1]==0: return
    cv2.imshow('image press any key to close', img)
    global c
    cv2.imwrite("edited"+str(c)+img_name,stack[-1])
    c+=1
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

count=0
stack=[]

def ask_image(): 
    ##user interaction
    print("enter the image name: ")
    img_name= input()
    global count
    img = cv2.imread(img_name)
    if img is None:
        if count==20:
            print("maximum limit reached re-run program")
            sys.exit()
            
        print(" no such image exits\n press 1 to reenter name \n press any other key to exit")    
        x = input()
        if x=="1":
            count+=1          
            ask_image()
        else:
            sys.exit()
    else:
        stack.append(img)
        return [img, img_name]

       
def choice(img,img_name):
    print(" enter 1 to apply lipstick\n enter 2 to apply lashes and liner\n enter 3 to adjust brows \n enter 4 view \n enter any other key to exit and save")
    x= input()
    if x=="1":
        curr= stack[-1]
        curr= apply_lipstick(curr)
        stack.append(curr)
        view(curr)
        choice(img,img_name)
    elif x=="2":
        curr= stack[-1]
        curr= apply_lashes(curr)
        stack.append(curr)
        view(curr)
        choice(img,img_name)  
    elif x=="3":
        curr= stack[-1]
        curr=eye_brow(curr)
        stack.append(curr)
        view(curr)
        choice(img,img_name)
    elif x=="4":
        view(stack[-1])
        choice(img,img_name)   
    else:
        cv2.imwrite("edited"+img_name,stack[-1])   
        print(" file saved ")
        sys.exit()
    
[img, img_name]= ask_image()    
choice(img, img_name)    