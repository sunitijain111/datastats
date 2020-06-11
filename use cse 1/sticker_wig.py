import cv2
import datetime
import os
import numpy as np
#these are the cascades: thay are already present in the opencv
cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml') 

#function ot detect the face

# gray image of our face, original image:
def detection(grayscale, img):
    
    #1.3 is the scale factor , 5 is te min neighbours
    face = cascade_face.detectMultiScale(grayscale, 1.3, 5)
    
    #4 corrdinates in faces
    for (x_face, y_face, w_face, h_face) in face:
        

        img_wig= cv2.imread('hair2.png',-1)
        
        ## adjusing the width & dept
        width, depth, _= img_wig.shape
        width= int(w_face + 0.1* width)
        depth= int(h_face* 2/3)
        img_wig= cv2.resize(img_wig, (width, depth))
        
        ## coordinates
        centre_x= x_face+ w_face/2
        x_wig= int(centre_x-width/2)
        
        y_wig= int(y_face- (depth*2/3) )
        
        ##cropping 
        
        if x_wig<0:
            x_wig=0
        if y_wig<0    :
            y_wig=0
        y2=int(y_wig+depth)
        if(y2>= img.shape[0]):
            y2= img.shape[0]
        x2= int(x_wig+width)
        if(x2>= img.shape[1]):
            x2= img.shape[1]
           
           
        img_wig= img_wig[0: y2-y_wig, 0:x2-x_wig] 
        bg=  img[y_wig: y2, x_wig: x2]
        sg = np.atleast_3d(255 - img_wig[:, :,3])/255.0
        np.multiply(bg, sg, out=bg, casting="unsafe")
        np.add(bg, 255-img_wig[:, :,0:3] * np.atleast_3d(img_wig[:, :,3]), out=bg)
        img[y_wig: y2, x_wig: x2] = bg
        
        return img 
    #return image        
    


vc = cv2.VideoCapture(0) 
#path
path= os.getcwd()

# Create directory
dirName = 'tempimage_folder'
try:
        os.mkdir(dirName)
except FileExistsError:
        print("Directory " , dirName ,  " already exists")
path= path+'/'+dirName
cnt=0
while cnt<500:
    #read status of camera and frame
    _, img = vc.read() 
    
    #convert image ot grayscale
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    #reuslt from detection function
    final = detection(grayscale, img) 
    
    if final is not None:
        cv2.imshow('Video', final) 
    
    #name of our image, wiht current time , so that it has new name each time.
    string = "pic"+str(datetime.datetime.now())+".jpg"
    
    
    #save image
    cv2.imwrite(os.path.join(path, string),final)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
    
    cnt+=1

vc.release() 
cv2.destroyAllWindows() 

