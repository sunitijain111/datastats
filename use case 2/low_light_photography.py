
import cv2
import os
import numpy as np
path= os.getcwd()+ '/'+  'folder2'


files = list([os.path.join(path, f) for f in os.listdir(path)])

f= files[0]
image= cv2.imread(f).astype(np.float)
cnt=0
for temp in files[1:]:
    temp_img= cv2.imread(temp)
    image+=temp_img
    cnt+=1
    if cnt==5:
        break
image/=cnt
      
image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
cv2.imwrite('output.png', image)
cv2.imshow('after',cv2.imread('output.png'))
cv2.imshow('before',cv2.imread(files[0]))
cv2.waitKey(0)
cv2.destroyAllWindows()     