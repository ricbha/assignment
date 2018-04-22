# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 09:53:48 2018

@author: MAHE
"""

import numpy as np
import cv2


cap = cv2.VideoCapture("C:\\Users\\MAHE\\Desktop\\New folder\\a\\person01_walking_d2_uncomp.avi")
ret, frame = cap.read()
while(ret):
  # Capture frame-by-frame
   ret, frame = cap.read()

   # Our operations on the frame come here
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   blur = cv2.GaussianBlur(gray,(5,5),0)
   

   ret, thresh_img = cv2.threshold(blur,91,255,cv2.THRESH_BINARY)

   contours =  cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2]
   for c in contours:
        cv2.drawContours(frame, [c], -1, (0,255,0), 3)
        M=cv2.moments(c)
        if M["m00"] !=0:
            cX=int(M["m10"] / M["m00"])
            cY=int(M["m01"] / M["m00"])
        else:
            cX, cY=0,0
            
        cv2.circle(frame,(cX, cY), 7, (255,255,255),-1)
     # Display the resulting frame
   cv2.imshow('frame',frame)
   if cv2.waitKey(1) & 0xFF == ord('q'):
        break
   ret, frame=cap.read()
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()