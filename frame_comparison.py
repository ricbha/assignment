# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 11:44:07 2018

@author: MAHE
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 10:17:23 2018

@author: MAHE
"""

import cv2
import sys
import numpy as np

if len(sys.argv) < 2:
   video = cv2.VideoCapture('C:\\Users\\MAHE\\Desktop\\New folder\\a\\person01_walking_d3_uncomp.avi')
else:
   video = cv2.VideoCapture(sys.argv[1])

#video = cv2.VideoCapture('C:\\Users\\MAHE\\Desktop\\New folder\\a\\person01_walking_d2_uncomp.avi')
ret, last_frame = video.read()
ret, current_frame = video.read()
gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
i = 0
sum=0
asd = np.empty(11)
#a=0
while(ret):
    # We want two frames- last and current, so that we can calculate the different between them.
    # Store the current frame as last_frame, and then read a new one
    last_frame = current_frame
    ret, current_frame = video.read()
    if ret==0:
        break
    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(last_frame, current_frame)
    i += 1
    sum=(sum+np.mean(diff))/i
    asd = np.append(asd,np.mean(diff))
    if i % 10 == 0:
        i = 0
    print (np.mean(current_frame))
    print (np.mean(diff))
    
    z=np.zeros(11)
 #   if a<11:
  #      z[a]=np.mean(diff)
   # a=a+1        
   # n=np.saveint(np.mean(diff))
    if np.mean(diff) > 10:
        print("Achtung! Motion detected.")
        
# Find the absolute difference between frames
    cv2.imshow('Video',diff)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        
        break
    ret,current_frame=video.read()
# When everything done, release the capture
print(sum)
p=np.amax(asd)
print("max", p)
video.release()
cv2.destroyAllWindows()