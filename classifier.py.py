# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 11:51:18 2018

@author: MAHE
"""

import os
import re
import sys
import cv2
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn import cross_validation, grid_search
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.manifold import TSNE

#DirBase = 'E:/OneDrive/OneDrive - Manipal University/Research 2017/Cleed/Codes/Test/TensorFlow_SVM/tutorials/image/imagenet/TUTORIAL_DIR/images/'
DirBase = 'C:\\Users\\MAHE\\Desktop\\New folder\\a'
DirBase = DirBase.replace("\\","/")
model_dir = 'C:\\Users\\MAHE\\Desktop\\New folder\\a'
#c://users//mahe//desktop//cv
def Read_Images_from_Folders(DirBase):
    """
     returns the full path of all the images in the subfolders of DirBase
    """
    DirList = os.listdir(DirBase)# returns all folders in directory DirBase
    list_images = []


    for dir in DirList:
        DirName = DirBase + '\\' + dir +'\\'
        dir_images = [DirName+f for f in os.listdir(DirName) if re.search('avi', f)]

        list_images = np.append(list_images,dir_images,axis=0)
    return list_images

def program(x):
    """
        Load the pretrained graph (i.e. inception v3 with weights)
        """
    if len(sys.argv) < 2:
       video = cv2.VideoCapture(x)
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
        #print (np.mean(current_frame))
        #print (np.mean(diff))
        
        
     #   if a<11:
      #      z[a]=np.mean(diff)
       # a=a+1        
       # n=np.saveint(np.mean(diff))
        #if np.mean(diff) > 10:
            #print("Achtung! Motion detected.")
            
    # Find the absolute difference between frames
        
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            
            break
        ret,current_frame=video.read()
# When everything done, release the capture
    
    p=np.amax(asd)
    
    video.release()
    return p
def extract_features(list_images):
    nb_features = 2048
    features = np.empty((len(list_images),nb_features))
    labels = []
    
    
  
    for ind, vid in enumerate(list_images):
        features[ind,:]=program(vid)
                

                
                
              
#                labels.append(re.split('_\d+',image.split('/')[-2])[0])
        labels.append(vid.split('\\')[-2])
    return features, labels

def train_svm_classifer(features, labels):
    """
    train_svm_classifer will train a SVM, saved the trained and SVM model and
    report the classification performance

    features: array of input features
    labels: array of labels associated with the input features
    model_output_path: path for storing the trained svm model
    """
    # save 20% of data for performance evaluation
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.2)

    param = [
            {
             "kernel": ["linear"],
             "C": [1, 10, 100, 1000]
        },

        {
            "kernel": ["rbf"],
            "C": [1, 10, 100, 1000],
            "gamma": [1e-5, 1e-6, 1e-7]
        }
       
    ]

    # request probability estimation
    svm = SVC(probability=True)

    # 10-fold cross validation, use 4 thread as each fold and each parameter set can be train in parallel
    clf = grid_search.GridSearchCV(svm, param,cv=10, verbose=3)

    clf.fit(X_train, y_train)

#    if os.path.exists(model_output_path):
#        joblib.dump(clf.best_estimator_, model_output_path)
    
#    else:
#        print("Cannot save trained svm model to {0}.".format(model_output_path))

    print("\nBest parameters set:")
    print(clf.best_params_)

    y_predict=clf.predict(X_test)

    labels=sorted(list(set(labels)))
    print("\nConfusion matrix:")
    print("Labels: {0}\n".format(",".join(labels)))
    print(confusion_matrix(y_test, y_predict, labels=labels))

    print("\nClassification report:")
    print(classification_report(y_test, y_predict))

"""
Main

"""

list_images = Read_Images_from_Folders(DirBase)

features,labels = extract_features(list_images)


train_svm_classifer(features,labels)
