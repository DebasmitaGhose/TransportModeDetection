# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 01:46:04 2018

@author: User
"""

import math
import numpy as np
import scipy as sc
import pandas as pd
import xlwt
import csv

####################################################################################################
######################################### Change Point Segmentation ###################################
###################################################################################################

########################################### Step 1 ###################################################

book = xlwt.Workbook(encoding="utf-8")
sheet1 = book.add_sheet("Sheet 1")
sheet1.write(0,0,"Latitude")
sheet1.write(0,1,"Longitude")
sheet1.write(0,2,"Average Velocity")
sheet1.write(0,3,"Average Acceleration")
sheet1.write(0,4,"Mode")
df = pd.read_csv('C:\\Users\\N1600060B\\Documents\\Transport Mode Detection\\DataSet - Singapore\\Trip 4.csv',sep=',',header=None)
#ff=open('dumpfile.csv','w+')
data = df.values
#print data
# data = datat.tolist()
#print len(data[:,6])
cValues = np.where((data[:,6])>1.6)[0]
#print len(cValues)
segment = data[3:cValues[0]+1,:]
#print segment
segments = [segment]
#print segments
start = 4
end = -1
for i in range(3,len(cValues)):
    if cValues[i]!=cValues[i-1]+1:
        end = cValues[i-1]+1
        seg1 = data[start:end,:]
        start2 = cValues[i-1]+1
        end2 = cValues[i]
        seg2 = data[start2:end2,:]
        if len(seg1)!=0:
            #print len(seg1)
            segments.append(seg1)
        if len(seg2)!=0:
            segments.append(seg2)
            #print len(seg2)
        start = cValues[i]
#print segments
#print len(segments)
################################################### Step 2 ###########################################
        
newseglist = []
newseglist.append(segments[0])
mergeIndex=0
for ii in range(1,len(segments)):
    if len(segments[ii])<=3:
        newseglist[mergeIndex] = np.concatenate((newseglist[mergeIndex],segments[ii]),axis=0)
    else:
        newseglist.append(segments[ii])
        mergeIndex = mergeIndex+1
#-------------------------------debugging-----------------------------------#
# print len(newseglist)
# for jj in range(0,len(newseglist)):
#   print len(newseglist[jj])
# print newseglist
#-----------------------------end of debugging------------------------------#
   
########################################### Step 3 ###############################################    

seglistReborn = []
mergeMode = 0
mergeIndex = 0
for j in range(0,len(newseglist)):
    if len(newseglist[j])>10:
        seglistReborn.append(newseglist[j])
        mergeMode=0
        seglistBorn = [seglistReborn]
    elif len(newseglist[j])<=10 and mergeMode==0:
        seglistReborn.append(newseglist[j])
        mergeIndex = len(seglistReborn)-1
        mergeMode=1
        seglistBorn = [seglistReborn]
    elif len(newseglist[j])<=10 and mergeMode==1:
        seglistReborn[mergeIndex] = np.concatenate((seglistReborn[mergeIndex],newseglist[j]),axis=0)
        seglistBorn = [seglistReborn]
    #print seglistBorn[0]
del newseglist
#-------------------------------debugging-----------------------------------#
#print len(seglistReborn)
#for jj in range(0,len(seglistReborn)):
#     print len(seglistReborn[jj])
#print seglistReborn[1][0][0]
#-----------------------------end of debugging------------------------------#

#################################### Step 4 / Calculating Features ########################################

array_segment=[]
#meanData = np.sum(seglistReborn[0][:,[6,7]],axis=0)/len(seglistReborn[0])
#print meanData
for i in range(1,(len(seglistBorn[0]))):
    #print seglistBorn[0][i]
    if len(seglistBorn[0][i]) is not 0:
        meanVelocity = sum(seglistBorn[0][i][:,[6]])/len(seglistBorn[0][i])
        meanAcceleration = sum(seglistBorn[0][i][:,[7]])/len(seglistBorn[0][i])
    #print "Length of Segment = "
    #print len(seglistReborn[i])
    #print "Start Lattitude = "
    #print seglistReborn[i][0][0]
    #print "Start Longitude = "
    #print seglistReborn[i][0][1]
    #print "End Lattitude = "
    #print seglistReborn[i+1][0][0]
    #print "End Longitude = "
    #print seglistReborn[i+1][0][1]
    #print "Mode = "
    #print seglistReborn[i][0][8]
    #print " Features = "
    #print np.sum(seglistReborn[i][:,[6,7]],axis=0)/len(seglistReborn[i])
    #print "--------------------------------------"
        #print len(seglistBorn[0][i])
        sheet1.write(i,0,seglistReborn[i][0][0])
        sheet1.write(i,1,seglistReborn[i][0][1])
        sheet1.write(i,2,meanVelocity[0])
        sheet1.write(i,3,meanAcceleration[0])
        sheet1.write(i,4,seglistReborn[i][0][8])
book.save("trip_4_segments.csv")

####################################################################################################
################################## Classification of Segmented Data ######################################
####################################################################################################

import csv
import numpy as np

f1 = file('C:\\Users\\N1600060B\\segments_train_data_2.csv','r')
f2 = file('C:\\Users\\N1600060B\\trip_segments.csv','r')

c1=csv.reader(f1)
c2=csv.reader(f2)

#################################### Header contains feature names ######################################

#training data
row_train=next(c1)                 
feature_names_train=np.array(row_train)
#print feature_names_train 

#testing data
row_test=next(c2)                  
feature_names_test=np.array(row_test)
#print feature_names_test

################################# Load dataset and target classes #########################################

mode_X_train,mode_y_train=[],[]    
mode_X_test,mode_y_test=[],[]

#training data
for row_train in c1:              
    mode_X_train.append(row_train)
    mode_y_train.append(row_train[4])
mode_X_train = np.array(mode_X_train)
mode_y_train = np.array(mode_y_train)
#print mode_X_train
#print mode_y_train[1]

#testing data
for row_test in c2:               
    mode_X_test.append(row_test)
    mode_y_test.append(row_test[4])
mode_X_test = np.array(mode_X_test)
mode_y_test = np.array(mode_y_test)
#print mode_X_test
#print mode_y_test[1]

#####################################2 Retain required columns #########################################

#training data
mode_X_train = mode_X_train[:, [2,3]]   
feature_names_train = feature_names_train[[2,3]]
#print feature_names_train
#print mode_X_train

#testing data
mode_X_test = mode_X_test[:, [2,3]]     
feature_names_test = feature_names_test[[2,3]]
#print feature_names_test
#print mode_X_test

########################################### Encode mode #############################################

from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()

#training data
label_encoder_train = enc.fit(mode_y_train[:])       
print("Categorical Classes Training:", label_encoder_train.classes_)
integer_classes_train = label_encoder_train.transform(label_encoder_train.classes_)
print("Integer Classes Training:", integer_classes_train)
t_1 = label_encoder_train.transform(mode_y_train[:])
mode_y_train[:] = t_1
#print (mode_X_train[4], mode_y_train[4])

#testing data
label_encoder_test = enc.fit(mode_y_test[:])       
print("Categorical Classes Testing:", label_encoder_test.classes_)
integer_classes_test = label_encoder_test.transform(label_encoder_test.classes_)
print("Integer Classes Testing:", integer_classes_test)
t_2 = label_encoder_test.transform(mode_y_test[:])
mode_y_test[:] = t_2
#print (mode_X_test[4], mode_y_test[4])

####################################### Classification #################################################

#decision tree
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=30, min_samples_leaf=3, presort=True,min_samples_split=20)
clf = clf.fit(mode_X_train,mode_y_train) 
#print clf.predict(mode_X_test)

#random forests
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(n_jobs=3, criterion='entropy', max_depth=30,min_samples_split=20,min_samples_leaf=5)
clf_rf = clf_rf.fit(mode_X_train,mode_y_train)

############################# Function to calculate classification accuracy ##################################

from sklearn import metrics
    
def measure_performance(X,y,clf, show_accuracy=True, show_classification_report=True, show_confusion_matrix=True):
    y_pred=clf.predict(X)
    if show_accuracy:
        print("Accuracy:{0:.3f}".format(metrics.accuracy_score(y,y_pred)),"\n")
    if show_classification_report:
        print("Classification Report")
        print(metrics.classification_report(y,y_pred),"\n")
    if show_confusion_matrix:
        print("Confusion Matrix")
        print(metrics.confusion_matrix(y,y_pred),"\n")
    
#measure_performance(mode_X_train,mode_y_train,clf, show_accuracy=True, show_classification_report=True, show_confusion_matrix=True)

############# Evaluating the performance of the classifiers on the testing data ###################################

clf_dt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=30, min_samples_leaf=3, presort=True,min_samples_split=20)
clf_dt.fit(mode_X_test, mode_y_test)
clf_rf_test = RandomForestClassifier(n_jobs=3, criterion='entropy', max_depth=30,min_samples_split=20,min_samples_leaf=5)
clf_rf_test = clf_rf_test.fit(mode_X_test,mode_y_test)
print "----------------------Decision Tree----------------------------------"
measure_performance(mode_X_test, mode_y_test,clf_dt)
print "----------------------Random Forests---------------------------------"
measure_performance(mode_X_test, mode_y_test,clf_rf_test)
