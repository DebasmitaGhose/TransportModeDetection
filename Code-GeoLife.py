# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 01:44:14 2018

@author: User
"""

import math
import numpy as np
import scipy as sc
import pandas as pd
import xlwt
import csv

###################################################################################################################################### Change Point Segmentation ##########################################
####################################################################################################

############################################### Step 1 ###############################################

book = xlwt.Workbook(encoding="utf-8")
sheet1 = book.add_sheet("Sheet 1")
sheet1.write(0,0,"Latitude")
sheet1.write(0,1,"Longitude")
sheet1.write(0,2,"Average Velocity")
sheet1.write(0,3,"Average Acceleration")
sheet1.write(0,4,"Mode")
df = pd.read_csv('C:\\Users\\N1600060B\\Documents\\Transport Mode Detection\\Final\\test.csv',sep=',',header=None)
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

############################################## Step 2 ###############################################
        
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
   
############################################# Step 3 ################################################  

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

##################################### Step 4 / Calculating Features #######################################
array_segment=[]
meanData = np.sum(seglistReborn[0][:,[6,7]],axis=0)/len(seglistReborn[0])
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
book.save("segments_train_1.csv")

###################################################################################################
################################## Classification of Segmented Data #####################################
####################################################################################################

print "########################## Classification of Segmented Data ########################################"

with open('C:\\Users\\N1600060B\\segments_data.csv','r') as csvfile:
    mode_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    
####################################### Header contains feature names ###################################
    
    row=next(mode_reader)
    feature_names=np.array(row)
    #print feature_names
    
########################################## Load dataset and target classes ################################
    
    mode_X,mode_y=[],[]
    for row in mode_reader:
        mode_X.append(row)
        mode_y.append(row[4])
    mode_X = np.array(mode_X)
    mode_y = np.array(mode_y)
    #print mode_X
    #print mode_y
    
########################################## Retain required columns #####################################
    
    mode_X = mode_X[:, [2,3]]
    feature_names = feature_names[[2,3]]
    #print feature_names
    #print mode_X
    
############################################ Encode mode ############################################
    
    from sklearn.preprocessing import LabelEncoder
    enc = LabelEncoder()
    label_encoder = enc.fit(mode_y[:])
    print("Categorical Classes:", label_encoder.classes_)
    integer_classes = label_encoder.transform(label_encoder.classes_)
    print("Integer Classes:", integer_classes)
    t = label_encoder.transform(mode_y[:])
    mode_y[:] = t
    #print (mode_X[4], mode_y[4])
    
##################################### Separate training and test sets ######################################
    
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(mode_X, 
                                                        mode_y, 
                                                        test_size=0.1, 
                                                        random_state=30, 
                                                        stratify=None)
    #print X_test
    
##################################### Decision tree ###################################################
    
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(criterion='entropy',
                                      splitter='best',
                                      max_depth=30, 
                                      min_impurity_split=1e-09, 
                                      presort = True, 
                                      min_samples_leaf=3, 
                                      min_samples_split=20)
    clf = clf.fit(X_train,y_train)
    #print clf
    
####################################### Random Forests ###############################################

    from sklearn.ensemble import RandomForestClassifier
    clf_rf_train = RandomForestClassifier(n_jobs=3,
                                         criterion='entropy',
                                         max_depth=30,
                                         min_samples_split=20,
                                         min_samples_leaf=5,
                                         verbose=0,
                                         warm_start=False)
    clf_rf_train = clf_rf_train.fit(X_train,y_train)
    
############################### Measure the accuracy of the decision tree ###################################
    
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
    
    print "--------------------Performance of the Decision Tree on the Segmented Training Data---------------------------------"
    measure_performance(X_train,y_train,clf, show_accuracy=True, show_classification_report=True, show_confusion_matrix=True)
    
    print "--------------------Performance of the Random Forest on the Segmented Training Data----------------------------------"
    measure_performance(X_train,y_train,clf_rf_train, show_accuracy=True, show_classification_report=True, show_confusion_matrix=True)
    
################## Evaluating the performance of the decision tree on the testing data ############################
    
    clf_dt = tree.DecisionTreeClassifier(criterion='entropy', 
                                         splitter='best',  
                                         min_impurity_split=1e-09, 
                                         max_depth=30, 
                                         presort = True, 
                                         min_samples_leaf=3, 
                                         min_samples_split=20)
    clf_dt.fit(X_train, y_train)
    x = clf_dt.predict(X_test)
    #for i in range(1,(len(X_test))):
    #    print x[i]
    #    print y_test[i]
    
    print "------------------------Performance of the Decision Tree on the Segmented Testing Data-----------------------------------"
    measure_performance(X_test,y_test,clf_dt)

################# Evaluating the performance of the random forest on the testing data ############################

    clf_rf_test = RandomForestClassifier(n_jobs=3,
                                        criterion='entropy',
                                        max_depth=30,
                                        min_samples_split=20,
                                        min_samples_leaf=5,
                                        verbose=0,
                                        warm_start=False)
    clf_rf_test = clf_rf_test.fit(X_train,y_train)
    y = clf_rf_test.predict(X_test)
    
    print "-------------------------Performance of the Random Forests on the Segmented Testing Data-----------------------------------"
    measure_performance(X_test,y_test,clf_rf_test)
    
    
################################# Exporting results to a CSV File #########################################

#print len(X_test)
book_result = xlwt.Workbook(encoding="utf-8")
sheet2 = book.add_sheet("Sheet 2")
sheet2.write(0,0,"Average Velocity")
sheet2.write(0,1,"Average Acceleration")
sheet2.write(0,2,"Mode - Ground Truth")
sheet2.write(0,3,"Mode - Predicted")
for i in range(1,(len(X_test))):
    sheet2.write(i,0,X_test[i][0])
    sheet2.write(i,1,X_test[i][1])
    sheet2.write(i,2,y_test[i])
    sheet2.write(i,3,x[i])
#book_result.save("Test_Results.csv")
    
###################################### Data visualisation #############################################

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf,
                    out_file='tree_dt.dot')

#tree.export_graphviz(clf_rf_train,
#                    out_file='tree_rf.dot')

i_tree = 0
for tree_in_forest in clf_rf_train.estimators_:
    with open('tree_' + str(i_tree) + '.dot', 'w') as my_file:
        my_file = tree.export_graphviz(tree_in_forest, out_file = my_file)
        #(graph_rf,) = pydot.graph_from_dot_file('tree_' + str(i_tree) + '.dot')
        #graph_rf.write_pdf('randomForests.pdf') 
    i_tree = i_tree + 1

#graph = pydot.graph_from_dot_data(dot_data.getvalue())
#graph.write_pdf("mode.pdf")
#from subprocess import check_call
#check_call(['dot','-Tpng','tree.dot','-o','OutputFile.png'])

(graph_dt,) = pydot.graph_from_dot_file('tree_dt.dot')
graph_dt.write_pdf('dectree.pdf') 

(graph_rf_1,) = pydot.graph_from_dot_file('tree_1.dot')
graph_rf_1.write_pdf('randomForests_1.pdf') 

(graph_rf_2,) = pydot.graph_from_dot_file('tree_2.dot')
graph_rf_2.write_pdf('randomForests_2.pdf')

(graph_rf_3,) = pydot.graph_from_dot_file('tree_3.dot')
graph_rf_3.write_pdf('randomForests_3.pdf')

####################################################################################################
################################## Classification of Unsegmented Data ###################################
####################################################################################################

print "########################## Classification of Unsegmented Data #####################################"

with open('C:\\Users\\N1600060B\\Documents\\Transport Mode Detection\\Final\\test.csv','r') as csvfile_ns:
    mode_reader_ns = csv.reader(csvfile_ns, delimiter=',', quotechar='"')
    
################################## Header contains feature names ######################################
    
    row_ns=next(mode_reader_ns)
    feature_names_ns=np.array(row_ns)
    #print feature_names_ns
    
#################################### Load dataset and target classes #####################################
    
    mode_X_ns,mode_y_ns=[],[]
    for row_ns in mode_reader_ns:
        mode_X_ns.append(row_ns)
        mode_y_ns.append(row_ns[8])
    mode_X_ns = np.array(mode_X_ns)
    mode_y_ns = np.array(mode_y_ns)
    #print mode_X_ns
    #print mode_y_ns
    
################################### Retain required columns ###########################################
    
    mode_X_ns = mode_X_ns[:, [6,7]]
    feature_names_ns = feature_names_ns[[6,7]]
    #print feature_names_ns
    #print mode_X
    
############################################## Encode mode ##########################################
    
    from sklearn.preprocessing import LabelEncoder
    enc_ns = LabelEncoder()
    label_encoder_ns = enc_ns.fit(mode_y_ns[:])
    print("Categorical Classes:", label_encoder_ns.classes_)
    integer_classes_ns = label_encoder_ns.transform(label_encoder_ns.classes_)
    print("Integer Classes:", integer_classes_ns)
    t_ns = label_encoder_ns.transform(mode_y_ns[:])
    mode_y_ns[:] = t_ns
    #print (mode_X[4], mode_y[4])
    
##################################### Separate training and test sets #####################################
    
    from sklearn.cross_validation import train_test_split
    X_train_ns, X_test_ns, y_train_ns, y_test_ns = train_test_split(mode_X_ns,
                                                                    mode_y_ns,
                                                                    test_size=0.1,
                                                                    random_state=30,
                                                                    stratify=None)
    #print X_test
    
########################################## Decision tree ##############################################
    
    from sklearn import tree
    clf_ns = tree.DecisionTreeClassifier(criterion='entropy',
                                      splitter='best',
                                      max_depth=30, 
                                      min_impurity_split=1e-09, 
                                      presort = True, 
                                      min_samples_leaf=3, 
                                      min_samples_split=20)
    clf_ns = clf_ns.fit(X_train_ns,y_train_ns)
    #print clf
    
########################################### Random Forests ###########################################

    from sklearn.ensemble import RandomForestClassifier
    clf_rf_train_ns = RandomForestClassifier(n_jobs=3,
                                         criterion='entropy',
                                         max_depth=30,
                                         min_samples_split=20,
                                         min_samples_leaf=5,
                                         verbose=0,
                                         warm_start=False)
    clf_rf_train_ns = clf_rf_train_ns.fit(X_train_ns,y_train_ns)
    
################################ Measure the accuracy of the decision tree ##################################
    
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
    
    print "--------------------Performance of the Decision Tree on the Unsegmented Training Data---------------------------------"
    measure_performance(X_train_ns,y_train_ns,clf_ns, show_accuracy=True, show_classification_report=True, show_confusion_matrix=True)
    
    print "--------------------Performance of the Random Forest on the Unsegmented Training Data----------------------------------"
    measure_performance(X_train_ns,y_train_ns,clf_rf_train_ns, show_accuracy=True, show_classification_report=True, show_confusion_matrix=True)
    
################## Evaluating the performance of the decision tree on the testing data ############################
    
    clf_dt_ns = tree.DecisionTreeClassifier(criterion='entropy', 
                                         splitter='best',  
                                         min_impurity_split=1e-09, 
                                         max_depth=30, 
                                         presort = True, 
                                         min_samples_leaf=3, 
                                         min_samples_split=20)
    clf_dt_ns.fit(X_train_ns, y_train_ns)
    x_ns = clf_dt_ns.predict(X_test_ns)
    #for i in range(1,(len(X_test))):
    #    print x[i]
    #    print y_test[i]
    
    print "------------------------Performance of the Decision Tree on the Unsegmented Testing Data-----------------------------------"
    measure_performance(X_test_ns,y_test_ns,clf_dt_ns)

################# Evaluating the performance of the random forest on the testing data ############################

    clf_rf_test_ns = RandomForestClassifier(n_jobs=3,
                                        criterion='entropy',
                                        max_depth=30,
                                        min_samples_split=20,
                                        min_samples_leaf=5,
                                        verbose=0,
                                        warm_start=False)
    clf_rf_test_ns = clf_rf_test_ns.fit(X_train_ns,y_train_ns)
    y_ns = clf_rf_test_ns.predict(X_test_ns)
    
    print "-------------------------Performance of the Random Forests on the Unsegmented Testing Data-----------------------------------"
    measure_performance(X_test_ns,y_test_ns,clf_rf_test_ns)
    
    
###################################### Exporting results to a CSV File ####################################

#print len(X_test)
#book_result_ns = xlwt.Workbook(encoding="utf-8")
#sheet3 = book.add_sheet("Sheet 3")
#sheet3.write(0,0,"Average Velocity")
#sheet3.write(0,1,"Average Acceleration")
#sheet3.write(0,2,"Mode - Ground Truth")
#sheet3.write(0,3,"Mode - Predicted")
#for i in range(1,(len(X_test_ns))):
#    sheet3.write(i,0,X_test[i][0])
#    sheet3.write(i,1,X_test[i][1])
#    sheet3.write(i,2,y_test[i])
#    sheet3.write(i,3,x[i])
#book_result.save("Test_Results.csv")
    
########################################## Data visualisation ##########################################

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf_ns,
                    out_file='tree_dt_ns.dot')

#tree.export_graphviz(clf_rf_train,
#                    out_file='tree_rf.dot')

#i_tree_ns = 0
#for tree_in_forest in clf_rf_train.estimators_:
#    with open('tree_ns_' + str(i_tree) + '.dot', 'w') as my_file_ns:
#        my_file_ns = tree.export_graphviz(tree_in_forest, out_file = my_file)
        #(graph_rf,) = pydot.graph_from_dot_file('tree_' + str(i_tree) + '.dot')
        #graph_rf.write_pdf('randomForests.pdf') 
#    i_tree_ns = i_tree_ns + 1

#graph = pydot.graph_from_dot_data(dot_data.getvalue())
#graph.write_pdf("mode.pdf")
#from subprocess import check_call
#check_call(['dot','-Tpng','tree.dot','-o','OutputFile.png'])

(graph_dt_ns,) = pydot.graph_from_dot_file('tree_dt_ns.dot')
graph_dt_ns.write_pdf('dectree_ns.pdf') 

#(graph_rf_ns_1,) = pydot.graph_from_dot_file('tree_ns_1.dot')
#graph_rf_ns_1.write_pdf('randomForests_ns_1.pdf') 

#(graph_rf_ns_2,) = pydot.graph_from_dot_file('tree_ns_2.dot')
#graph_rf_ns_2.write_pdf('randomForests_ns_2.pdf')

#(graph_rf_ns_3,) = pydot.graph_from_dot_file('tree_ns_3.dot')
#graph_rf_ns_3.write_pdf('randomForests_ns_3.pdf')
 
