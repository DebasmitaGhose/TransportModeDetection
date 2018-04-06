import math
import numpy as np
import scipy as sc
import pandas as pd

df = pd.read_csv('sample.csv',sep=',',header=None)
data = df.values
# data = datat.tolist()
cValues = np.where((data[:,6])>1.6)[0]
print cValues
segment = data[3:cValues[0]+1,:]
segments = [segment]
start = 4
end = -1
for i in range(1,len(cValues)):
	if cValues[i]!=cValues[i-1]+1:
		end = cValues[i-1]+1
		seg1 = data[start:end,:]
		start2 = cValues[i-1]+1
		end2 = cValues[i]
		seg2 = data[start2:end2,:]
		if len(seg1)!=0:
			print len(seg1)
			segments.append(seg1)
		if len(seg2)!=0:
			segments.append(seg2)
			print len(seg2)
		start = cValues[i]
print segments
# print segments[2]
# print segments[3]
# segments[2] = np.concatenate((segments[2],segments[3]),axis=0)









del segments[3]
# segmentsNew = 
i=0
while len(segments<=3):

	i++
for i in range(0,len(segments)):
	if len(segments[i])<=3:
		something
	else:
		segmentsNew

# print segments[2][1][0]
def mergeSeg(segments, index1, index2):
	if index1<index2:
		segments[index1] = np.concatenate((segments[index1],segments[index2]),axis=0)
		del segments[index2]		
	else if index2<index1:
		segments[index1] = np.concatenate((segments[index2],segments[index1]),axis=0)
		del segments[index2]