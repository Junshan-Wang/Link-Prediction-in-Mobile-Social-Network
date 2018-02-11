#!/usr/bin python

import sys
from sklearn import svm
from sklearn import linear_model
from sklearn import tree
from sklearn import metrics
from operator import itemgetter

import random
import numpy as np
import scipy.io as sio
import networkx as nx
import predictLink as pl

def SVM(trainx, trainy, testx, testy):
	trainx_=np.array(trainx)
	trainx_=(trainx_-trainx_.mean())/trainx_.std()
	trainx=np.ndarray.tolist(trainx_)
	testx_=np.array(testx)
	testx_=(testx_-testx_.mean())/testx_.std()
	testx=np.ndarray.tolist(testx_)

	clf = svm.SVC(C=10., gamma=0.0001, probability=True, max_iter=10000)
	clf.fit(trainx, trainy) 
	predicty=clf.predict_proba(testx)
	return predicty

def LRC(trainx, trainy, testx, testy):
	clf=linear_model.LogisticRegression()
	clf=clf.fit(trainx, trainy)
	predicty=clf.predict_proba(testx)
	return predicty

def DT(trainx, trainy, testx, testy):
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(trainx, trainy)
	predicty=clf.predict_proba(testx)
	return predicty

def calculateAccuracy(testy, predicty, K):
	# testy and predicty is list, whose element is the single value
	
	predicty_=sorted(predicty,reverse=True)
	thredshold=predicty_[K]

	pt=zip(predicty,testy)
	pt=zip(*sorted(pt,key=itemgetter(0),reverse=True))[1]

	TP=float(sum(pt[:K]))
	FP=float(len(pt[:K])-sum(pt[:K]))
	FN=float(sum(pt[K+1:]))
	TN=float(len(pt[K+1:])-sum(pt[K+1:]))


	'''TP=FP=FN=TN=0.0
	for i in range(0,len(testy)):
		if testy[i]==1 and predicty[i]>thredshold:
			TP+=1
		elif testy[i]==1 and predicty[i]<=thredshold:
			FN+=1
		elif testy[i]==0 and predicty[i]>thredshold:
			FP+=1
		else:
			TN+=1'''

	print "potlinks num: %d" % len(testy)
	print "thredshold: %f" % thredshold
	print "implinks num: %f" % len([i for i in testy if i==1])
	print "TP: %d, FP: %d, FN: %d, TN: %d" % (TP,FP,FN,TN)

	if TP>0:
		precision=TP/(TP+FP)
		recall=TP/(TP+FN)
		F1=2*precision*recall/(precision+recall)
		print "Precision: %f" % precision
		print "Recall: %f" % recall
		print "F1: %f" % F1
	else:
		print "TP=0"

	AUC=metrics.roc_auc_score(testy, predicty)	
	print "AUC: %f" % AUC


def entrance(linktype, func):
	graph=pl.generateGraph(linktype)
	snodes=pl.sourceNode(graph,200)
	snode_n=len(snodes)

	index=range(0,snode_n)
	random.shuffle(index)
	train_snodes=[snodes[i] for i in index[0:snode_n/2]]
	test_snodes=[snodes[i] for i in index[snode_n/2:snode_n]]

	trainx=[]
	for i in range(0,4):
		f=pl.linkProbability(graph,train_snodes,i)
		trainx.append([j[2] for j in f])
	trainx=map(list, zip(*trainx))
	f=pl.linkGroundtrue(linktype,graph,train_snodes)
	trainy=[j[2] for j in f]

	testx=[]
	for i in range(0,4):
		f=pl.linkProbability(graph,test_snodes,i)
		testx.append([j[2] for j in f])
	testx=map(list, zip(*testx))
	f=pl.linkGroundtrue(linktype,graph,test_snodes)
	testy=[j[2] for j in f]

	if func==0:
		predicty=LRC(trainx,trainy,testx,testy)
		calculateAccuracy(testy,predicty[:,1],300000)
	elif func==1:
		predicty=DT(trainx,trainy,testx,testy)
		calculateAccuracy(testy,predicty[:,1],300000)
	else:
		predicty=SVM(trainx,trainy,testx,testy)
		calculateAccuracy(testy,predicty[:,1],300000)

def entrance2(linktype, func):
	graph=pl.generateGraph(linktype)

	trainx=[]
	for i in range(0,4):
		f=pl.linkProbability2(graph,i)
		trainx.append([j[2] for j in f])
	trainx=map(list, zip(*trainx))
	f=pl.linkGroundtrue2(linktype,graph)
	trainy=[j[2] for j in f]

	if func==0:
		predicty=LRC(trainx,trainy,trainx,trainy)
		calculateAccuracy(trainy,predicty[:,1],100000)
	elif func==1:
		predicty=DT(trainx,trainy,trainx,trainy)
		calculateAccuracy(trainy,predicty[:,1],29)
	else:
		predicty=SVM(trainx,trainy,trainx,trainy)
		calculateAccuracy(trainy,predicty[:,1],29)


nodeIndex={}
def generateNodeIndex(graph, source):
	global nodeIndex
	nodeIndex[source]=0
	i=1
	for node in graph.nodes():
		if not nodeIndex.has_key(node):
			nodeIndex[node]=i
			i+=1

def sourceNodeDestination(graph, trueMatrix, source):
	# given a source node, calculate its binary destination vector
	d=[0]*graph.number_of_nodes()
	# the bit of implicit node is set to 1
	for node1,node2,i in trueMatrix:
		if node1==source:
			d[nodeIndex[node2]]=i
		if node2==source:
			d[nodeIndex[node1]]=i
	return d

def oneFeatureVector(graph, f):
	# return one feature vector 
	tmp=[[0]*graph.number_of_nodes()]*graph.number_of_nodes()

	if f==0:
		prob=list(nx.resource_allocation_index(graph))
	elif f==1:
		prob=list(nx.jaccard_coefficient(graph))
	elif f==2:
		prob=list(nx.adamic_adar_index(graph))
	else:
		prob=list(nx.preferential_attachment(graph))

	for node1,node2,i in prob:
		tmp[nodeIndex[node1]][nodeIndex[node2]]=i
		tmp[nodeIndex[node2]][nodeIndex[node1]]=i

	return tmp


def generateFeatureMatrix(graph):
	# feature matrix of the graph, including 5 features:
	# whether two nodes are explicitly connected,
	# CN, AA, JA, RA
	matrix=[]

	tmp=[[1]*graph.number_of_nodes()]*graph.number_of_nodes()
	for node1,node2 in nx.non_edges(graph):
		tmp[nodeIndex[node1]][nodeIndex[node2]]=0
		tmp[nodeIndex[node2]][nodeIndex[node1]]=0
	matrix.append(tmp)

	tmp=oneFeatureVector(graph,0)
	matrix.append(tmp)

	tmp=oneFeatureVector(graph,1)
	matrix.append(tmp)

	tmp=oneFeatureVector(graph,2)
	matrix.append(tmp)

	tmp=oneFeatureVector(graph,3)
	matrix.append(tmp)

	return matrix


def loadSRWData():
	# graph
	graph=pl.generateGraph()
	print graph.number_of_nodes()
	print graph.number_of_edges()
	print len(list(nx.non_edges(graph)))
	
	snode=pl.sourceNode(graph,1)
	generateNodeIndex(graph,snode[0])
	# feature matrix
	psi=generateFeatureMatrix(graph)
	# destination vector for s
	true=pl.linkGroundtrue("colleague",graph)	
	d=sourceNodeDestination(graph,true,snode[0])
	# save the data
	sio.savemat('G.mat',{'psi':psi, 'd':d})
	
	'''outputFile=open('destination.txt','w')
	outputFile.write(snode[0]+'\n')
	outputFile.write(str(nx.degree(graph,snode[0]))+'\n')
	for each in d:
		outputFile.write(str(each))
		outputFile.write('\n')
	outputFile.close()'''


if __name__=="__main__":
	#loadSRWData()
	linktype="family"
	#entrance(linktype,1)
	#entrance(linktype,1)
	entrance2(linktype,1)



