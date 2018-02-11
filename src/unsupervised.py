#!/usr/bin python
import sys
import os
import util
import networkx as nx
from sklearn import metrics
from sklearn import preprocessing
import math
import json
import random
import numpy as np

def sigmoid(x):
	return 1.0/(1+math.exp(-x))


def generateRelations(linktype, transform={}):
	openFile=open("../data/"+linktype+"_info.txt")
	openFile.readline()
	lines=openFile.readlines()
	# each item in groups is a family/colleague
	groups={}
	for each in lines:
		items=each.strip().split('\t')
		if not groups.has_key(items[0]):
			groups[items[0]]=[]
		groups[items[0]].append(items[1])
	# all relations, including explicit and implicit
	relations={}
	for each in groups:
		for node1 in groups[each]:
			for node2 in groups[each]:
				if node1<node2:
					relations[(node1,node2)]=1

	openFile.close()

	if len(transform)>0:
		newrelations={}
		for relation in relations:
			if not transform.has_key(relation[0]):
				continue
			if not transform.has_key(relation[1]):
				continue
			newrelations[(transform[relation[0]],transform[relation[1]])]=1
		relations=newrelations

	return relations


def generateGraph(linktype):
	# Graph or Multigraph
	graph=nx.Graph()

	# Add node
	userInfoFile=open("../data/user_info.txt")
	userInfoFile.readline()
	lines=userInfoFile.readlines()
	for each in lines:
		items=each.strip().split('\t')
		graph.add_node(items[0])
	userInfoFile.close()

	relations=generateRelations(linktype)

	# Add explicit edge(call)
	callInfoFile=open("../data/call_info.txt")
	callInfoFile.readline()
	lines=callInfoFile.readlines()
	for each in lines:
		items=each.strip().split('\t')
		if (items[0],items[1]) in relations or (items[1],items[0]) in relations:
			graph.add_edge(items[0],items[1])
	callInfoFile.close()

	# Add explicit edge(message)
	msgInfoFile=open("../data/msg_info.txt")
	msgInfoFile.readline()
	lines=msgInfoFile.readlines()
	for each in lines:
		items=each.strip().split('\t')
		if (items[0],items[1]) in relations or (items[1],items[0]) in relations:
			graph.add_edge(items[0],items[1])
	msgInfoFile.close()

	'''newGraph=nx.Graph()
	for node in graph.nodes():
			if graph.degree(node)>0:
				newGraph.add_node(node)
	for edge in graph.edges():
		newGraph.add_edge(edge[0],edge[1])
	return newGraph'''
	return graph


def transformGraph(graph):
	# trasnform the graph, to a new graph with continues index
	# all nodes' degree > 0
	i=0
	transform={}
	newGraph=nx.Graph()
	for node in graph.nodes():
		if graph.degree(node)>0:
			newGraph.add_node(str(i))
			transform[node]=str(i)
			i+=1
	for edge in graph.edges():
		newGraph.add_edge(transform[edge[0]],transform[edge[1]])
	return (newGraph,transform)


def sourceNode(graph,num):
	# return num source nodes with highest degree
	D=nx.degree(graph)
	D=sorted(D.items(),lambda x,y:cmp(x[1],y[1]),reverse=True)
	if num==-1:
		D=[i[0] for i in D if i[1]>0]
		return D
	else:
		D=D[:num]
		D=[i[0] for i in D]
		return D


def linkGroundtrue(linktype, graph, snodes, transform={}):
	relations=generateRelations(linktype, transform)
	true=[]
	for node1,node2 in nx.non_edges(graph):
		if node2<node1:
			node1,node2=node2,node1
		if (node1,node2) in relations or (node2,node1) in relations:
			true.append((node1,node2,1))
		else:
			true.append((node1,node2,0))

	if len(snodes)<1:
		true_=[]
		for node1,node2,p in true:
			if node2<node1:
				node1,node2=node2,node1
			true_.append((node1,node2,p))
		true=list(set(true_))
		return true
	else:
		true_=[]
		for node1,node2,p in true:
			if node2<node1:
				node1,node2=node2,node1
			if node1 in snodes:
				true_.append((node1,node2,p))
			if node2 in snodes:
				true_.append((node1,node2,p))
		true=list(set(true_))
		return true


def linkProbability(graph, snodes, func, sig=0):
	# the probability of all potential links
	if func==0:
		prob=list(nx.resource_allocation_index(graph))
	elif func==1:
		prob=list(nx.jaccard_coefficient(graph))
	elif func==2:
		prob=list(nx.adamic_adar_index(graph))
	else:
		prob=list(nx.preferential_attachment(graph))

	if sig==1:
		prob=[(i,j,sigmoid(k)) for (i,j,k) in prob]

	if len(snodes)<1:
		prob_=[]
		for node1,node2,p in prob:
			if node2<node1:
				node1,node2=node2,node1
			prob_.append((node1,node2,p))
		prob=list(set(prob_))
		return prob
	else:
		prob_=[]
		for node1,node2,p in prob:
			if node2<node1:
				node1,node2=node2,node1
			if node1 in snodes:
				prob_.append((node1,node2,p))
			if node2 in snodes:
				prob_.append((node1,node2,p))
		prob=list(set(prob_))
		return prob

def calculateAccuracy(true, prob, K=1000):
	# true and prob is list, whose element is tuple (node1, node2, value)
	print "potlinks num: %d" % len(true)
	# get actual implicit links and true tags
	implinks=[(i[0],i[1]) for i in true if i[2]==1]
	m=np.median([i[2] for i in prob])
	#prob_=sorted([i[2] for i in prob],reverse=True)
	#m=prob_[15]
	print "median of probability: "+str(m)
	prelinks=[(i[0],i[1]) for i in prob if i[2]>m]

	# calculate TP,FP,FN
	TP=float(len(set(prelinks)&set(implinks)))
	FP=float(len(set(prelinks)-set(implinks)))
	FN=float(len(set(implinks)-set(prelinks)))
	print "prelinks num: %d, implinks num: %d" % (len(prelinks),len(implinks))
	print "TP: %d, FP: %d, FN: %d" % (TP,FP,FN)

	if TP>0:
		precision=TP/(TP+FP)
		recall=TP/(TP+FN)
		F1=2*precision*recall/(precision+recall)
		print "Precision: %f" % precision
		print "Recall: %f" % recall
		print "F1: %f" % F1
	else:
		print "TP=0"


def printEdgeFile(graph):
	outFile=open('../rgraph-master/reliability/network','w')
	#outFile=open('../bipartite-link-prediction-master/data/test/network','w')
	for edge in graph.edges():
		outFile.write(edge[0]+'  '+edge[1]+'\n')
	outFile.close()


def probBySBM(graph):
	prob,prob_=[],{}
	openFile=open('../rgraph-master/reliability/missing.dat')
	lines=openFile.readlines()
	for each in lines:
		each=each.strip().split(' ')
		prob_[(each[1],each[2])]=float(each[0])
	openFile.close()
	for each in nx.non_edges(graph):
		if prob_.has_key((each[0],each[1])):
			prob.append((each[0],each[1],prob_[(each[0],each[1])]))
		else: 
			pass
			#prob.append((each[0],each[1],0))
	#prob=sorted(prob)
	return prob


def printNeighborFile(graph, true):
	matrix={}

	for node1,node2,i in true:
		if node1 not in matrix:
			matrix[node1]={}
		matrix[node1][node2]=i

	with open('../bipartite-link-prediction-master/data/test/links.json', 'w') as f:
		f.write(json.dumps(matrix))


def calculateRWAccuracy(true, transform):
	predictions = util.load_json('../bipartite-link-prediction-master/data/test/random_walks.json')
	prob=[]
	for node1,node2,i in true:
		if not transform.has_key(node1) or not transform.has_key(node2):
			#prob.append((node1,node2,0))
			continue
		node1_=transform[node1]
		node2_=transform[node2]
		if predictions.has_key(node1_) and predictions[node1_].has_key(node2_):
			prob.append((node1,node2,predictions[node1_][node2_]))
		elif predictions.has_key(node2_) and predictions[node2_].has_key(node1_):
			prob.append((node1,node2,predictions[node2_][node1_]))
		else:
			pass
			#prob.append((node1,node2,0)
	calculateAccuracy(true,prob)


if __name__=="__main__":
	linktype="family"

	graph=generateGraph(linktype)
	printEdgeFile(graph)
	#(newgraph,transform)=transformGraph(graph)
	#true=linkGroundtrue(linktype,newgraph,[],transform)
	#printEdgeFile(newgraph)
	#printNeighborFile(newgraph,true)

	#snodes=sourceNode(graph,100)
	#true=linkGroundtrue(linktype,graph,snodes)
	#prob=linkProbability(graph,snodes,3)
	prob=probBySBM(graph)
	true=linkGroundtrue(linktype,graph,[])
	#calculateRWAccuracy(true,transform)
	calculateAccuracy(true,prob)
	print "Finished!"

