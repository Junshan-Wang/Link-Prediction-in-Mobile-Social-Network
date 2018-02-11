import util
from operator import itemgetter

def run_evaluation():
	predictions=util.load_json('./data/test/random_walks.json')
	groundtruth=util.load_json('./data/test/links.json')
	TP,FP,FN=0.0,0.0,0.0
	for i in predictions:
		for j in predictions[i]:
			if predictions[i][j]>0 and groundtruth[i][j]==1:
				TP+=1
			elif predictions[i][j]>0 and groundtruth[i][j]==0:
				FP+=1
			elif predictions[i][j]==0 and groundtruth[i][j]==1:
				FN+=1
	precision=TP/(TP+FP)
	recall=TP/(177849)
	F1=2*precision*recall / (precision+recall)

	print "TP:{:.6f}, FP:{:.6f}, FN:{:.6f}".format(TP,FP,FN)
	print "precision:{:.6f}, recall:{:.6f}, F1:{:.6f}".format(precision,recall,F1)


if __name__=="__main__":
	run_evaluation()