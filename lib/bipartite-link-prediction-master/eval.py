import util
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from operator import itemgetter


COLORS = ['r', 'b', 'g', 'm', 'y', 'c', 'k', '#FF9900', '#006600', '#663300']

def run_evaluation(examples, methods, precision_at=1):
    curve_args = []

    for i, method in enumerate(methods):
        predictions = util.load_json('./data/test/' + method + '.json')
        total_precision = 0.0
        total_recall = 0.0
        all_ys, all_ps, all_top_ys = [], [], []
        for u in predictions:
            ys, ps = zip(*[(examples[u][b], predictions[u][b]) for b in predictions[u]])
            all_ys += ys
            all_ps += ps

            n = min(precision_at, len(ys))
            top_ys = zip(*sorted(zip(ys, ps), key=itemgetter(1), reverse=True))[0][:n]
            total_precision += sum(top_ys) / float(n)

            n = min(precision_at, len(ps))
            total_recall += sum(top_ys) / len(top_ys)
            #print "     u {:}: top_ys {:}, ys_len {:}, ps_len {:}, n {:}".format(u, sum(top_ys), len(ys), len(ps), n)

        all_top_ys = zip(*sorted(zip(all_ys, all_ps), key=itemgetter(1), reverse=True))[0][:50]
        print sum(all_top_ys), len(all_top_ys), sum(all_ys)
        precision = sum(all_top_ys) / float(len(all_top_ys))
        recall = sum(all_top_ys) / float(sum(all_ys))
        F1 = 2*precision*recall / (precision+recall)
        print "precision:{:.6f}, recall:{:.6f}, F1:{:.6f}".format(precision,recall,F1)


        roc_auc = roc_auc_score(all_ys, all_ps)
        fpr, tpr, t = roc_curve(all_ys, all_ps)
        curve_args.append((fpr, tpr, method, COLORS[i % len(COLORS)]))

        print "Method:", method
        print "  total_precision {:}, examples_len {:}, predcition_len {:}".format(total_precision, len(examples), len(predictions))
        print "  Precision @{:} = {:.6f}".format(precision_at, total_precision / len(examples))
        print "  Recall @{:} = {:.6f}".format(precision_at, total_recall / len(examples))
        print "  F1 @{:} = {:.6f}".format(precision_at, 2*total_precision*total_recall / ((total_precision+total_recall) * len(examples)))
        print "  ROC Auc = {:.6f}".format(roc_auc)

    if i >= len(COLORS):
        print "Too many methods to plot all of them!"
        return

    '''plt.figure(figsize=(9, 9))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([0.0, 1.0])
    plt.title('ROC curves')
    for (fpr, tpr, label, color) in curve_args:
        plt.plot(fpr, tpr, label=label, color=color)
    plt.legend(loc="best")
    plt.show()'''


if __name__ == '__main__':
    run_evaluation(util.load_json('data/test/links.json'),['random_walks'])
'''                 ['examples',
                    'u_adamic',
                    'u_cn',
                    'u_jaccard',
                    'b_adamic',
                    'b_cn',
                    'b_jaccard',
                    'random_baseline',
                    'svd',
                    'random_walks',
                    'weighted_random_walks',
                    'supervised_random_walks',
                    'supervised_classifier'
                   ])'''



