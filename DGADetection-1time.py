# -*- coding: utf-8 -*-  
import numpy as np
import DistanceCal
from scipy import interp  
import matplotlib.pyplot as plt  
from sklearn.metrics import roc_curve, auc  
from sklearn.model_selection import StratifiedShuffleSplit  
import sys
import random

pos_sample_n = 500000
neg_sample_n = 500000
if len(sys.argv)<2:
	print 'Format: python DGADetection.py DGA_name'
	sys.exit()
DGA_name = sys.argv[1]

legi_domains = []
fi = open("RDNS_main_domains_filtered.txt")
for f in fi:
	legi_domains.append(f.strip())
fi.close()
legi_domains = random.sample(legi_domains,500000)#Randomly sample 500000 domains
y = [1 for i in xrange(pos_sample_n)]

mal_domains = []
fi = open("DGA_samples/"+DGA_name+".txt")
for f in fi:
	mal_domains.append(f.strip())
fi.close()
y.extend([-1 for i in xrange(neg_sample_n)])

domains = legi_domains
del legi_domains
domains.extend(mal_domains)

count = 0

sss  = StratifiedShuffleSplit(n_splits=1000,train_size=.001,test_size=.001)
for train_indices, test_indices in sss.split(domains, y):#Provides train/test indices to split data in train/test sets.
	print len(train_indices),len(test_indices)
	delta_kl_list = []
	
	y = [1 for i in xrange(pos_sample_n)]
	y.extend([-1 for i in xrange(neg_sample_n)])
	
	for te_index in test_indices:
		te_model = DistanceCal.StrDistanceModel(domains[te_index])
				
		sym_kl_list = []
		for tr_index in train_indices:
			tr_model = DistanceCal.StrDistanceModel(domains[tr_index])
			sym_kl_list.append(DistanceCal.sym_kl_between(te_model,tr_model)*y[tr_index])#∆KL = KL(d; L) − KL(d; M). y=-1 means train sample is malicious, y=1 means train sample is legitimate
		delta_kl = np.mean(sym_kl_list)
		delta_kl_list.append(delta_kl)
	y = [0 for i in xrange(pos_sample_n)]
	y.extend([1 for i in xrange(neg_sample_n)])
	fpr, tpr, thresholds = roc_curve([y[i] for i in test_indices], delta_kl_list)  
	roc_auc = auc(fpr, tpr) 
	plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))  
	plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')  
	plt.xlim([0, 1])  
	plt.ylim([0, 1])
	plt.xlabel('False Positive Rate')  
	plt.ylabel('True Positive Rate')  
	plt.title('Receiver operating characteristic example')  
	plt.legend(loc="lower right")  
	plt.show()  
	if count ==2:
		break
	count+=1
'''

###############################################################################  
# Data IO and generation,导入iris数据，做数据准备  
  
# import some data to play with  
iris = datasets.load_iris()  
X = iris.data  

y = [1 for i in xrange(500000)]

###############################################################################  
# Classification and ROC analysis  
#分类，做ROC分析  
  
# Run classifier with cross-validation and plot ROC curves  
#使用1000折交叉验证，并且画ROC曲线  
cv = StratifiedKFold(y, n_folds=1000)  
classifier = svm.SVC(kernel='linear', probability=True,  
                     random_state=random_state)#注意这里，probability=True,需要，不然预测的时候会出现异常。另外rbf核效果更好些。  
  
mean_tpr = 0.0  
mean_fpr = np.linspace(0, 1, 100)  
all_tpr = []  
  
for i, (train, test) in enumerate(cv):  
    #通过训练数据，使用svm线性核建立模型，并对测试集进行测试，求出预测得分  
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])  
	for i in i
	delta_kl = 
#    print set(y[train])                     #set([0,1]) 即label有两个类别  
#    print len(X[train]),len(X[test])        #训练集有84个，测试集有16个  
#    print "++",probas_                      #predict_proba()函数输出的是测试集在lael各类别上的置信度，  
#    #在哪个类别上的置信度高，则分为哪类  
    # Compute ROC curve and area the curve  
    #通过roc_curve()函数，求出fpr和tpr，以及阈值  
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])  
    mean_tpr += interp(mean_fpr, fpr, tpr)          #对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数  
    mean_tpr[0] = 0.0                               #初始处为0  
    roc_auc = auc(fpr, tpr)  
    #画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来  
    #plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))  
  
#画对角线  
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')  
  
mean_tpr /= len(cv)                     #在mean_fpr100个点，每个点处插值插值多次取平均  
mean_tpr[-1] = 1.0                      #坐标最后一个点为（1,1）  
mean_auc = auc(mean_fpr, mean_tpr)      #计算平均AUC值  
#画平均ROC曲线  
#print mean_fpr,len(mean_fpr)  
#print mean_tpr  
plt.plot(mean_fpr, mean_tpr, 'k--',  
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)  
  
plt.xlim([-0.05, 1.05])  
plt.ylim([-0.05, 1.05])  
plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate')  
plt.title('Receiver operating characteristic example')  
plt.legend(loc="lower right")  
plt.show()  
'''

