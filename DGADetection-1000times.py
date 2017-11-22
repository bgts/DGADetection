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

sss  = StratifiedShuffleSplit(n_splits=1000,train_size=.001,test_size=.001).split(domains, y)
mean_tpr = 0.0  
mean_fpr = np.linspace(0, 1, 100)  

for train_indices, test_indices in sss:#Provides train/test indices to split data in train/test sets.
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
	mean_tpr += interp(mean_fpr, fpr, tpr)          #对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数  
	mean_tpr[0] = 0.0                               #初始处为0

fig = plt.gcf()
#画对角线  
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')  
  
mean_tpr /= 1000                     	#在mean_fpr100个点，每个点处插值插值多次取平均  
mean_tpr[-1] = 1.0                      #坐标最后一个点为（1,1）  
mean_auc = auc(mean_fpr, mean_tpr)      #计算平均AUC值  
#画平均ROC曲线  
plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2) 
plt.xlim([0, 1])  
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate')  
plt.title('KL ROC of 500 data, 1000 times, 100 thresholds')  
plt.legend(loc="lower right")  
plt.show()  
fig.savefig("ROC/"+DGA_name+"-KL.png",dpi=100)
