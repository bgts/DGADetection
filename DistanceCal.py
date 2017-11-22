import collections
import numpy as np
import re

def sym_kl_between(p, q):# p,q are another StrDistanceModels
	return 0.5*(p.kl_to(q)+q.kl_to(p))

class StrDistanceModel(object):
	def __init__(self, str):
		self._str = str
		self._ch_counts = dict(zip('abcdefghijklmnopqrstuvwxyz0123456789-', [0 for i in xrange(37)]))
		self._total_chs = 0
		for ch in self._str:
			self._ch_counts[ch] += 1
			self._total_chs += 1
			
	def log_likelihood(self, ch):
		if ch not in self._ch_counts:
			print 'haha'
			return -np.inf
		else:
			return np.log(self._ch_counts[ch]) - np.log(self._total_chs)

	def kl_to(self, p):
		# p is another StrDistanceModel			
		log_likelihood_ratios = []
		for ch in self._str:
			if p._ch_counts[ch]==0:
				log_likelihood_ratios.append(0)
			else:
				log_likelihood_ratios.append(float(self._ch_counts[ch])/self._total_chs*(self.log_likelihood(ch) - p.log_likelihood(ch)))
		return np.mean(log_likelihood_ratios)
	
	def ed_to(self, p):  #edit_distance
		# p is another StrDistanceModel	
		matrix = [[i+j for j in range(len(p._str) + 1)] for i in range(len(self._str) + 1)]
		for i in xrange(1,len(self._str)+1):  
			for j in xrange(1,len(p._str)+1):  
				if self._str[i-1] == p._str[j-1]:  
					d = 0  
				else:  
					d = 2  
				matrix[i][j] = min(matrix[i-1][j]+1,matrix[i][j-1]+1,matrix[i-1][j-1]+d)
		return float(matrix[len(self._str)][len(p._str)])
	
	def jaccard_to(self, p):
		# p is another StrDistanceModel	
		str1 = set(self._str)
		str2 = set(p._str)
		return float(len(str1 & str2)) / len(str1 | str2)

if __name__ == '__main__':
	p=StrDistanceModel('ofailing')
	q=StrDistanceModel('ofalng')
	print sym_kl_to(p,q),p.ed_to(q),p.jaccard_to(q)
