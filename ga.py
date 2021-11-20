import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance as dis
import util

class GA_TSP:
	def __init__(self,path=None,n_gene = 256,n_parent = 10,change_ratio = 0.1):
		""" 初期化を行う関数 """
		self.n_gene = n_gene							# 一世代の遺伝子の個数
		self.n_parent = 10								# 親として残す個体数
		self.change_ratio = change_ratio				# 突然変異で変化させる場所の数
		if path is not None:
			self.set_loc(np.array(pd.read_csv(path)))
	
	def init_genes(self,):
		""" 遺伝子をランダムに初期化 """
		self.genes = np.zeros((self.n_gene,self.n_data),np.int)
		order = np.arange(self.n_data)
		for i in range(self.n_gene):
			np.random.shuffle(order)
			self.genes[i] = order.copy()
		self.sort()
	
	def set_loc(self,locations):
		""" 位置座標を設定する関数 """
		self.loc = locations							# x,y座標
		self.n_data = len(self.loc)						# データ数
		self.dist = dis.squareform(dis.pdist(self.loc))	# 距離の表を作成
		self.init_genes()								# 遺伝子を初期化
	
	def cost(self,order):
		""" 指定された順序のコスト計算関数 """
		return np.sum( [ self.dist[order[i],order[(i+1)%self.n_data]] for i in np.arange(self.n_data) ] )

	def plot(self,order=None):
		""" 指定された順序でプロットする関数 """
		if order is None:
			plt.plot(self.loc[:,0],self.loc[:,1])
		else:
			plt.plot(self.loc[order,0],self.loc[order,1])
		plt.savefig('result_ga.png')
	
	def solve(self,n_step=1000):
		""" 遺伝的アルゴリズムで解く関数 """
		for i in range(n_step):
			print("Generation ... %d, Cost ... %lf" % (i,self.cost(self.genes[0])))
			self.step()
		self.result = self.genes[0]
		
		return self.result
	
	def step(self):
		""" 遺伝子を一世代分進化させる関数 """
		# 突然変異
		for i in range(self.n_parent,self.n_gene):
			self.genes[i] = self.mutation( np.random.randint(self.n_parent) )
		self.sort() # ソートする

	def sort(self):
		""" 遺伝子を昇順にする関数 """
		# コストを計算し，ソート
		gene_cost = np.array([self.cost(i) for i in self.genes])
		self.genes = self.genes[ np.argsort(gene_cost) ]

	def mutation(self,index):
		""" 突然変異を起こした遺伝子を返す関数 """
		n_change = int(self.change_ratio * self.n_data)
		gene = self.genes[index].copy()
		
		for i in range(n_change):
			# n_changeの個数だけ値を入れ替える
			left = np.random.randint(self.n_data)
			right = np.random.randint(self.n_data)
			
			temp = gene[left]
			gene[left] = gene[right]
			gene[right] = temp
		
		return gene
	
# if __name__=="__main__":
#     tsp = GA_TSP()
#     data = util.read_data('./tsp100.txt')
#     data = np.array(data)
#     tsp.set_loc(data)
#     tsp.solve()
#     tsp.plot(tsp.result)