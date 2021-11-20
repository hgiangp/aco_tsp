import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance as dis
import util

"""
参考URL
[1] 蟻コロニー最適化 - Wikipedia https://ja.wikipedia.org/wiki/蟻コロニー最適化
[2] 任意の確率密度分布に従う乱数の生成（von Neumannの棄却法） | Pacocat's Life http://pacocat.com/?p=596
"""

class ACO_TSP:
	def __init__(self,path=None,alpha = 1.0,beta = 1.0,Q = 1.0,vanish_ratio = 0.95):
		""" 初期化を行う関数 """
		self.alpha = alpha					# フェロモンの優先度
		self.beta = beta					# ヒューリスティック情報(距離)の優先度
		self.Q = Q							# フェロモン変化量の係数
		self.vanish_ratio = vanish_ratio	# 蒸発率
		if path is not None:
			self.set_loc(np.array(pd.read_csv(path)))
	
	def set_loc(self,locations):
		""" 位置座標を設定する関数 """
		self.loc = locations							# x,y座標
		self.n_data = len(self.loc)						# データ数
		self.dist = dis.squareform(dis.pdist(self.loc))	# 距離の表を作成
		self.weight = np.random.random_sample((self.n_data,self.n_data))	# フェロモンの量
		self.result = np.arange(self.n_data)			# もっともよかった順序を保存する
		
	def cost(self,order):
		""" 指定された順序のコスト計算関数 """
		n_order = len(order)
		return np.sum( [ self.dist[order[i],order[(i+1)%n_order]] for i in np.arange(n_order) ] )
	
	def plot(self,order=None):
		""" 指定された順序でプロットする関数 """
		if order is None:
			plt.plot(self.loc[:,0],self.loc[:,1])
		else:
			plt.plot(self.loc[order,0],self.loc[order,1])
		plt.savefig('result_aco.png')
	
	def solve(self,n_agent=1000):
		""" 巡回セールスマン問題を蟻コロニー最適化で解く """
		
		order = np.zeros(self.n_data,np.int) 		# 巡回経路
		delta = np.zeros((self.n_data,self.n_data))	#フェロモン変化量
		
		for k in range(n_agent):
			city = np.arange(self.n_data)
			now_city = np.random.randint(self.n_data)	# 現在居る都市番号
			
			city = city[ city != now_city ]
			order[0] = now_city
			
			for j in range(1,self.n_data):
				upper = np.power(self.weight[now_city,city],self.alpha)*np.power(self.dist[now_city,city],-self.beta)
				upper[upper==np.inf] = 0.1
				evaluation = upper / np.sum(upper)				# 評価関数
				percentage = evaluation / np.sum(evaluation)	# 移動確率
				index = self.random_index(percentage)			# 移動先の要素番号取得
				
				# 状態の更新
				now_city = city[ index ]
				city = city[ city != now_city ]
				order[j] = now_city
			
			L = self.cost(order) # 経路のコストを計算
			
			# フェロモンの変化量を計算
			delta[:,:] = 0.0
			c = self.Q / L
			for j in range(self.n_data-1):
				delta[order[j],order[j+1]] = c
				delta[order[j+1],order[j]] = c
			
			# フェロモン更新
			self.weight *= self.vanish_ratio 
			self.weight += delta
			
			# 今までで最も良ければ結果を更新
			if self.cost(self.result) > L:
				self.result = order.copy()
			
			# デバッグ用
			print("Agent ... %d , Cost ... %lf" % (k,self.cost(self.result)))
		
		return self.result

	
	def random_index(self,percentage):
		""" 任意の確率分布に従って乱数を生成する関数 """
		n_percentage = len(percentage)
		
		while True:
			index = np.random.randint(n_percentage)
			y = np.random.random()
			if y < percentage[index]:
				return index

# if __name__=="__main__":
#     tsp = ACO_TSP()
#     data = util.read_data('./tsp100.txt')
#     data = np.array(data)
#     tsp.set_loc(data)
#     tsp.solve(n_agent=1000)
#     tsp.plot(tsp.result)