import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx
import itertools as itt
from typing import Union
from PIL import Image

#====================================
class Distances():
	
	def __init__(self):
		return NotImplemented
	
	@classmethod
	def shortest_dist(cls, G) -> tuple:
		if G._isnx():
			min_dist = np.full([G.count_nodes()]*2, np.inf)
			for vi, distances in nx.all_pairs_shortest_path_length(G.data):
				for vj in distances:
					min_dist[vi, vj] = distances[vj]
		elif G._isig():
			min_dist = G.data.distances()
		return tuple(map(tuple, min_dist))
	
	@classmethod
	def shortest_dist_distrib(cls, G, min_dist:tuple=None) -> list:
		if min_dist is None:	min_dist = cls.shortest_dist(G)
		counts = {}
		for line in min_dist:
			for value in line:
				if value not in counts:
				    counts[value] = 0
				counts[value] += 1
		N = G.count_nodes()
		finite_dists = [ *filter(lambda x: x < np.inf, counts.keys()) ]
		p_min_dist = np.zeros([ int(np.max(finite_dists)) + 2 ])
		for d in counts:
			idx = (d if d < np.inf else -1)
			p_min_dist[idx] = counts[d] / N**2
		return list(p_min_dist)
	
	@classmethod
	def diameter(cls, G, p_min_dist:list=None) -> float:
		if p_min_dist is None:	p_min_dist = cls.shortest_dist_distrib(G)
		return len(p_min_dist)-2 if len(p_min_dist) >= 2 else np.inf
	
	@classmethod
	def avg_geodesic_dist(cls, G, p_min_dist:list=None) -> float:
		if p_min_dist is None:	p_min_dist = cls.shortest_dist_distrib(G)
		return np.sum([ p*d for (d,p) in enumerate(p_min_dist[:-1]) ]) / np.sum(p_min_dist[:-1])
	
	@classmethod
	def global_efficiency(cls, G, p_min_dist:list=None) -> float:
		if p_min_dist is None:	p_min_dist = cls.shortest_dist_distrib(G)
		return np.sum([ p/d for (d,p) in enumerate(p_min_dist) if d > 0 ])
	
	@classmethod
	def hmean_geodesic_dist(cls, G, effic:float=None) -> float:
		if effic is None:	effic = cls.global_efficiency(G)
		return (1.0 / effic) if effic > 0 else np.inf
	
	@classmethod
	def as_dataframe(cls):
		return pd.DataFrame({
			'name':['shortest distances', 'shortest distances distribution', 'diameter','average geodesic distance', 'global efficiency', 'harmonic mean of the geodesic distances'],
			'varname':['min_dist', 'p_min_dist', 'diam', 'avg_gdist', 'effic', 'hmean_gdist'],
			'symbol': ['d_ij', 'P(d_ij)', 'D', 'l', 'E', 'h'],
			'scope': ['vertex pair', 'distrib', 'graph', 'graph', 'graph', 'graph'],
			'function': [cls.shortest_dist, cls.shortest_dist_distrib, cls.diameter, cls.avg_geodesic_dist, cls.global_efficiency, cls.hmean_geodesic_dist],
			'dependence': [None, ['shortest distances'], ['shortest distances distribution'], ['shortest distances distribution'], ['shortest distances distribution'], ['global efficiency']],
			'default': ([None] * 6),
			'category': 'distances'
		})

#====================================
class Connectivity():
	
	def __init__(self):
		return NotImplemented
	
	@classmethod
	def lth_stat_moment(cls, P:list, l:int, center:float=0, norm:float=1) -> float:
		norm = 1 if not norm else norm
		samples = (np.arange(len(P)) - center) / norm
		return float(np.sum([ (x**l)*p for p, x in zip(P, samples) ]))
	
	@classmethod
	def vertex_degree(cls, G) -> list:
		return [ *dict(G.degree()).values() ]
	
	@classmethod
	def max_degree(cls, G, deg:list=None) -> int:
		if deg is None:			deg = cls.vertex_degree(G)
		return int(np.max(deg))
	
	@classmethod
	def deg_distrib(cls, G, deg:list=None) -> list:
		if deg is None:			deg = cls.vertex_degree(G)
		deg = np.array(deg)
		if len(deg) > 0:
			deg_list, counts = np.unique(deg, return_counts=True)
			df1 = pd.DataFrame({ 'degree': range(0, deg.max()+1) })
			df2 = pd.DataFrame({ 'degree': deg_list, 'count': counts })
			df3 = df1.merge(df2, on='degree', how='left').fillna(0).astype(int)
			return list(df3['count'].values / len(deg))
		else:
			return list([1.0])
	
	@classmethod
	def rem_deg_distrib(cls, G, p_deg:list=None) -> list:
		if p_deg is None:		p_deg = cls.deg_distrib(G)
		deg_list = np.arange(0, len(p_deg))
		q_deg = np.array([ (k+1)*p_deg[k+1] for k in deg_list[:-1] ] + [0])
		avg_deg = (deg_list * p_deg).sum()
		if avg_deg > 0:
			return list(q_deg / avg_deg)
		else:
			return list(np.array([ 1.0*(i == 0) for i in range(len(p_deg)) ]))
	
	@classmethod
	def avg_degree(cls, G, p_deg:list=None) -> float:
		if p_deg is None:		p_deg = cls.deg_distrib(G)
		return cls.lth_stat_moment(p_deg, 1)
	
	@classmethod
	def var_degree(cls, G, p_deg:list=None, avg_deg:float=None) -> float:
		if p_deg is None:		p_deg = cls.deg_distrib(G)
		if avg_deg is None:		avg_deg = cls.avg_degree(G, p_deg)
		return cls.lth_stat_moment(p_deg, 2, avg_deg)
	
	@classmethod
	def skw_degree(cls, G, p_deg:list=None, avg_deg:float=None, var_deg:float=None) -> float:
		if p_deg is None:		p_deg = cls.deg_distrib(G)
		if avg_deg is None:		avg_deg = cls.avg_degree(G, p_deg)
		if var_deg is None:		var_deg = cls.var_degree(G, p_deg, avg_deg)
		return cls.lth_stat_moment(p_deg, 3, avg_deg, np.std(var_deg))
	
	@classmethod
	def krt_degree(cls, G, p_deg:list=None, avg_deg:float=None, var_deg:float=None) -> float:
		if p_deg is None:		p_deg = cls.deg_distrib(G)
		if avg_deg is None:		avg_deg = cls.avg_degree(G, p_deg)
		if var_deg is None:		var_deg = cls.var_degree(G, p_deg, avg_deg)
		return cls.lth_stat_moment(p_deg, 4, avg_deg, np.std(var_deg))
	
	# def avg_nearest_neighbour_lth(G, l, p_deg=None):
	#	 if p_deg is None:
	#		 p_deg = deg_distrib(G)
	#	 l = int(l)
	#	 return np.sum([ (k**l)*p for k, p in enumerate(p_deg) ])
	
	# def avg_nearest_neighbour_l1(G, p_deg=None):
	#	 return avg_nearest_neighbour_lth(G, 1, p_deg)
	
	# def avg_nearest_neighbour_l2(G, p_deg=None):
	#	 return avg_nearest_neighbour_lth(G, 2, p_deg)
	
	# def avg_nearest_neighbour_l3(G, p_deg=None):
	#	 return avg_nearest_neighbour_lth(G, 3, p_deg)
	
	@classmethod
	def as_dataframe(cls):
		return pd.DataFrame({
			'name':['vertex degree', 'maximum degree', 'degree distribution','remaining degree distribution', 
					'average degree', 'degree\'s variance', 'degree\'s skewness', 'degree\'s kurtosis'],
			'varname':['deg', 'max_deg', 'p_deg', 'q_deg', 'avg_deg', 'var_deg', 'skw_deg', 'krt_deg'],
			'symbol': ['k_i', 'k_max', 'P(k)', 'q(k)', 'Avg[k]', 'Var[k]', 'Skew[k]', 'Kurt[k]'],
			'scope': ['vertex', 'graph', 'distrib', 'distrib', 'graph', 'graph', 'graph', 'graph'],
			'function': [cls.vertex_degree, cls.max_degree, cls.deg_distrib, cls.rem_deg_distrib, 
						 cls.avg_degree, cls.var_degree, cls.skw_degree, cls.krt_degree],
			'dependence': [None, ['vertex degree'], ['vertex degree'], ['degree distribution'], 
						   ['degree distribution'], ['degree distribution', 'average degree'], 
						   ['degree distribution', 'average degree', 'degree\'s variance'], 
						   ['degree distribution', 'average degree', 'degree\'s variance']],
			'default': None,
			'category': 'connectivity'
		})

#====================================
class ClusteringAndCycles():
	
	def __init__(self):
		return NotImplemented
	
	@classmethod
	def qt_triplets(cls, G, deg:list=None) -> list:
		if deg is None:			deg = Connectivity.vertex_degree(G)
		return [ k*(k-1)/2 for k in deg ]
	
	@classmethod
	def vertex_clustering_coeff(cls, G, qt_tri:list=None) -> list:
		if qt_tri is None:		qt_tri = cls.qt_triplets(G)
		V = G.nodes()
		cc = []
		
		for v in V:
			# lista de vértices U conectados à V
			U = G.neighbors(v)
			
			# máximo de arestas possíveis entre os vértices de U
			M = qt_tri[v] #len(U)*(len(U) - 1)
			
			# contagem de arestas existentes entre os vértices de U
			B = 0
			for u in U:
				W = set(U) - set([u])
				for w in W:
					if G._isnx():
						B += G.data.has_edge(u, w)
					elif G._isig():
						B += (G.data.get_eid(u, w, error=False) != -1)
			
			# coeficiente de aglomeração
			cc.append(B/M if M != 0 else 0)
		
		return cc
	
	@classmethod
	def network_clustering_coeff(cls, G, cc:list=None) -> float:
		if cc is None:			cc = cls.vertex_clustering_coeff(G)
		return float(np.mean(cc))
		
	@classmethod
	def as_dataframe(cls):
		return pd.DataFrame({
			'name':['qt. of triplets', 'vertex clustering coefficient','network clustering coefficient'],
			'varname':['qt_tri', 'cc', 'avg_cc'],
			'symbol': ['N3(i)', 'C_i', 'C'],
			'scope': ['vertex', 'vertex', 'graph'],
			'function': [cls.qt_triplets, cls.vertex_clustering_coeff, cls.network_clustering_coeff],
			'dependence': [['vertex degree'], ['qt. of triplets'], ['vertex clustering coefficient']],
			'default': None,
			'category': 'clustering and cycles'
		})

#====================================
class EntropyAndEnergy():
	
	def __init__(self):
		return NotImplemented
	
	@classmethod
	def energy(cls, G, p_deg:list=None) -> float:
		if p_deg is None:		p_deg = Connectivity.deg_distrib(G)
		# Ng = np.prod([ np.math.factorial(k)**p for k, p in enumerate(p_deg) ])
		Ng = np.prod([ np.prod( np.array(range(2, k+1))**p ) for k, p in enumerate(p_deg) ])
		return float(np.log(Ng))
	
	@classmethod
	def entropy(cls, G, p_deg:list=None) -> float:
		if p_deg is None:		p_deg = Connectivity.deg_distrib(G)
		return -float(np.sum([ p * np.log(p) for p in p_deg if p > 0]))
	
	@classmethod
	def rem_entropy(cls, G, q_deg:list=None) -> float:
		if q_deg is None:		q_deg = Connectivity.rem_deg_distrib(G)
		return -float(np.sum([ p * np.log(p) for p in q_deg if p > 0]))
			
	@classmethod
	def as_dataframe(cls):
		return pd.DataFrame({
			'name':['energy', 'entropy', 'entropy of remaining degree'],
			'varname':['p_deg_enrg', 'p_deg_entr', 'q_deg_entr'],
			'symbol': ['E(P(k))', 'H', 'H*'],
			'scope': ['graph', 'graph', 'graph'],
			'function': [cls.energy, cls.entropy, cls.rem_entropy],
			'dependence': [['degree distribution'], ['degree distribution'], ['remaining degree distribution']],
			'default': None,
			'category': 'entropy and energy'
		})

#====================================
class Centrality():
	
	def __init__(self):
		return NotImplemented
	
	@classmethod
	def betweenness_centrality(cls, G) -> list:
		if G._isnx():
			G = G.to('igraph')
		return G.data.betweenness()
	
	@classmethod
	def avg_betweenness_centrality(cls, G, bc:list=None) -> float:
		if bc is None:			bc = cls.betweenness_centrality(G)
		return float(np.mean(bc))
	
	@classmethod
	def central_point_dominance(cls, G, bc:list=None) -> float:
		if bc is None:			bc = cls.betweenness_centrality(G)
		bc_max = np.max(bc)
		return float(np.sum([ bc_max - b for b in bc ]) / (len(bc) - 1))
	
	@classmethod
	def as_dataframe(cls):
		return pd.DataFrame({
			'name':['betweenness centrality', 'average betweenness centrality','central point domimnance'],
			'varname':['bc', 'avg_bc', 'cpd'],
			'symbol': ['B_i', '<B>', 'CPD'],
			'scope': ['vertex', 'graph', 'graph'],
			'function': [cls.betweenness_centrality, cls.avg_betweenness_centrality, cls.central_point_dominance],
			'dependence': [None, ['betweenness centrality'], ['betweenness centrality']],
			'default': None,
			'category': 'centrality'
		})

#====================================
#class SmallWorldness():
#	
#	def __init__(self):
#		return NotImplemented
#	
#	@classmethod
#	def omega(cls, G, avg_deg:float=None, avg_gdist:float=None, avg_cc:float=None, precision:int=8) -> float:
#		if avg_deg is None:		avg_deg = Connectivity.avg_degree(G)
#		if avg_gdist is None:	avg_gdist = Distances.avg_geodesic_dist(G)
#		if avg_cc is None:		avg_cc = ClusteringAndCycles.network_clustering_coeff(G)
#		N = G.count_nodes()
#		avg_gdist = round(avg_gdist, precision)
#		avg_gdist_rand = 1/2 + (np.log(N) - np.euler_gamma) / np.log(avg_deg)
#		avg_cc_lattice = round((3.0/4.0) * (avg_deg-2)/(avg_deg-1), precision) if avg_deg > 1 else np.inf
#		return (avg_gdist_rand/avg_gdist - avg_cc/avg_cc_lattice) if (avg_gdist and avg_cc_lattice) else np.nan
#	
#	@classmethod
#	def as_dataframe(cls):
#		return pd.DataFrame({
#			'name':['small-world coefficient'],
#			'varname':['omega'],
#			'symbol': ['omega'],
#			'scope': ['graph'],
#			'function': [cls.omega],
#			'dependence': [['average degree','average geodesic distance','network clustering coefficient']],
#			'default': [{'precision': 8}], 
#			'category': 'small-worldness'
#		})

#====================================
class SpectralAnalysis():
	
	def __init__(self):
		return NotImplemented
	
	@classmethod
	def spectrum(cls, G, precision:int=12, max_nodes:int=100) -> list:
		if G.count_nodes() <= max_nodes:
			A = G.to_adj_matrix()
			eigvals = np.linalg.eigvals(A)
			return [ round(num.real, precision) + round(num.imag, precision)*1j for num in eigvals ]
		else:
			return [np.nan]
	
	@classmethod
	def eigval_moment_lth(cls, G, l:int, spectrum:list=None) -> float:
		if spectrum is None:	spectrum = cls.spectrum(G)
		m = np.mean(np.array(spectrum)**int(l))
		return np.real(m) if np.imag(m) == 0 else m
	
	@classmethod
	def moment_l1(cls, G, spectrum:list=None) -> float:
		return cls.eigval_moment_lth(G, 1, spectrum)
	
	@classmethod
	def moment_l2(cls, G, spectrum:list=None) -> float:
		return cls.eigval_moment_lth(G, 2, spectrum)
	
	@classmethod
	def moment_l3(cls, G, spectrum:list=None) -> float:
		return cls.eigval_moment_lth(G, 3, spectrum)
	
	@classmethod
	def moment_l4(cls, G, spectrum:list=None) -> float:
		return cls.eigval_moment_lth(G, 4, spectrum)
	
	@classmethod
	def as_dataframe(cls):
		return pd.DataFrame({
			'name':['spectrum', '1st moment', '2nd moment', '3rd moment', '4th moment'],
			'varname':['spectrum', 'moment_l1', 'moment_l2', 'moment_l3', 'moment_l4'],
			'symbol': ['lambda_i', 'M(l=1)', 'M(l=2)', 'M(l=3)', 'M(l=4)'],
			'scope': ['transform', 'graph', 'graph', 'graph', 'graph'],
			'function': [cls.spectrum, cls.moment_l1, cls.moment_l2, cls.moment_l3, cls.moment_l4],
			'dependence': [None, ['spectrum'], ['spectrum'], ['spectrum'], ['spectrum']],
			'default': [{'precision':12, 'max_nodes':100}, *([None] * 4)],
			'category': 'spectral analysis'
		})

#====================================
class Walks():
	
	def __init__(self):
		return NotImplemented
	
	# funções auxiliares
	#====================================
	@classmethod
	def deterministic_walk(cls, G, v, stop_cond, memory, choose):
		V = G.nodes()
		route = [v]
		timeout = np.zeros([len(V)], dtype=int)
		timeout[v] = memory
		
		while not stop_cond(route):
			neigh = [ u for u in G.neighbors(v) if not timeout[u] ]
			timeout = np.clip(timeout - 1, 0, None)
			if len(neigh) == 0:
				break
			v = choose(v, neigh)
			route.append(v)
			timeout[v] = memory
		return route
	
	@classmethod
	def stochastic_walk(cls, G, v, stop_cond, memory):
		random_choice = lambda u, V: np.random.choice(V)
		return cls.deterministic_walk(G, v, stop_cond, memory, random_choice)
	
	@classmethod
	def walk_routes(cls, V, walk_strategy, qt_reps:int=1) -> dict:
		routes = { v: [] for v in V }
		for (v, _) in itt.product(V, range(qt_reps)):
			routes[v].append(walk_strategy(v))
		return routes
	
	@classmethod
	def walk_frequencies(cls, V, routes:dict) -> dict:
		freqs = np.zeros([len(V)])
		all_routes = list(routes.values())
		visited = itt.chain( *list( itt.chain(all_routes) ) )
		for v in visited:
			freqs[v] += 1
		total_count = freqs.sum()
		# calcula a qtde. de vezes que um nó foi visitado, 
		# ignorando o registro de início da caminhada
		return { v: (freqs[v] - len(routes[v])) / total_count for v in V }
	
	@classmethod
	def walk_deg_distrib(cls, deg:dict, freqs:dict) -> list:
		max_deg = np.max([ *deg.values() ])
		P = np.zeros([ max_deg + 1 ])
		for v in deg:
			P[ deg[v] ] += freqs[v]
		return P
	
	# rotas das caminhadas
	#====================================
	@classmethod
	def random_walk(cls, G, max_iter:int=20, qt_reps:int=5) -> dict:
		stop_cond = lambda r: len(r) >= max_iter
		walk_strategy = lambda v: cls.stochastic_walk(G, v, stop_cond, 0)
		return cls.walk_routes(G.nodes(), walk_strategy, qt_reps)
	
	@classmethod
	def self_avoiding_walk(cls, G, max_iter:int=20, qt_reps:int=5) -> dict:
		stop_cond = lambda r: len(r) >= max_iter
		walk_strategy = lambda v: cls.stochastic_walk(G, v, stop_cond, 1)
		return cls.walk_routes(G.nodes(), walk_strategy, qt_reps)
	
	@classmethod
	def det_tourist_walk(cls, G, max_iter:int=100, memory:int=1) -> dict:
		stop_cond = lambda r: len(r) >= max_iter or (len(r) and r[-1] in r[:-1])
		deg_min_diff = lambda u, V: V[ np.argmin(np.abs( np.array(G.degree(V))[:,1] - G.degree(u) )) ]
		walk_strategy = lambda v: cls.deterministic_walk(G, v, stop_cond, memory, deg_min_diff)
		return cls.walk_routes(G.nodes(), walk_strategy, 1)
	
	# frequência de visitação dos nó
	#====================================
	@classmethod
	def random_walk_freqs(cls, G, rw:dict=None) -> dict:
		if rw is None:			rw = cls.random_walk(G)
		return cls.walk_frequencies(G.nodes(), rw)
	
	@classmethod
	def self_avoiding_walk_freqs(cls, G, saw:dict=None) -> dict:
		if saw is None:			saw = cls.self_avoiding_walk(G)
		return cls.walk_frequencies(G.nodes(), saw)
	
	@classmethod
	def det_tourist_walk_freqs(cls, G, dtw:dict=None) -> dict:
		if dtw is None:			dtw = cls.det_tourist_walk(G)
		return cls.walk_frequencies(G.nodes(), dtw)
	
	# distribuição de probabilidades
	#====================================
	@classmethod
	def random_walk_distrib(cls, G, deg:dict=None, freq_rw:dict=None) -> list:
		if deg is None:			deg = Connectivity.vertex_degree(G)
		if freq_rw is None:		freq_rw = cls.walk_frequencies(G.nodes(), cls.random_walk(G))
		return cls.walk_deg_distrib(deg, freq_rw)
	
	@classmethod
	def self_avoiding_walk_distrib(cls, G, deg:dict=None, freq_saw:dict=None) -> list:
		if deg is None:			deg = Connectivity.vertex_degree(G)
		if freq_saw is None:	freq_saw = cls.walk_frequencies(G.nodes(), cls.self_avoiding_walk(G))
		return cls.walk_deg_distrib(deg, freq_saw)
	
	@classmethod
	def det_tourist_walk_distrib(cls, G, deg:dict=None, freq_dtw:dict=None) -> list:
		if deg is None:			deg = Connectivity.vertex_degree(G)
		if freq_dtw is None:	freq_dtw = cls.walk_frequencies(G.nodes(), cls.det_tourist_walk(G))
		return cls.walk_deg_distrib(deg, freq_dtw)
	
	@classmethod
	def as_dataframe(cls):
		return pd.DataFrame({
			'name':['random walk', 'self-avoiding walk', 'deterministic tourist walk', 
					'random walk frequencies', 'self-avoiding walk frequencies', 
					'deterministic tourist walk frequencies', 
					'random walk distribution', 'self-avoiding walk distribution', 
					'deterministic tourist walk distribution'],
			'varname':['rw', 'saw', 'dtw', 'freq_rw', 'freq_saw', 'freq_dtw', 'p_deg_rw', 'p_deg_saw', 'p_deg_dtw'],
			'symbol': ['RW', 'SAW', 'DTW', 
					   'P(RW|v)', 'P(SAW|v)', 'P(DTW|v)', 
					   'P(RW|k)', 'P(SAW|k)', 'P(DTW|k)'],
			'scope': ['iteration', 'iteration', 'iteration', 
					  'vertex', 'vertex', 'vertex', 
					  'distrib', 'distrib', 'distrib'],
			'function': [cls.random_walk, cls.self_avoiding_walk, cls.det_tourist_walk, 
						 cls.random_walk_freqs, cls.self_avoiding_walk_freqs, cls.det_tourist_walk_freqs, 
						 cls.random_walk_distrib, cls.self_avoiding_walk_distrib, cls.det_tourist_walk_distrib],
			'dependence': [None, None, None, 
						   ['random walk'], ['self-avoiding walk'], ['deterministic tourist walk'], 
						   ['vertex degree', 'random walk frequencies'], 
						   ['vertex degree', 'self-avoiding walk frequencies'], 
						   ['vertex degree', 'deterministic tourist walk frequencies']],
			'default': [{'max_iter':20, 'qt_reps':5}, 
						{'max_iter':20, 'qt_reps':5}, 
						{'max_iter':100, 'memory':1}, 
						*([None] * 6)],
			'category': 'walks'
		})

#====================================
class Automata():
	
	def __init__(self):
		return NotImplemented
	
	# autômato generalista
	#====================================
	@classmethod
	def randomized_states(cls, G, state_space:list) -> np.ndarray:
		state_space = list(set(state_space))
		return np.random.choice(state_space, G.count_nodes())
	
	@classmethod
	def generalized_automaton(cls, G, transition_rule, num_steps:int, init_state:np.ndarray) -> tuple:
		teps = np.zeros([ G.count_nodes(), num_steps+1 ], dtype=init_state.dtype)
		teps[:, 0] = init_state
		for t in range(num_steps):
			for v in G.nodes():
				own_state = teps[v, t]
				neigh_state = teps[G.neighbors(v), t]
				teps[v, t+1] = transition_rule(own_state, neigh_state)
		return tuple(map(tuple, teps))
	
	# descritores generalizados
	#====================================
	@classmethod
	def shannon_entropy(cls, G, teps:np.ndarray) -> list:
		f = lambda p: float(-p*np.log2(p) if p else 0)
		Es = []
		for samples in teps:
			(qt0, qt1) = np.bincount(samples, minlength=2)
			p0 = qt0 / float(qt0+qt1)
			p1 = qt1 / float(qt0+qt1)
			Es.append(f(p0) + f(p1))
		return Es
	
	@classmethod
	def word_entropy(cls, G, teps:np.ndarray) -> list:
		f = lambda p: float(-p*np.log2(p) if p else 0)
		Ew = []
		for phrase in teps:
			p = 0
			# under revision
			#-------------------------------------------------
			#for word_len in range(1, len(phrase) - 2):
			#	vocabulary_len = (len(phrase) - word_len + 1)
			#	qt_const_words = np.sum([ 1 if not (phrase[i:i+word_len].sum() % word_len) else 0 for i in range(vocabulary_len) ])
			#	p += f(qt_const_words / float(vocabulary_len))
			#-------------------------------------------------
			Ew.append(p)
		return Ew
	
	@classmethod
	def binary_patterns(cls, G, teps:np.ndarray, digits:int=5, normalize:bool=True) -> list:
		BP = []
		norm_factor = 2**digits - 1 if normalize else 1
		binary_filter = 2**(np.arange(digits, 0, -1) - 1)
		for signal in teps:
			pattern = np.convolve(signal.astype(float), binary_filter, 'valid')
			BP.append(pattern / norm_factor)
		return BP
	
	@classmethod
	def density_patterns(cls, G, teps:np.ndarray) -> tuple:
		dteps = np.zeros(teps.shape)
		for v in G.nodes():
			neighs = G.neighbors(v)
			if len(neighs):
				dteps[v] = teps[neighs].mean(0)
		sdteps = ((teps - 0.5) * dteps) + 0.5
		return (
			tuple(map(tuple, dteps)),
			tuple(map(tuple, sdteps))
		)
	
	@classmethod
	def spatial_teps_descriptors(cls, G, teps:tuple=None, extractors:list=['Es'], ti:Union[int,object]=1, tf:Union[int,object]=None, resolution:Union[int,object]=None, **kwargs) -> dict:
		teps = np.array(teps)[:, ti:tf]
		if extractors == ['all']:
			extractors = ['Es', 'Ew', 'BP', 'P', 'Q']
		(dteps, sdteps) = cls.density_patterns(G, teps) if ('P' in extractors or 'Q' in extractors) else (None, None)
		f = {
			'Es': 	cls.shannon_entropy,
			#'Ew': 	cls.word_entropy,
			'BP': 	cls.binary_patterns,
			'P': 	lambda *args, **kwargs: dteps,
			'Q': 	lambda *args, **kwargs: sdteps
		}
		translate = lambda e: e.split('_')[0]
		params = { e: (kwargs[e] if e in kwargs else {}) for e in extractors }
		return { e: f[ translate(e) ](G, teps, **params[e]) for e in extractors if translate(e) in f }
	
	@classmethod
	def temporal_teps_descriptors(cls, G, teps:tuple=None, extractors:list=['Es'], ti:Union[int,object]=1, tf:Union[int,object]=None, resolution:int=0, **kwargs) -> dict:
		teps = np.array(teps)[:, ti:tf]
		if extractors == ['all']:
			extractors = ['Es', 'Ew', 'BP', 'P', 'Q']
		if ('P' in extractors or 'Q' in extractors):
			(dteps, sdteps) = cls.density_patterns(G, teps)
		else:
			(dteps, sdteps) = (None, None)
		f = {
			'Es': 	cls.shannon_entropy,
			#'Ew': 	cls.word_entropy, 
			'BP': 	cls.binary_patterns,
			'P': 	lambda *args, **kwargs: tuple(map(tuple, np.array(dteps).T)),
			'Q': 	lambda *args, **kwargs: tuple(map(tuple, np.array(sdteps).T))
		}
		hist = lambda z: np.histogram(z, bins=resolution, range=(0,1), density=True)[0] / resolution
		g = lambda x: tuple(map(hist, x)) if resolution and len(np.shape(x)) > 1 else x
		translate = lambda e: e.split('_')[0]
		params = { e: (kwargs[e] if e in kwargs else {}) for e in extractors }
		return { e: g(f[ translate(e) ](G, teps.T, **params[e])) for e in extractors if translate(e) in f }
	
	@classmethod
	def aggregation_parameters(cls, extractors:list, resolution:Union[int,list,dict]) -> list:
		if isinstance(resolution, int):
			return [ (e, resolution) for e in extractors ]
		elif isinstance(resolution, list):
			return [ *itt.product(extractors, resolution) ]
		elif isinstance(resolution, dict):
			params = []
			for e in extractors:
				params += cls.aggregation_parameters([e], resolution[e])
			return params
	
	@classmethod
	def aggregated_teps_descriptors(cls, G, zv:dict={}, resolution:Union[int,list,dict]=10, segment_by:list=None) -> dict:
		params = cls.aggregation_parameters(zv.keys(), resolution)
		hist = lambda z, r: np.histogram(z, bins=r, range=(0,1), density=True)[0] / r
		if segment_by is None:
			return { f'[{e},{r}]': tuple(hist(np.array(zv[e]).flatten(), r)) for (e, r) in params }
		criterion = np.array(segment_by)
		masks = [ (criterion == c) for c in np.unique(criterion) ]
		zg_agg = {}
		for e, r in params:
			arr = np.array(zv[e])
			zv_arr = [ arr[m] for m in masks ]
			zg_mtx = np.stack([ hist(zv_c.flatten(), r) for zv_c in zv_arr ])
			zg_agg.update({ 
				f'avg([{e},{r}])': tuple(zg_mtx.mean(0)), 
				f'std([{e},{r}])': tuple(zg_mtx.std(0)) 
			})
		return zg_agg
	
	# autômato totalístico
	#====================================
	@classmethod
	def tstna(cls, G, num_steps:int=100, threshold:float=0.5) -> tuple:
		transition_rule = lambda so, sn: (so if not(len(sn)) or np.mean(sn) <= threshold else not so)
		init_state = cls.randomized_states(G, [False, True])
		return cls.generalized_automaton(G, transition_rule, num_steps, init_state)
	
	@classmethod
	def tstna_aggregated_descriptors(cls, G, tstna_teps:list=None, extractors:list=['all'], resolution:Union[int,list,dict]=10, **kwargs) -> dict:
		tstna_zv = cls.spatial_teps_descriptors(G, tstna_teps, extractors, **kwargs)
		return cls.aggregated_teps_descriptors(G, tstna_zv, resolution, None)
	
	# autômato inspirado no jogo da vida
	#====================================
	@classmethod
	def pertinence_check(cls, density:float, resolution:int, rule:list) -> bool:
		for k in rule:
			interval_check = (k <= density*resolution and density*resolution < k+1)
			r_border_check = (k+1==resolution and density==1)
			if interval_check or r_border_check:
				return True
		return False
	
	@classmethod
	def llna(cls, G, num_steps:int=100, resolution:int=5, x:list=[1,3], y:list=[2,4]) -> tuple:
		for part in [x,y]:
			assert (min(part) >= 0 and max(part) < resolution)
		density = lambda sn: np.mean(sn) if len(sn) else 0
		born = lambda so, sn: (not so and cls.pertinence_check(density(sn), resolution, x))
		survive = lambda so, sn:  (so and cls.pertinence_check(density(sn), resolution, y))
		transition_rule = lambda so, sn: (born(so, sn) or survive(so, sn))
		init_state = cls.randomized_states(G, [False, True])
		return cls.generalized_automaton(G, transition_rule, num_steps, init_state)
	
	@classmethod
	def llna_aggregated_descriptors(cls, G, llna_teps:list=None, deg:list=None, extractors:list=['all'], resolution:Union[int,list,dict]=10, **kwargs) -> dict:
		if deg is None:			deg = Connectivity.vertex_degree(G)
		llna_zv = cls.spatial_teps_descriptors(G, llna_teps, extractors, **kwargs)
		llna_zg = {}
		llna_zg.update( cls.aggregated_teps_descriptors(G, llna_zv, resolution, None) )
		llna_zg.update( cls.aggregated_teps_descriptors(G, llna_zv, resolution, deg) )
		return llna_zg
	
	@classmethod
	def as_dataframe(cls):
		return pd.DataFrame({
			'name':['totalistic single-threshold network automaton', 
					'network descriptors from TSTNA\'s TEPs',
					'life-like network automaton', 
					'network descriptors from LLNA\'s TEPs'],
			'varname':['tstna_teps', 'tstna_hist', 
					   'llna_teps', 'llna_hist'],
			'symbol': ['TSTNA-TEPs', 'hist(TSTNA-TEPs|G)', 
					   'LLNA-TEPs', 'hist(LLNA-TEPs|G,k)'],
			'scope': ['iteration', 'graph', 
					  'iteration', 'graph'],
			'function': [cls.tstna, cls.tstna_aggregated_descriptors,
						 cls.llna, cls.llna_aggregated_descriptors],
			'dependence': [None, ['totalistic single-threshold network automaton'], 
						   None, ['life-like network automaton', 'vertex degree']],
			'default': [{'num_steps':100, 'threshold':0.5}, 
						{'extractors':['Es'], 'BP':{'digits':5}, 'resolution':[16]},
						{'num_steps':100, 'resolution':5, 'x':[1,3], 'y':[2,4]}, 
						{'extractors':['Es'], 'BP':{'digits':5}, 'resolution':[16]}], 
			'category': 'automata'
		})


#====================================
#class Representation():
#	
#	def __init__(self):
#		return NotImplemented
#	
#	@classmethod
#	def vertex_ordering(cls, G, by:list=[], asc:bool=True) -> list:
#		if not isinstance(by, (list, tuple)) or not len(by):
#			raise
#		V = G.nodes()
#		df = pd.DataFrame({'vertex': V})
#		cols = []
#		for i, x in enumerate(by):
#			c = 'x' + str(i)
#			cols.append(c)
#			df[c] = [ x[v] for v in V ]
#		df.sort_values(by=cols, ascending=asc, inplace=True)
#		return list(df['vertex'].values)
#	
#	@classmethod
#	def adj_mtx_image(cls, G, l:int=128, s:float=0.25, mapping:str='lin') -> Image:
#		if mapping == 'lin':
#			n = max(l, np.ceil(G.count_nodes()*s + 1).astype(int))
#			f = lambda e: np.ceil(e*s).astype(int)
#		elif mapping == 'exp':
#			n = max(l, np.ceil(G.count_nodes()*s + 1).astype(int))
#			f = lambda e: np.ceil((l-1) * (1 - np.exp(-s*e/50))).astype(int)
#		else:
#			raise
#		img = Image.new('L', (n,n))
#		for e in np.array(G.edges()):
#			u,v = f(e)
#			if max(u,v) > n:
#				raise
#			img.putpixel((u,v), (255))
#			img.putpixel((v,u), (255))
#		return img.crop((0,0,l,l))
#	
#	@classmethod
#	def ordered_graph(cls, G, deg:list=None, bc:list=None, asc:bool=False):
#		if deg is None:			deg = Connectivity.vertex_degree(G)
#		if bc is None:			bc = Centrality.betweenness_centrality(G)
#		V = cls.vertex_ordering(G, [deg, bc], asc)
#		mapping = { v: i for i, v in enumerate(V) }
#		E = G.edges()
#		H = G.copy()
#		for u, v in E:	H.rmv_edge((u,v))
#		for u, v in E: 	H.add_edge((mapping[u], mapping[v]))
#		return H
#	
#	@classmethod
#	def ordered_graph_image(cls, G, g_sorted=None, **kwargs):
#		if g_sorted is None:	g_sorted = cls.ordered_graph(G)
#		return np.stack([ 
#			cls.adj_mtx_image(g_sorted, mapping='lin', **kwargs),
#			cls.adj_mtx_image(g_sorted, mapping='exp', **kwargs)
#		], axis=0)
#	
#	@classmethod
#	def as_dataframe(cls):
#		return pd.DataFrame({
#			'name':['ordered graph', 'ordered graph image'],
#			'varname':['g_sorted', 'graph2img'],
#			'symbol': ['sort(G|k,B))', 'img(G|k,B))'],
#			'scope': ['transform', 'transform'],
#			'function': [cls.ordered_graph, cls.ordered_graph_image],
#			'dependence': [['vertex degree','betweenness centrality'], ['ordered graph']],
#			'default': [{'asc':False}, {'l':128, 's':0.25}],
#			'category': 'representation'
#		})

