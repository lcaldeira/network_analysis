import numpy as np
import igraph as ig
import networkx as nx
import itertools as itt

class GraphAPI():
	def __init__(self, graph): self.data = graph
	def __len__(self): return self.num_nodes() + self.num_edges()
	
	def _isnx(self): return isinstance(self.data, tuple([nx.Graph, nx.DiGraph]))
	def _isig(self): return isinstance(self.data, ig.Graph)
	def _isint(self, var): return isinstance(var, int)
	def _islst(self, var): return isinstance(var, list)
	def _istpl(self, var): return isinstance(var, tuple)
	
	def engine(self): return 'networkx' if self._isnx() else 'igraph'
	
	def copy(self):
		return GraphAPI(self.data.copy())
	
	def to(self, engine):
		if self._isig() and engine in ['nx', 'networkx']:
			A = self.data.get_edgelist()
			return GraphAPI( nx.DiGraph(A) if self.is_directed() else nx.Graph(A) )
		elif self._isnx() and engine in ['ig', 'igraph']:
			return GraphAPI( ig.Graph.TupleList(self.data.edges(), directed=self.is_directed()) )
		return self.copy()
	
	def is_directed(self):
		if   self._isnx(): return isinstance(self.data, nx.DiGraph)
		elif self._isig(): return self.data.is_directed()
		else			 : raise TypeError('Unknown graph data type.')
	
	def to_directed(self):
		if self.is_directed():
			return self
		elif self._isnx():
			return self.data.to_directed()
		elif self._isig():
			G = self.copy()
			G.data.to_directed()
			return G
	
	def to_undirected(self):
		if self.is_undirected():
			return self
		elif self._isnx():
			return self.data.to_undirected()
		elif self._isig():
			G = self.copy()
			G.data.to_undirected()
			return G
	
	def nodes(self):
		if   self._isnx(): return list(self.data.nodes)
		elif self._isig(): return list([ v.index for v in self.data.vs ])
		else			 : return []
	
	def edges(self):
		if   self._isnx(): return list(self.data.edges)
		elif self._isig(): return list([ e.tuple for e in self.data.es ])
		else			 : return []
	
	def neighbors(self, u, mode:str='all', astype=list):
		assert mode in ['all', 'in', 'out'], ValueError('The value of "mode" must be one of ["all", "in", "out"].')
		if   self._isnx() and mode == 'out': return astype(self.data.neighbors(u))
		elif self._isnx() and mode == 'in': return astype(self.data.predecessors(u))
		elif self._isnx() and mode == 'all': return astype(itt.chain(self.data.neighbors(u), self.data.predecessors(u)))
		elif self._isig(): return astype(self.data.neighbors(u, mode))
		else			 : return astype([])
	
	def to_adj_matrix(self):
		if   self._isnx(): return nx.to_numpy_array(self.data)
		elif self._isig(): return np.array(list(self.data.get_adjacency()))
		else			 : return np.array([[]])
	
	def degree(self, param=None, astype=list):
		if   self._isnx() and param is None	 : return astype(nx.degree(self.data))
		if   self._isnx() and self._isint(param): return astype(nx.degree(self.data, param))
		elif self._isnx() and self._islst(param): return astype(nx.degree(self.data, param))
		elif self._isig() and param is None	 : return astype(enumerate(self.data.degree()))
		elif self._isig() and self._isint(param): return astype(self.data.degree(param))
		elif self._isig() and self._islst(param): return astype(zip(param, self.data.degree(param)))
		else									: return None
	
	def resume(self, dtype=True):
		dtype_str = f'{type(self.data)}\n' if dtype else ''
		return f'{dtype_str}{self.num_nodes()} nodes, {self.num_edges()} edges'
	
	def add_node(self, param):
		if   self._isnx() and self._isint(param): self.data.add_node(param)
		elif self._isnx() and self._islst(param): self.data.add_nodes_from(param)
		elif self._isig() and self._isint(param): self.data.add_vertex(param)
		elif self._isig() and self._islst(param): self.data.add_vertices(param)
		else									: raise TypeError('Unknown graph data type.')
	
	def add_edge(self, param):
		if   self._isnx() and self._istpl(param): self.data.add_edge(*param)
		elif self._isnx() and self._islst(param): self.data.add_edges_from(param)
		elif self._isig() and self._istpl(param): self.data.add_edge(*param)
		elif self._isig() and self._islst(param): self.data.add_edges(param)
		else									: raise TypeError('Unknown graph data type.')
	
	def rmv_node(self, param):
		if   self._isnx() and self._istpl(param): self.data.remove_node(*param)
		elif self._isnx() and self._islst(param): self.data.remove_nodes_from(param)
		elif self._isig() and self._istpl(param): self.data.delete_vertex(*param)
		elif self._isig() and self._islst(param): self.data.delete_vertices(param)
		else									: raise TypeError('Unknown graph data type.')
	
	def rmv_edge(self, param):
		if   self._isnx() and self._istpl(param): self.data.remove_edge(*param)
		elif self._isnx() and self._islst(param): self.data.remove_edges_from(param)
		elif self._isig() and self._istpl(param): self.data.delete_edges(self.data.get_eid(*param))
		elif self._isig() and self._islst(param): self.data.delete_edges([ self.data.get_eid(*p) for p in param ])
		else									: raise TypeError('Unknown graph data type.')
	
	def add_self_loops(self):
		self.add_edge([ (v,v) for v in self.nodes() if v not in self.neighbors(v) ])
	
	def rmv_self_loops(self):
		self.rmv_edge([ (v,v) for v in self.nodes() if v in self.neighbors(v) ])
	
	def rmv_unconnected(self):
		self.rmv_node([ v for (v, k) in self.degree() if k == 0 ])
	
	def rmv_duplicates(self):
		E = []
		isdir = self.is_directed()
		if not isdir:
			self.to_directed()
		for v in self.nodes():
			for (neigh, count) in zip(*np.unique(self.neighbors(v, 'out'), return_counts=True)):
				if count > 1:
					E += [(v, neigh)]*(count-1)
		self.rmv_edge(E)
		if not isdir:
			self.to_undirected()
	
	def simplified(self):
		G = self.copy()
		G.rmv_self_loops()
		G.rmv_duplicates()
		G.rmv_unconnected()
		return G
	
	def subgraph(self, nodes:list=None, edges:list=None):
		G = self.copy()
		if nodes is not None:
			G.rmv_node([ v for v in G.nodes() if v not in nodes ])
		if edges is not None:
			G.rmv_edge([ e for e in G.edges() if e not in edges ])
		return G
	
	def connected_components(self):
		G = self.copy()
		cc = []
		while len(G) > 0:
			next_nodes = set(G.nodes()[:1])
			reached_nodes = set()
			while len(next_nodes) > 0:
				curr_nodes = next_nodes
				next_nodes = set()
				for v in curr_nodes:
					next_nodes.update(G.neighbors(v))
					reached_nodes.add(v)
				next_nodes -= reached_nodes
			reached_nodes = list(reached_nodes)
			cc.append(G.subgraph(nodes=reached_nodes))
			G.rmv_node(reached_nodes)
		return sorted(cc, key=len, reverse=True)
	
	def num_nodes(self):
		if   self._isnx(): return self.data.number_of_nodes()
		elif self._isig(): return len(self.data.vs)
		else			 : return 0
	
	def num_edges(self):
		if   self._isnx(): return self.data.number_of_edges()
		elif self._isig(): return len(self.data.es)
		else			 : return 0
	
	def draw(self, **kwargs):
		if   self._isnx(): return nx.draw(self.data, **kwargs)
		elif self._isig(): return ig.plot(self.data, **kwargs)
		else			 : return None
	
	@staticmethod
	def generate(engine, style='empty', **kwargs):
		if engine=='networkx' or engine=='nx':
			if   style=='ba'	: return GraphAPI(nx.barabasi_albert_graph(**kwargs))
			elif style=='er'	: return GraphAPI(nx.erdos_renyi_graph(**kwargs))
			elif style=='ws'	: return GraphAPI(nx.watts_strogatz_graph(**kwargs))
			elif style=='tree'  : return GraphAPI(nx.balanced_tree(**kwargs))
			elif style=='empty' : return GraphAPI(nx.Graph())
			else				: return None
		elif engine=='igraph' or engine=='ig':
			if   style=='ba'	: return GraphAPI(ig.Graph.Barabasi(**kwargs))
			elif style=='er'	: return GraphAPI(ig.Graph.Erdos_Renyi(**kwargs))
			elif style=='ws'	: return GraphAPI(ig.Graph.Watts_Strogatz(**kwargs))
			elif style=='tree'  : return GraphAPI(ig.Graph.Tree(**kwargs))
			elif style=='empty' : return GraphAPI(ig.Graph())
			else				: return None
		else: raise ValueError('The "engine" value must be one of ["igraph", "ig", "networkx", "nx"].')


