import numpy as np
import igraph as ig
import networkx as nx

class GraphAPI():
	def __init__(self, graph): self.data = graph
	def __len__(self): return self.count_nodes() + self.count_edges()
	
	def _isnx(self): return isinstance(self.data, nx.Graph)
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
			return GraphAPI( nx.DiGraph(A) if self.data.is_directed() else nx.Graph(A) )
		elif self._isnx() and engine in ['ig', 'igraph']:
			return GraphAPI( ig.Graph.TupleList(self.data.edges(), directed=self.data.is_directed()) )
		return self.copy()
	
	def nodes(self):
		if   self._isnx(): return list(self.data.nodes)
		elif self._isig(): return list([ v.index for v in self.data.vs ])
		else			 : return []
	
	def edges(self):
		if   self._isnx(): return list(self.data.edges)
		elif self._isig(): return list([ e.tuple for e in self.data.es ])
		else			 : return []
	
	def neighbors(self, u):
		if   self._isnx(): return list(self.data.neighbors(u))
		elif self._isig(): return self.data.neighbors(u)
		else			 : return []
	
	def to_adj_matrix(self):
		if   self._isnx(): return nx.to_numpy_array(self.data)
		elif self._isig(): return np.array(list(self.data.get_adjacency()))
		else			 : return np.array([[]])
	
	def degree(self, param=None):
		if   self._isnx() and param is None	 : return nx.degree(self.data)
		if   self._isnx() and self._isint(param): return nx.degree(self.data, param)
		elif self._isnx() and self._islst(param): return nx.degree(self.data, param)
		elif self._isig() and param is None	 : return [ (u,k) for u,k in enumerate(self.data.degree()) ]
		elif self._isig() and self._isint(param): return self.data.degree(param)
		elif self._isig() and self._islst(param): return [ (u,k) for u,k in zip(param, self.data.degree(param)) ]
		else									: return None
	
	def resume(self, dtype=True):
		dtype_str = f'{type(self.data)}\n' if dtype else ''
		return f'{dtype_str}{self.count_nodes()} nodes, {self.count_edges()} edges'
	
	def add_node(self, param):
		if   self._isnx() and self._isint(param): self.data.add_node(param)
		elif self._isnx() and self._islst(param): self.data.add_nodes_from(param)
		elif self._isig() and self._isint(param): self.data.add_vertex(param)
		elif self._isig() and self._islst(param): self.data.add_vertices(param)
		else									: raise
	
	def add_edge(self, param):
		if   self._isnx() and self._istpl(param): self.data.add_edge(*param)
		elif self._isnx() and self._islst(param): self.data.add_edges_from(param)
		elif self._isig() and self._istpl(param): self.data.add_edge(*param)
		elif self._isig() and self._islst(param): self.data.add_edges(param)
		else									: raise
	
	def rmv_node(self, param):
		if   self._isnx() and self._istpl(param): self.data.remove_node(*param)
		elif self._isnx() and self._islst(param): self.data.remove_nodes_from(param)
		elif self._isig() and self._istpl(param): self.data.delete_vertex(*param)
		elif self._isig() and self._islst(param): self.data.delete_vertices(param)
		else									: raise
	
	def rmv_edge(self, param):
		if   self._isnx() and self._istpl(param): self.data.remove_edge(*param)
		elif self._isnx() and self._islst(param): self.data.remove_edges_from(param)
		elif self._isig() and self._istpl(param): self.data.delete_edges(self.data.get_eid(*param))
		elif self._isig() and self._islst(param): self.data.delete_edges([ self.data.get_eid(p) for p in param ])
		else									: raise
	
	def count_nodes(self):
		if   self._isnx(): return self.data.number_of_nodes()
		elif self._isig(): return len(self.data.vs)
		else			 : return 0
	
	def count_edges(self):
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
		else: raise


