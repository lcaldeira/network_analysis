import os
import numpy as np
import pandas as pd
import itertools as itt
from tqdm import trange, tqdm
from .containers import GraphAPI
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

#====================================
class BaseNetworkCollection():
	_base = '.'
	_seed = None
	_mode = 'full'
	_collection = ''
	_datasets = []
	
	def __init__(self):
		raise NotImplemented
	
	@classmethod
	def _filelist(cls, folder:str):
		raise NotImplementedError
	
	@classmethod
	def _parse(cls, content):
		raise NotImplementedError
	
	@classmethod
	def _load(cls, dataset:str, samples_per_label:int=None, balanced:bool=False, transform:callable=None, allow_invalid:bool=False):
		raise NotImplementedError
	
	@classmethod
	def __getitem__(cls, file_name:str, skip_rows:int=0):
		with open(file_name, 'r') as f:
			return cls._parse(f.readlines()[skip_rows:])
	
	@classmethod
	def _sampling(cls, samples, number:int, balanced:bool=False):
		if balanced:
			number = np.min([ *samples.groupby('label').count()['source'], number ])
			func = lambda x: x.sample(n=number, replace=False, random_state=cls._seed)
		else:
			func = lambda x: x.sample(n=np.min([ len(x), number ]), random_state=cls._seed)
		grouped = samples.groupby('label', group_keys=False)
		return grouped[samples.columns].apply(func, include_groups=False)
	
	@classmethod
	def _progressbar(cls, files, msg:str):
		pbar_msg = f'[{msg}]'
		return tqdm(files, desc=f'{pbar_msg:30s}', ncols=80, leave=True, position=0)
	
	@classmethod
	def all(cls, *args, **kwargs):
		return pd.concat([ cls._load(sg, *args, **kwargs) for sg in cls._datasets ], ignore_index=True)
	
	@classmethod
	def setup(cls, base_folder:str='.', random_seed:int=None, loader_mode:str='scan'):
		if not isinstance(base_folder, str):
			raise TypeError("The value of `base_folder` must be a string")
		if not isinstance(random_seed, int) and random_seed is not None:
			raise TypeError("Only integers are allowed as `random_seed`")
		if loader_mode not in ['full', 'part', 'scan']:
			raise Exception("Allowed values for `loader_mode` are \"full\", \"part\" and \"scan\"")
		cls._base = base_folder
		cls._seed = random_seed
		cls._mode = loader_mode
	

#====================================
class AicsSyntheticNetworks(BaseNetworkCollection):
	
	_collection = 'aics-synthetic'
	_datasets = ['classic', 'scalefree', 'noise=10', 'noise=20', 'noise=30']
	
	def __init__(self):
		return NotImplemented
	
	@classmethod
	def _filelist(cls, folder):
		assert(folder in cls._datasets)
		file_list = []
		rename = {
			'WS': ['watts-strogatz', 'WS'],
			'ER': ['erdos-renyi', 'ER'],
			'BA': ['barabasi-albert', 'BA'],
			'GEO': ['geographic', 'GEO'],
			'DM': ['dorogovtsev-mendes', 'DM'],
			'BANL05': ['scale-free nlin (0.5)', 'BAnl05'],
			'BANL15': ['scale-free nlin (1.5)', 'BAnl15'],
			'BANL20': ['scale-free nlin (2.0)', 'BAnl20']
		}
		for subdir, _, files in os.walk(cls._base + f'/{cls._collection}/{folder}'):
			if len(files) <= 1:
				continue
			last_dir = os.path.normpath(subdir).split(os.sep)[-1]
			file_list.append(pd.DataFrame({
				'collection': cls._collection,
				'dataset': folder,
				'label': rename[last_dir][0],
				'source': [ os.path.join(subdir, os.path.normpath(name)) for name in files ],
				'prefix': rename[last_dir][1]
			}))
		return pd.concat(file_list, ignore_index=True)
	
	@classmethod
	def _parse(cls, content):
		G = GraphAPI.generate('ig')
		N = content[1]
		v0 = content[2]
		G.add_node(list(range(N)))
		for line in content[0]:
			V = [ int(v) - v0 for v in line.split(' ') if v not in ['\n','\t'] ]
			G.add_edge(list( itt.product(V[:1], V[1:]) ))
		return G
	
	@classmethod
	def _load(cls, dataset:str, samples_per_label:int=None, balanced:bool=False, transform:callable=None, allow_invalid:bool=False):
		metadata = { key: [] for key in ['abbr', 'N', 'M', 'K', 'graph', 'source'] }
		samples = cls._filelist(dataset)
		if samples_per_label is not None:
			samples = cls._sampling(samples, samples_per_label, balanced)
		
		print(f'\n{cls._collection}:{dataset}\n' + '='*50)
		
		for label in sorted(list(samples['label'].unique())):
			mask = (samples['label'] == label)
			for _, row in cls._progressbar(list(samples[mask].iterrows()), label):
				filename = os.path.normpath(row['source'])
				prefix = row['prefix']
				with open(os.path.join(filename), 'r') as f:
					nameinfo = { 
						m.split('=')[0]: eval(m.split('=')[-1]) 
						for m in filename[:-len('.txt') ].split(os.sep)[-1].split('_')[1:]
					}
					N = nameinfo['n']
					if not dataset.startswith('noise'):
						G = (None if cls._mode in ['scan','part'] else cls._parse([ f.readlines(), N, 0 ]))
						K = nameinfo['k']
					else:
						G = cls._parse([ f.readlines(), N, 1 ])
						K = round(2*G.num_edges()/N if N else 0, 2)
						if cls._mode in ['scan','part']:
							G = None
					r = nameinfo['i']
					p = nameinfo['p'] if 'p' in nameinfo else None
					abbr = f'{prefix}(N={N}, K={K}' + (f', p={p}' if p is not None else '') + f')_#{r}'
					M = (int(N*K/2.0) if cls._mode in ['scan','part'] else G.num_edges())
					
				netword_id = filename.split(os.sep)[-1]
				metadata['abbr'].append(abbr)
				metadata['graph'].append(G if cls._mode != 'part' else None)
				metadata['N'].append(N)
				metadata['M'].append(M)
				metadata['K'].append(K)
				metadata['source'].append(filename)
		
		samples.drop(columns='prefix', inplace=True)
		metadata = pd.DataFrame(metadata)
		columns = list(samples.columns[:-1]) + list(metadata.columns)
		networks = samples.merge(metadata)[columns].sort_values('abbr')
		if not allow_invalid:
			mask = (networks['N'] > 0) & (networks['M'] > 0)
			networks = networks[mask]
		return networks if transform is None else transform(networks)
	
	@classmethod
	def classic(cls, *args, **kwargs):
		return cls._load('classic', *args, **kwargs)
	
	@classmethod
	def scalefree(cls, *args, **kwargs):
		return cls._load('scalefree', *args, **kwargs)
	
	@classmethod
	def noise10(cls, *args, **kwargs):
		return cls._load('noise=10', *args, **kwargs)
	
	@classmethod
	def noise20(cls, *args, **kwargs):
		return cls._load('noise=20', *args, **kwargs)
	
	@classmethod
	def noise30(cls, *args, **kwargs):
		return cls._load('noise=30', *args, **kwargs)


#====================================
class KeggMetabolicNetworks(BaseNetworkCollection):
	
	_collection = 'kegg-metabolic'
	_datasets = ['actinobacteria', 'firmicutes-bacillis', 'protist', 'fungi', 'plant', 'animals', 'kingdom']
	
	def __init__(self):
		return NotImplemented
	
	@classmethod
	def _filelist(cls, folder):
		assert(folder in cls._datasets)
		subdir = cls._base + f'/{cls._collection}/{folder}'
		df = pd.read_csv(subdir + '/labels/classes.txt', sep=' ', header=None, names=['label','source'])
		df['source'] = list(map(lambda name: os.path.join(subdir, 'graphs', os.path.normpath(name).split(os.sep)[-1]), df['source']))
		df.insert(0, 'collection', cls._collection)
		df.insert(1, 'dataset', folder)
		return df
	
	@classmethod
	def _parse(cls, content):
		N = len(content)
		E = [ (u, int(v)) for u, line in enumerate(content) for v in line.split(' ') ]
		G = GraphAPI.generate('ig')
		G.add_node(list(range(N)))
		G.add_edge(E)
		return G
	
	@classmethod
	def _load(cls, dataset:str, samples_per_label:int=None, balanced:bool=False, transform:callable=None, allow_invalid:bool=False):
		metadata = { key: [] for key in ['abbr', 'N', 'M', 'K', 'graph', 'source'] }
		samples = cls._filelist(dataset)
		if samples_per_label is not None:
			samples = cls._sampling(samples, samples_per_label, balanced)
		
		print(f'\n{cls._collection}:{dataset}\n' + '='*50)
		
		for label in list(samples['label'].unique()):
			mask = (samples['label'] == label)
			prefix = dataset[:5].capitalize() + label[:5].capitalize()
			for filename in cls._progressbar(samples[mask]['source'], label):
				with open(os.path.join(filename), 'r') as f:
					suffix = filename.split('_')[-2]
					content = f.readlines()
					N = len(content)
					G = (None if cls._mode == 'scan' else cls._parse(content))
					M = (None if cls._mode == 'scan' else G.num_edges())
					K = (None if cls._mode == 'scan' else round(M/N if N else 0, 2))
					
				metadata['abbr'].append(f'{prefix}:{suffix}')
				metadata['graph'].append(G if cls._mode != 'part' else None)
				metadata['N'].append(N)
				metadata['M'].append(M)
				metadata['K'].append(K)
				metadata['source'].append(filename)
				
		metadata = pd.DataFrame(metadata)
		columns = list(samples.columns[:-1]) + list(metadata.columns)
		networks = samples.merge(metadata)[columns].sort_values('abbr')
		if not allow_invalid:
			mask = (networks['N'] > 0) & (networks['M'] > 0)
			networks = networks[mask]
		return networks if transform is None else transform(networks)
	
	@classmethod
	def actinobac(cls, *args, **kwargs):
		return cls._load('actinobacteria', *args, **kwargs)
	
	@classmethod
	def animals(cls, *args, **kwargs):
		return cls._load('animals', *args, **kwargs)
	
	@classmethod
	def fbacillis(cls, *args, **kwargs):
		return cls._load('firmicutes-bacillis', *args, **kwargs)
	
	@classmethod
	def fungi(cls, *args, **kwargs):
		return cls._load('fungi', *args, **kwargs)
	
	@classmethod
	def kingdom(cls, *args, **kwargs):
		return cls._load('kingdom', *args, **kwargs)
	
	@classmethod
	def plant(cls, *args, **kwargs):
		return cls._load('plant', *args, **kwargs)
	
	@classmethod
	def protist(cls, *args, **kwargs):
		return cls._load('protist', *args, **kwargs)


#====================================
class SnapEgoNetworks(BaseNetworkCollection):
	
	_collection = 'snap-ego'
	_datasets = ['social-circles']
	
	def __init__(self):
		return NotImplemented
	
	@classmethod
	def _filelist(cls, folder):
		assert(folder in cls._datasets)
		file_list = []
		for subdir, _, files in os.walk(os.path.normpath(cls._base + f'/{cls._collection}/{folder}')):
			if len(files) <= 1:
				continue
			files = set(map(lambda name: name.split('.')[0], files))
			file_list.append(pd.DataFrame({
				'collection': cls._collection,
				'dataset': folder,
				'label': subdir.split(os.sep)[-1],
				'source': [ os.path.join(subdir, name) for name in files ]
			}))
		return pd.concat(file_list, ignore_index=True)
	
	@classmethod
	def _parse(cls, content):
		edge_list = list(map(lambda line: [ int(v) for v in line.split(' ') ], content))
		V = { v: i for i,v in enumerate(np.unique(edge_list)) }
		E = [ (V[i], V[j]) for i,j in edge_list ]
		G = GraphAPI.generate('ig')
		G.add_node(list(V.values()))
		G.add_edge(E)
		return G
	
	@classmethod
	def _load(cls, dataset:str, samples_per_label:int=None, balanced:bool=False, transform:callable=None, allow_invalid:bool=False):
		metadata = { key: [] for key in ['abbr', 'N', 'M', 'K', 'graph', 'source'] }
		encode = {
			'facebook': 'fb',
			'gplus': 'gp',
			'twitter': 'tt',
		}
		samples = cls._filelist(dataset)
		if samples_per_label is not None:
			samples = cls._sampling(samples, samples_per_label, balanced)
		
		print(f'\n{cls._collection}:{dataset}\n' + '='*50)
		
		for label in list(samples['label'].unique()):
			mask = (samples['label'] == label)
			for filename in cls._progressbar(samples[mask]['source'], label):
				filename = os.path.normpath(filename)
				with open(os.path.join(filename + '.edges'), 'r') as f:
					content = f.readlines()
					M = len(content)
					G = (None if cls._mode == 'scan' else cls._parse(content))
					N = (None if cls._mode == 'scan' else G.num_nodes())
					K = (None if cls._mode == 'scan' else round(M/N if N else 0, 2))
				netword_id = filename.split(os.sep)[-1]
				metadata['abbr'].append(f'{encode[label]}#{netword_id}')
				metadata['graph'].append(G if cls._mode != 'part' else None)
				metadata['N'].append(N)
				metadata['M'].append(M)
				metadata['K'].append(K)
				metadata['source'].append(filename)
		
		metadata = pd.DataFrame(metadata)
		columns = list(samples.columns[:-1]) + list(metadata.columns)
		networks = samples.merge(metadata)[columns].sort_values('abbr')
		if not allow_invalid:
			mask = (networks['N'] > 0) & (networks['M'] > 0)
			networks = networks[mask]
		return networks if transform is None else transform(networks)
	
	@classmethod
	def socialcircles(cls, *args, **kwargs):
		return cls._load('social-circles', *args, **kwargs)


#====================================
class TudNetworkCollection(BaseNetworkCollection):
	
	def __init__(self):
		return NotImplemented
	
	@classmethod
	def _filelist(cls, folder):
		assert(folder in cls._datasets)
		file_path = os.path.join(cls._base, cls._collection, folder, f'{folder}_')
		label_name = lambda code: cls._datasets[folder][int(code)]
		with open(file_path + 'graph_labels.txt', 'r') as f:
			content = f.readlines()
			labels = pd.DataFrame({
				'collection': cls._collection,
				'dataset': folder.lower(),
				'label': list(map(label_name, content)),
				'source': file_path + 'A.txt'
			})
		with open(file_path + 'graph_indicator.txt', 'r') as f:
			content = f.readlines()
			indicators = pd.DataFrame({
				'sample': list(map(int, content)),
				'node': list(range(len(content)))
			}).groupby('sample').aggregate(['min', 'max'])
			indicators.columns = ['begin', 'end']
			indicators.index.name = ''
			indicators.index -= 1
		return labels.join(indicators)
	
	@classmethod
	def _parse(cls, content):
		edge_list = list(map(lambda line: [ int(v) for v in line.split(',') ], content))
		V = { v: i for i,v in enumerate(np.unique(edge_list)) }
		E = [ (V[i], V[j]) for i,j in edge_list ]
		G = GraphAPI.generate('ig')
		G.add_node(list(V.values()))
		G.add_edge(E)
		return G
	
	@classmethod
	def _load(cls, dataset:str, samples_per_label:int=None, balanced:bool=False, transform:callable=None, allow_invalid:bool=False):
		metadata = { key: [] for key in ['abbr', 'N', 'M', 'K', 'graph', 'begin'] }
		samples = cls._filelist(dataset)
		if samples_per_label is not None:
			samples = cls._sampling(samples, samples_per_label, balanced).sort_values('begin')
		
		print(f'\n{cls._collection}:{dataset.lower()}\n' + '='*50)
		
		for label in sorted(samples['label'].unique()):
			mask = (samples['label'] == label)
			filename = os.path.normpath(samples[mask]['source'].values[0])
			assert(samples[mask]['label'].is_monotonic_increasing)
			with open(filename, 'r') as f:
				enumfile = enumerate(f)
				for idx in cls._progressbar(samples[mask].index, label):
					begin, end = samples.loc[idx, ['begin','end']]
					content = []
					for i, line in enumfile:
						if i < begin:
							continue
						elif i <= end:
							content.append(line)
						if i == end:
							break
					M = len(content)
					G = (None if cls._mode == 'scan' else cls._parse(content))
					N = (None if cls._mode == 'scan' else G.num_nodes())
					K = (None if cls._mode == 'scan' else round(M/N if N else 0, 2))
					metadata['abbr'].append(f'{dataset.lower()}#{idx+1}')
					metadata['graph'].append(G if cls._mode != 'part' else None)
					metadata['N'].append(N)
					metadata['M'].append(M)
					metadata['K'].append(K)
					metadata['begin'].append(begin)
					
		metadata = pd.DataFrame(metadata)
		columns = list(samples.columns[:-3]) + list(metadata.columns[:-1]) + ['source']
		networks = samples.merge(metadata)[columns].sort_values('abbr')
		if not allow_invalid:
			mask = (networks['N'] > 0) & (networks['M'] > 0)
			networks = networks[mask]
		return networks if transform is None else transform(networks)

	
#====================================
class TudSocialNetworks(TudNetworkCollection):
	
	_collection = 'tud-social'
	# the label names per dataset presented below may be incorrectly assigned to each label code, 
	# as the source website (TUDatasets) only provide the codes
	_datasets = {
		'COLLAB': {1:'High Energy Physics', 2:'Condensed Matter Physics', 3:'Astro Physics'}, 
		'IMDB-BINARY': {0:'Action', 1:'Romance'}, 
		'IMDB-MULTI': {1:'Comedy', 2:'Romance', 3:'Sci-Fi'}, 
		'REDDIT-BINARY': {-1:'question/answer-based', 1:'discussion-based'}, 
		'REDDIT-MULTI-5K': { i+1: sub for i, sub in enumerate([
				'worldnews', 'videos', 'AdviceAnimals', 'aww', 'mildlyinteresting'
			]) }, 
		'REDDIT-MULTI-12K': { i+1: sub for i, sub in enumerate([
				'AskReddit', 'AdviceAnimals', 'atheism', 'aww', 'IAmA', 'mildlyinteresting', 
				'Showerthoughts', 'videos', 'todayilearned', 'worldnews', 'TrollXChromosomes'
			]) }, 
		'deezer-ego-nets': { i: f'music genre #{i}' for i in range(2) }, 
		'github-stargazers': {0:'machine learning repo', 1:'web development repo'}, 
		'twitch-egos': {0:'single game player', 1:'multiple games player'}
	}
	
	def __init__(self):
		return NotImplemented
	
	@classmethod
	def collab(cls, *args, **kwargs):
		return cls._load('COLLAB', *args, **kwargs)
	
	@classmethod
	def imdb_b(cls, *args, **kwargs):
		return cls._load('IMDB-BINARY', *args, **kwargs)
	
	@classmethod
	def imdb_m(cls, *args, **kwargs):
		return cls._load('IMDB-MULTI', *args, **kwargs)
	
	@classmethod
	def reddit_b(cls, *args, **kwargs):
		return cls._load('REDDIT-BINARY', *args, **kwargs)
	
	@classmethod
	def reddit_m5k(cls, *args, **kwargs):
		return cls._load('REDDIT-MULTI-5K', *args, **kwargs)
	
	@classmethod
	def reddit_m12k(cls, *args, **kwargs):
		return cls._load('REDDIT-MULTI-12K', *args, **kwargs)
	
	@classmethod
	def deezer(cls, *args, **kwargs):
		return cls._load('deezer-ego-nets', *args, **kwargs)
	
	@classmethod
	def github(cls, *args, **kwargs):
		return cls._load('github-stargazers', *args, **kwargs)
	
	@classmethod
	def twitch(cls, *args, **kwargs):
		return cls._load('twitch-egos', *args, **kwargs)
	

#====================================
class TudBioinformaticNetworks(TudNetworkCollection):
	
	_collection = 'tud-bioinformatics'
	# the label names per dataset presented below may be incorrectly assigned to each label code, 
	# as the source website (TUDatasets) only provide the codes
	_datasets = {
		'DD': { i+1: f'dd_class_#{i+1:02d}' for i in range(2) }, 
		'ENZYMES': { i+1: f'enzymes_class_#{i+1:02d}' for i in range(6) }, 
		'MUTAG': { +1:'mutag_pos', -1:'mutag_neg' }, 
		'NCI1': { i: f'nci1_class_#{i+1:02d}' for i in range(2) }, 
		'NCI109': { i: f'nci109_class_#{i+1:02d}' for i in range(2) }, 
		'PROTEINS': { i+1: f'proteins_class_#{i+1:02d}' for i in range(2) } 
	}
	
	def __init__(self):
		return NotImplemented
	
	@classmethod
	def dd(cls, *args, **kwargs):
		return cls._load('DD', *args, **kwargs)
	
	@classmethod
	def enzymes(cls, *args, **kwargs):
		return cls._load('ENZYMES', *args, **kwargs)
	
	@classmethod
	def mutag(cls, *args, **kwargs):
		return cls._load('MUTAG', *args, **kwargs)
	
	@classmethod
	def nci_1(cls, *args, **kwargs):
		return cls._load('NCI1', *args, **kwargs)
	
	@classmethod
	def nci_109(cls, *args, **kwargs):
		return cls._load('NCI109', *args, **kwargs)
	
	@classmethod
	def proteins(cls, *args, **kwargs):
		return cls._load('PROTEINS', *args, **kwargs)
	


#====================================
class NetworksCrossCollection():
	def __init__(self, include:list=None, *args, **kwargs):
		if include is None:
			include = [
				SyntheticNetworks,
				KeggMetabolicNetworks,
				SnapEgoNetworks,
				TudSocialNetworks,
				TudBioinformaticNetworks
			]
		self.setup_args = args
		self.setup_kwargs = kwargs
		self.sources = {
			source._collection: source
			for source in include
		}
	
	def load(self, collection, dataset, *args, **kwargs):
		coll = self.sources[collection]
		coll.setup(*self.setup_args, **self.setup_kwargs)
		return coll._load(dataset, *args, **kwargs)
	
	def datasets(self):
		#return {
		#	coll: list(self.sources[coll]._datasets)
		#	for coll in self.sources
		#}
		return [
			(collname, dsname)
			for collname in self.sources
			for dsname in list(self.sources[collname]._datasets)
		]
	

