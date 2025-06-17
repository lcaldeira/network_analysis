import os
import gc
import time
import pickle
import numpy as np
import pandas as pd
import itertools as itt
from functools import reduce
from tqdm import tqdm

import sklearn as skl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import *



#====================================
class DataPool():
	
	def __init__(self):
		self.networks = pd.DataFrame(columns=['collection','dataset','label','N','M','K','abbr','graph','source'])
		self.measures = pd.DataFrame(columns=['name','varname','symbol','scope','category','function','dependence','default'])
		self.results  = pd.DataFrame(columns=['network_id', 'measure_id', 'value', 'time'])
		self.persist_fetched = True
		self.partitioned = False
	
	def reset(self, n=False, m=False, r=False):
		if n:			self.networks.drop(index=self.networks.index, inplace=True)
		if m:			self.measures.drop(index=self.measures.index, inplace=True)
		if m or n or r:	self.results.drop(index=self.results.index, inplace=True)
	
	def update(self, n=None, m=None, r=None):
		if n is not None:
			self.networks = pd.concat([self.networks, n], ignore_index=True)
			self.networks.drop_duplicates(subset=['dataset','abbr'], keep='first', inplace=True)
			self.networks.reset_index(drop=True, inplace=True)
			self.networks.index.name = 'network_id'
			self.networks = self.networks.astype({
				'collection': 'category',
				'dataset': 'category',
				'N': int,
				'M': int,
				'K': float,
				'label': 'category',
				'abbr': str,
				'graph': object,
				'source': str
			})
		if m is not None:
			# validar dependências antes de inserir
			self.measures = pd.concat([self.measures, m], ignore_index=True)
			self.measures.drop_duplicates(subset=['name'], keep='first', inplace=True)
			self.measures.reset_index(drop=True, inplace=True)
			self.measures.index.name = 'measure_id'
			self.measures = self.measures.astype(object)
		if r is not None:
			idx_cols = ['network_id', 'measure_id']
			if self.results.index.names != idx_cols:
				self.results[idx_cols] = self.results[idx_cols].astype(int)
				self.results.set_index(idx_cols, inplace=True)
			if r.index.names != idx_cols:
				r[idx_cols] = r[idx_cols].astype(int)
				r.set_index(idx_cols, inplace=True)
			self.results = pd.concat([self.results, r])
			self.results = self.results.iloc[ ~self.results.index.duplicated(keep='last') ]
	
	def save(self, folder:str, n:bool=False, m:bool=False, r:bool=False, strict:bool=False, tag:str='', subset:object=None):
		os.makedirs(folder, exist_ok=True)
		assert(subset is None or sum([n,m,r])==1)
		filepath = lambda name: os.path.join(folder, f'{name}.dat' if not len(tag) else f'{name}_[{tag}].dat')
		# control flags
		if not strict:
			with open(filepath('flags'), 'wb') as f:
				flags = (self.persist_fetched, self.partitioned)
				pickle.dump(flags, f, pickle.HIGHEST_PROTOCOL)
				print(f.name + ' saved in disk')
		# dataframes
		params = [
			('networks', self.partitioned, ['collection', 'dataset']), 
			('measures', False, None), 
			('results', False, None) 
		]
		for name, part, cols in [ p for (p, cond) in zip(params, [n, m, r]) if cond ]:
			if not part or strict:
				data = getattr(self, name) if subset is None else getattr(self, name)[subset]
				with open(filepath(name), 'wb') as f:
					pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
					print(f.name + ' saved in disk')
			else:
				data = getattr(self, name)
				concat = lambda row: ':'.join(row.values.astype(str))
				unique = sorted(set(data[cols].apply(concat, axis=1).values))
				kwargs = { kw: (kw==name[0]) for kw in ['n', 'm', 'r'] }
				#self.partitioned = False
				for value in unique:
					mask = reduce(lambda m1, m2: (m1 & m2), map(lambda c, n: (data[c]==n), cols, value.split(':')))
					self.save(folder, strict=True, tag=value, subset=mask, **kwargs)
				#self.partitioned = True
	
	def load(self, folder:str, n:bool=False, m:bool=False, r:bool=False, strict:bool=False, tag:str=''):
		if not os.path.exists(folder):
			print('the specified folder does not exists')
			return
		filepath = lambda name: os.path.join(folder, f'{name}.dat' if not len(tag) else f'{name}_[{tag}].dat')
		# control flags
		if not strict and os.path.exists(filepath('flags')):
			with open(filepath('flags'), 'rb') as f:
				flags = pickle.load(f)
				(self.persist_fetched, self.partitioned) = flags
				print(f.name + ' loaded in memory')
		# dataframes
		params = [
			('networks', self.partitioned), 
			('measures', False), 
			('results', False) 
		]
		for name, part in [ p for (p, cond) in zip(params, [n, m, r]) if cond ]:
			if (not part or strict) and os.path.exists(filepath(name)):
				with open(filepath(name), 'rb') as f:
					setattr(self, name, pickle.load(f))
					print(f.name + ' loaded in memory')
			elif part:
				kwargs = { kw: (kw==name[0]) for kw in ['n', 'm', 'r'] }
				content = filter(lambda item: item.startswith(name + '_'), os.listdir(folder))
				content = map(lambda item: os.path.join(folder, item), content)
				content = sorted(filter(os.path.isfile, content))
				dflist = []
				#self.partitioned = False
				for value in content:
					tag = value.split(']')[-2].split('[')[-1]
					self.load(folder, strict=True, tag=tag, **kwargs)
					dflist.append(getattr(self, name))
				setattr(self, name, pd.concat(dflist))
				#self.partitioned = True
	
	def info(self, include=None):
		info_dict = {}
		
		for attr in ['networks', 'measures', 'results']:
			if ((isinstance(include, str) and attr != include) or 
				(isinstance(include, list) and attr not in include)):
				continue
			table = getattr(self, attr)
			info_dict[attr] = pd.concat([
				pd.DataFrame({
					'dtype': str(table.index.dtype), 
					'count': len(table.index),
					'unique': len(table.index)
				}, index=['Index']),
				pd.merge(
					pd.DataFrame(table.dtypes, columns=['dtype']),
					table.describe(exclude=['Sparse[float]']).transpose()[['count', 'unique']],
					left_index=True,
					right_index=True,
					how='left'
				)
			]).join(
				pd.DataFrame(table.memory_usage(deep=True), columns=['bytes']),
				how='right'
			)
		keys = list(info_dict.keys())	
		return info_dict if len(keys) > 1 else info_dict[ keys[0] ]
	
	def drop(self, network_subset=None, measure_subset=None, results_only=True):
		if network_subset is not None:
			n_filter = self.networks.index[network_subset]
			r_filter = (self.results.index.get_level_values(0).isin(n_filter))
			if not results_only:
				self.networks.drop(index=n_filter, inplace=True)
			self.results.drop(index=self.results.index[r_filter], inplace=True)
		if measure_subset is not None:
			m_filter = self.measures.index[measure_subset]
			r_filter = (self.results.index.get_level_values(1).isin(m_filter))
			if not results_only:
				self.measures.drop(index=m_filter, inplace=True)
			self.results.drop(index=self.results.index[r_filter], inplace=True)
	
	def fetch_dependences(self, network_id, measure_id, recursion_limit=None, log_info=False):
		dep_list = self.measures.iloc[measure_id]['dependence']
		default_params = self.measures.iloc[measure_id]['default']
		fetched = {}
		
		if default_params is not None:
			fetched.update(default_params)
		
		if dep_list is None or len(dep_list) == 0:
			# não há dependências buscáveis
			return fetched
		
		for dep_name in dep_list:
			dep_mask = self.measures['name'].eq(dep_name)
			dep_isin = np.sum(dep_mask) > 0
			
			if not dep_isin:
				# impossível calcular a dependência
				continue
			
			dep_idx = dep_mask.argmax()
			var_name = self.measures.iloc[dep_idx]['varname']
			res_mask = (self.results.index == (network_id, dep_idx))
			res_isin = np.sum(res_mask) > 0
			res_val = None
			
			if res_isin:
				# recupera a dependência
				res_idx = res_mask.argmax()
				res_val = self.results.iloc[res_idx]['value']
			elif recursion_limit is None or recursion_limit > 0:
				# calcula a dependência
				if log_info:
					nname = self.networks.loc[network_id]['abbr']
					mname = self.measures.loc[dep_idx]['name']
					print(f' -> Calculating missing result (network: {nname}; measure: {mname})')
				rec_lim = None if recursion_limit is None else recursion_limit-1
				res = self.evaluate(
					network_subset=(self.networks.index==network_id),
					measure_subset=dep_mask,
					recursion_limit=rec_lim
				)
				self.update(r=res)
				res_val = res.iloc[0]['value']
			
			if res_val is not None:
				fetched[var_name] = res_val
		return fetched
	
	def evaluate(self, network_subset=None, measure_subset=None, recursion_limit=None, log_info=False, verbose:bool=False, callback=None, force=False):
		networks = self.networks if network_subset is None else self.networks[network_subset]
		measures = self.measures if measure_subset is None else self.measures[measure_subset]
		
		res_dict = { c: [] for c in ['network_id', 'measure_id', 'value', 'time'] }
		if log_info:
			pbar = tqdm(total=(len(networks)*len(measures)), ncols=80, position=0, leave=True)
			count = 0
		
		for i, n in networks.iterrows():
			g = n['graph']
			
			if not self.persist_fetched:
				pre_existing_filter = (self.results.index.get_level_values(0) == i)
				pre_existing_values = set(
					self.results.index[ pre_existing_filter ].get_level_values(1) if len(self.results) else []
				)
				
			if log_info:
				count += 1
				pbar.set_description_str(desc=f'Network {count:3d}/{len(networks)}', refresh=True)
			
			for j, m in measures.iterrows():
				if force or (i,j) not in self.results.index:
					dt = time.time()
					f = m['function']
					d = self.fetch_dependences(i, j, recursion_limit, log_info=verbose)
					v = f(g, **d)
					dt = time.time() - dt
				else:
					v = self.results.loc[(i,j), 'value']
					dt = self.results.loc[(i,j), 'time']
				
				res_dict['network_id'].append(i)
				res_dict['measure_id'].append(j)
				res_dict['value'].append(v)
				res_dict['time'].append(dt)
				
				if callback is not None:
					callback((i, j))
				if log_info:
					pbar.update()
					pbar.display()
			
			if not self.persist_fetched and len(self.results):
				all_existing_filter = (self.results.index.get_level_values(0) == i)
				all_existing_values = set(self.results.index[ all_existing_filter ].get_level_values(1))
				new_dependences = list(all_existing_values - pre_existing_values)
				rmv_filter = self.results.index.get_level_values(1).isin(new_dependences)
				if rmv_filter.sum():
					self.results.drop(index=self.results.index[rmv_filter], inplace=True)
		
		results = pd.DataFrame(res_dict)
		results.set_index(['network_id', 'measure_id'], inplace=True)
		
		if log_info:
			pbar.close()
		gc.collect()	
		return results
	
	def command_listener(self, recv, send):
		cmd, args = recv()
		while(cmd != 'return'):
			func = getattr(self, cmd)
			send(func(*args))
			cmd, args = recv()
		return 0
	


#====================================
class Report():
	
	def __init__(self):
		raise NotImplemented
	
	@classmethod
	def _validate_dtypes_(cls, arr, dtype=None):
		# direct check -> boolean response
		if dtype is not None:
			for item in arr:
				if not isinstance(item, dtype):
					return False
			return True
		# indirect check -> categorical response
		else:
			dcategs = {
				(int,float,bool): 'num',
				(list,tuple,np.ndarray): 'seq',
				(dict): 'map',
			}
			for dc in dcategs:
				if cls._validate_dtypes_(arr, dc):
					return dcategs[dc]
			return 'unk'
	
	@classmethod
	def _column_expansion_(cls, pivot_table, scope, dcateg, col_list):
		changes = {'add':[], 'upd':[], 'rmv':[]}
		# transform map keys into new columns
		if (scope in ['graph', 'vertex'] and dcateg=='map'):
			for c in col_list:
				aux_table = pd.DataFrame.from_records(pivot_table[c].values, index=pivot_table.index).add_prefix(c+':')
				pivot_table = pd.concat([pivot_table, aux_table], axis=1)
				del pivot_table[c]
				changes['add'] += list(aux_table.columns)
				changes['rmv'].append(c)
		# transform sequence indices into new columns
		elif (scope in ['graph', 'distrib'] and dcateg=='seq'):
			for c in col_list:
				aux_table = pd.DataFrame(pivot_table[c].tolist(), index=pivot_table.index)
				aux_table.columns = [ f'{c}#{idx+1:03d}' for idx in range(len(aux_table.columns)) ]
				if scope=='distrib':
					aux_table.fillna(0.0, inplace=True)
				pivot_table = pivot_table.join(aux_table)
				del pivot_table[c]
				changes['add'] += list(aux_table.columns)
				changes['rmv'].append(c)
		# transform sequence indices into new index lines
		elif (scope=='vertex' and dcateg=='seq'):
			qt_items = [ np.max([ len(pivot_table.loc[i,c]) for c in col_list ]) for i,_ in pivot_table.iterrows() ]
			pivot_table[scope] = [ np.arange(n) for n in qt_items ]
			pivot_table = pivot_table.explode(col_list + [scope]).set_index(scope, append=True)
			changes['upd'] += col_list
		return pivot_table, changes
	
	@classmethod
	def _join_(cls, networks, measures, results):
		# join tables of results, measures and networks
		pivot_table = results.join(
			measures, 
			on='measure_id',
			how='inner'
		).join(
			networks, 
			on='network_id',
			how='inner'
		)
		# pivot table of the specified content
		pivot_table = pivot_table.pivot( 
			index=(['dataset','abbr']), 
			columns=measures.columns[:1],
			values=results.columns[0]
		)
		# adjust columns and titles
		pivot_table = pivot_table[[ c for c in measures.values[:,0] if c in pivot_table.columns ]]
		pivot_table.index.name = 'networks'
		pivot_table.columns.name = 'measures'
		return pivot_table
	
	@classmethod
	def _expand_(cls, pivot_table, measures):
		# create a table to track column expansion
		header = measures.columns[0]
		measure_mask = (measures[header].isin(pivot_table.columns))
		column_track = measures.loc[measure_mask].copy()
		column_track['dcateg'] = [ cls._validate_dtypes_(pivot_table[m].values) for m in column_track[header].values ]
		# reshape column content into wider/longer format
		for scope in ['graph','distrib','vertex']:
			for dcateg in ['map', 'seq']:
				filter1 = (column_track['scope']==scope)
				filter2 = (column_track['dcateg']==dcateg)
				# expand columns
				selected_cols = list(column_track[(filter1 & filter2)][header].values)
				if not len(selected_cols):
					continue
				pivot_table, changes = cls._column_expansion_(pivot_table, scope, dcateg, selected_cols)
				# insert in column track all added columns
				for m in changes['add']:
					idx = column_track.index.max()
					column_track.loc[idx+1] = [m, scope, cls._validate_dtypes_(pivot_table[m].values) ]
				# refresh column track dcategs for all updated columns
				for m in changes['upd']:
					idx = column_track.index[(column_track[header]==m)][0]
					column_track.loc[idx,'dcateg'] = cls._validate_dtypes_(pivot_table[m].values)
				# delete from column track all removed columns
				rmv_mask = column_track[header].isin(changes['rmv'])
				column_track.drop(index=column_track.index[rmv_mask], inplace=True)
		return pivot_table
	
	@classmethod
	def _detail_(cls, pivot_table, networks, multilevel):
		mcols = list(pivot_table.columns)
		ncols = ['collection', 'dataset', 'label', 'network']
		mapping = {
			'abbr': 'network'
		}
		pivot_table = networks.join(pivot_table, on=['dataset','abbr'], how='right')
		pivot_table = pivot_table.rename(columns=mapping)[ ncols + mcols ]
		pivot_table['label'] = pivot_table['label'].astype('category')
		pivot_table.index.name = 'sample'
		if multilevel:
			pivot_table.columns = pd.MultiIndex.from_tuples([
				*list(itt.product(['Y'], ncols)),
				*list(itt.product(['X'], mcols)),
			])
		return pivot_table
	
	@classmethod
	def _split_(cls, pivot_table, partition):
		strat = [ f'{x}/{y}' for x,y in pivot_table['Y'][['dataset','label']].values ]
		# train/test split
		if isinstance(partition, [int, float]):
			idx_tr, idx_te = train_test_split(pivot_table.index, train_size=partition, stratify=strat)
			pivot_table.loc[idx_tr, 'split'] = 'train'
			pivot_table.loc[idx_te, 'split'] = 'test'
			pivot_table.set_index('split', append=True, inplace=True)
		# k-fold split
		elif isinstance(partition, [list, tuple]):
			#idx_tr, idx_te = train_test_split(pivot_table.index, train_size=train_size, stratify=strat)
			#pivot_table.loc[idx_tr, 'fold'] = 'train'
			#pivot_table.loc[idx_te, 'fold'] = 'test'
			#pivot_table.set_index('fold', append=True, inplace=True)
			raise
		pivot_table = pivot_table.swaplevel().sort_index()
		return pivot_table
	
	@classmethod
	def make(cls, datapool, network_subset=None, measure_subset=None, content='value', header='symbol', transform=None, expand=False, detail=False, split=None, multilevel=True):
		assert(content in ['value','time'])
		networks = datapool.networks if network_subset is None else datapool.networks.loc[network_subset]
		measures = datapool.measures if measure_subset is None else datapool.measures.loc[measure_subset]
		measures = measures[[header, 'scope']]
		results = datapool.results[[content]]
		pivot_table = cls._join_(networks, measures, results)
		if transform is not None:
			pivot_table = pivot_table.applymap(transform)
		if expand:
			pivot_table = cls._expand_(pivot_table, measures)
		if detail:
			pivot_table = cls._detail_(pivot_table, networks, multilevel)
		if split is not None:
			pivot_table = cls._split_(pivot_table, split)
		return pivot_table
	


#====================================
class ModelSelector():
	
	def __init__(self, classifier_list):
		self.classifiers = classifier_list
		self.evaluation = None
	
	@ignore_warnings(category=ConvergenceWarning)
	def fit(self, i, data):
		(X_tr, y_tr) = data
		ti = time.time()
		self.classifiers[i].fit(X_tr, y_tr)
		tf = time.time()
		return tf - ti
	
	def evaluate(self, i, data, fit_time=None):
		(X_te, y_te) = data
		clf = self.classifiers[i]
		ti = time.time()
		y_pr = clf.predict(X_te)
		tf = time.time()
		lbl = list(set(y_pr))
		clf_name = type(clf).__name__
		return {
			# model info
			'classifier': clf,
			'model name': clf_name,
			'model params': str(clf)[len(clf_name)+1:-1], 
			# experiment info
			'fold': None, 
			'repetition': None,
			'fit time (s)': fit_time,
			'prediction time (ms/sample)': 1e3 * (tf - ti) / len(y_te),
			# evaluation info
			'accuracy': accuracy_score(y_te, y_pr),
			'balanced accuracy': balanced_accuracy_score(y_te, y_pr),
			'precision (micro)': precision_score(y_te, y_pr, average='micro', labels=lbl),
			'precision (macro)': precision_score(y_te, y_pr, average='macro', labels=lbl),
			'precision (weighted)': precision_score(y_te, y_pr, average='weighted', labels=lbl),
			'recall (micro)': recall_score(y_te, y_pr, average='micro', labels=lbl),
			'recall (macro)': recall_score(y_te, y_pr, average='macro', labels=lbl),
			'recall (weighted)': recall_score(y_te, y_pr, average='weighted', labels=lbl),
			'f1 (micro)': f1_score(y_te, y_pr, average='micro', labels=lbl),
			'f1 (macro)': f1_score(y_te, y_pr, average='macro', labels=lbl),
			'f1 (weighted)': f1_score(y_te, y_pr, average='weighted', labels=lbl),
			'confusion matrix': confusion_matrix(y_te, y_pr)
		}
	
	def split_datasets(self, data, folds:int=1, reps:int=1, astype=iter):
		kfolds = RepeatedStratifiedKFold(n_splits=folds, n_repeats=reps)
		if isinstance(data, dict):
			indices = { name: astype(kfolds.split(X, y)) for name, (X, y) in data.items() }
		elif isinstance(data, tuple):
			indices = astype(kfolds.split(*data))
		return indices
	
	def run_experiment(self, identifier:str, datasets:dict, folds:int=1, reps:int=1, indices:dict=None, 
						exclude:list=[], verbose:bool=True):
		old_setting = np.seterr(divide='ignore', over='ignore')
		to_num = lambda X: np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
		if indices is None:
			indices = self.split_dataset(datasets, folds, reps)
		if verbose:
			pbar = tqdm(total=len(self.classifiers)*len(datasets)*reps, ncols=80, position=0, leave=True)
		for dsname in datasets:
			(X, y) = datasets[dsname]
			results = []
			for k, (tr_idx, te_idx) in enumerate(indices[dsname]):
				scaler = StandardScaler()
				train_data = scaler.fit_transform(X[tr_idx]), y[tr_idx]
				test_data  = to_num(scaler.transform(X[te_idx])), y[te_idx]
				for i in range(len(self.classifiers)):
					self.reset_estimators(i)
					res = self.evaluate(i, test_data, self.fit(i, train_data))
					res['repetition'] = (k//folds)+1
					res['fold'] = (k%folds)+1
					for col in exclude:
						del res[col]
					results.append(res)
					if verbose and res['fold']==folds:
						pbar.update()
			df = pd.DataFrame(results)
			df.insert(0, 'experiment', identifier)
			df.insert(1, 'dataset', dsname)
			self.evaluation = (df if self.evaluation is None else pd.concat([self.evaluation, df], ignore_index=True))
			for col in ['experiment', 'dataset', 'model name', 'model params']:
				self.evaluation[col] = self.evaluation[col].astype('category')
		np.seterr(**old_setting)
	
	def analyze(self, dataset, by, top=None, level=0):
		if not isinstance(by, list):
			by = [by]
		identifiers = ['experiment', 'dataset', 'model name', 'model params']
		reductions = ['mean', 'std', 'max', 'min'][:level+1]
		exp_filter = (self.evaluation['dataset'] == dataset)
		df = self.evaluation[exp_filter][identifiers + by]
		df = df.groupby(identifiers, observed=True).aggregate(reductions)[by]
		df.sort_values(list(itt.product(by, ['mean'])), kind='stable', ascending=False, inplace=True)
		df.reset_index(inplace=True)
		df.index = df.index + 1
		if level == 0:
			df.columns = df.columns.droplevel(1)
		return df if top is None else df.head(top)
	
	def compare(self, by, sorting_score=True, top=None):
		df = pd.pivot_table(
			pd.concat([ 
				self.analyze(dataset, by, level=0) 
				for dataset in self.evaluation['dataset'].unique() 
			], ignore_index=True),  
			values=[by],
			index=['experiment', 'model name', 'model params'],
			columns=['dataset'],
			observed=True,
			sort=False
		)
		df.columns = df.columns.droplevel(0)
		if sorting_score:
			df.insert(0, 'global score', df.mean(axis=1).values)
			df.sort_values(by='global score', kind='stable', ascending=False, inplace=True)
			df.reset_index(inplace=True)
			df.columns = pd.MultiIndex.from_tuples([
				*list(itt.product(['model info'], ['experiment', 'classifier', 'params', 'global score'])),
				*list(itt.product([by], self.evaluation['dataset'].unique())),
			])
			df.index = df.index + 1
		else:
			df.sort_values(axis=0, by=['experiment', 'model name', 'model params'], inplace=True)
		df.columns.name=''
		return df if top is None else df.head(top)
	
	def reset_estimators(self, i=None):
		if i is None:
			self.classifiers = skl.base.clone(self.classifiers)
		else:
			self.classifiers[i] = skl.base.clone(self.classifiers[i])

