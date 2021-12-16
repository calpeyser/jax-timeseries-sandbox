import pandas as pd
from tqdm import trange
import btalib
import time
from datetime import datetime, timedelta

RAW_DATASET_PATH = 'raw/dataset'
RAW_EVAL_PATH = 'raw/eval_dataset'

OUT_FILE = 'features/dataset'
EVAL_FILE = 'features/eval_dataset'

def make_feature(f, output_name, new_name):
	def feature_fn(dataset):
		res = f(dataset)
		res = res[res[output_name].notnull()]
		res = res.rename(columns={output_name: new_name})
		return res
	return feature_fn

def value_after_minutes(dataset, num_minutes):
	dataset_copy = dataset.copy(deep=True)
	dataset_copy['DateColumn'] = dataset_copy.index - timedelta(minutes=num_minutes)
	dataset_copy.set_index('DateColumn', inplace=True)

	dataset['AfterMinutes'] = dataset_copy['Close']

	return dataset


METRICS = []
for i in range(120)[1::10]:
	METRICS.append(make_feature(lambda d: btalib.sma(d, period=i).df, 'sma', 'sma_' + str(i) + 'm'))

def _sma_diff(i, j):
	def res(d):
		sma_i = btalib.sma(d, period=i).df
		sma_j = btalib.sma(d, period=j).df
		return sma_i - sma_j
	return res

# for i in range(120)[1::10]:
# 	for j in range(120)[1::10]:
# 		METRICS.append(make_feature(_sma_diff(i, j), 'sma', 'sma_' + str(i) + '_' + str(j) + 'm'))

METRICS.extend([
	make_feature(lambda d: btalib.sma(d, period=1).df, 'sma', 'sma_1m'),
	make_feature(lambda d: btalib.sma(d, period=2).df, 'sma', 'sma_2m'),
	make_feature(lambda d: btalib.sma(d, period=3).df, 'sma', 'sma_3m'),
	make_feature(lambda d: btalib.sma(d, period=4).df, 'sma', 'sma_4m'),
	make_feature(lambda d: btalib.sma(d, period=5).df, 'sma', 'sma_5m'),
	make_feature(lambda d: btalib.sma(d, period=10).df, 'sma', 'sma_10m'),
	make_feature(lambda d: btalib.sma(d, period=20).df, 'sma', 'sma_20m'),
	make_feature(lambda d: btalib.sma(d, period=30).df, 'sma', 'sma_30m'),
	make_feature(lambda d: btalib.sma(d, period=60).df, 'sma', 'sma_1h'),
	make_feature(lambda d: btalib.sma(d, period=3 * 60).df, 'sma', 'sma_3h'),
	make_feature(lambda d: btalib.sma(d, period=6 * 60).df, 'sma', 'sma_6h'),
	make_feature(lambda d: value_after_minutes(d, 1), 'AfterMinutes', 'after_1m'),
	# make_feature(lambda d: value_after_minutes(d, 30), 'AfterMinutes', 'after_30m'),
	# make_feature(lambda d: value_after_minutes(d, 60 * 5), 'AfterMinutes', 'after_5h'),
	# make_feature(lambda d: value_after_minutes(d, 60 * 24), 'AfterMinutes', 'after_1d'),
])

def process_dataset(in_path, out_path):
	dataset = pd.read_pickle(in_path)

	features = []
	for metric in METRICS:
		features.append(metric(dataset))

	for feature in features:
		dataset = dataset.merge(feature, on='Date')

	dataset = dataset.drop(['AfterMinutes'], axis=1)
	for col in dataset.columns:
		if '_y' in col:
			dataset = dataset.drop([col], axis=1) 
		else:
			print(col)
	dataset.to_pickle(out_path, protocol=4)

process_dataset(RAW_DATASET_PATH, OUT_FILE)
process_dataset(RAW_EVAL_PATH, EVAL_FILE)
