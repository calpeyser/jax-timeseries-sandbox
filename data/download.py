import time
from datetime import datetime, timedelta
import random

import cbpro
import pandas as pd
from tqdm import trange

RATE_LIMIT_SIGNAL = 'RATE_LIMIT_SIGNAL'
OUT_FILE = 'raw/dataset'
EVAL_FILE = 'raw/eval_dataset'

c = cbpro.PublicClient()
now = datetime.fromtimestamp(time.time())
delta = timedelta(minutes=30)


def get_data(start, end):
	api_result = c.get_product_historic_rates(
		product_id='BTC-USD',
		start=start.isoformat(),
		end=end.isoformat(),
		granularity=60)
	try:
		historical = pd.DataFrame(api_result)
	except Exception as e:
		if isinstance(e, ValueError):
			return RATE_LIMIT_SIGNAL
		else:
			raise e
	historical.columns= ["Date","Open","High","Low","Close","Volume"]
	historical['Date'] = pd.to_datetime(historical['Date'], unit='s')
	historical.set_index('Date', inplace=True)
	return historical


start = now - delta
end = now
dataset_fragments = []
eval_fragments = []

for _ in trange(100):
	next_fragment = get_data(start, end)
	while(not isinstance(next_fragment, pd.DataFrame) and next_fragment == RATE_LIMIT_SIGNAL):
		time.sleep(1)
		next_fragment = get_data(start, end)

	if random.random() < 0.9:
		dataset_fragments.append(next_fragment)
	else:
		eval_fragments.append(next_fragment)
	end = start
	start = start - delta

dataset = pd.concat(dataset_fragments)
dataset.sort_values(by='Date', ascending=True, inplace=True)
print(dataset)
dataset.to_pickle(OUT_FILE)

eval_dataset = pd.concat(eval_fragments)
eval_dataset.sort_values(by='Date', ascending=True, inplace=True)
eval_dataset.to_pickle(EVAL_FILE)
