import pandas as pd
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np

DATASET_PATH = '../data/features/dataset'
EVAL_DATASET_PATH = '../data/features/eval_dataset'
CONFIG = {
	'learning_rate': 0.001,
	'batch_size': 4,
	'num_epochs': 25,
}

class FFRegressionModel(nn.Module):

	@nn.compact
	def __call__(self, x):
		for _ in range(5):
			x = nn.Dense(features=32)(x)
			x = nn.relu(x)
		x = nn.Dense(features=32)(x)
		x = nn.Dense(features=1)(x)
		return jnp.squeeze(x)

@jax.jit
def apply_model(state, features, labels):
	def loss_fn(params):
		prediction = FFRegressionModel().apply({'params': params}, features)
		loss = jnp.abs(prediction - labels)
		loss = jnp.mean(loss)
		return loss, prediction

	grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
	(loss, prediction), grads = grad_fn(state.params)
	return grads, loss, prediction

@jax.jit
def update_model(state, grads):
	return state.apply_gradients(grads=grads) 

def train_epoch(state, train_ds, eval_ds, batch_size, rng):
	train_ds_size = len(train_ds)
	steps_per_epoch = train_ds_size // batch_size

	perms = jax.random.permutation(rng, len(train_ds))
	perms = perms[:steps_per_epoch * batch_size]
	perms = perms.reshape((steps_per_epoch, batch_size))

	epoch_loss = []

	for perm in perms:
		batch_features = train_ds[perm, :-1]
		batch_labels = train_ds[perm, -1]

		grads, loss, _ = apply_model(state, batch_features, batch_labels)
		state = update_model(state, grads)
		epoch_loss.append(loss)
	train_loss = np.mean(epoch_loss)

	eval_ds_size = len(eval_dataset)
	eval_steps_per_epoch = eval_ds_size // batch_size
	eval_perms = jnp.arange(eval_ds_size)[:eval_steps_per_epoch * batch_size].reshape((eval_steps_per_epoch, batch_size))

	eval_epoch_loss = []
	directional_predictions = 0.0

	for perm in eval_perms:
		batch_features = eval_ds[perm, :-1]
		batch_labels = eval_ds[perm, -1]
		_, loss, prediction = apply_model(state, batch_features, batch_labels)
		eval_epoch_loss.append(loss)

		up_preds = np.greater(prediction - batch_features[:, 3], 0)
		up_gt = np.greater(batch_labels - batch_features[:, 3], 0)
		agreement = np.equal(up_preds, up_gt)
		directional_predictions += np.sum(agreement.astype(int))

	eval_loss = np.mean(eval_epoch_loss)
	accuracy = directional_predictions / (batch_size * len(eval_perms))

	return state, train_loss, eval_loss, accuracy

def load_dataset():
	dataset = pd.read_pickle(DATASET_PATH)
	dataset = dataset.to_numpy()

	eval_dataset = pd.read_pickle(EVAL_DATASET_PATH)
	eval_dataset = eval_dataset.to_numpy()

	return dataset, eval_dataset

def create_train_state(rng, config):
	model = FFRegressionModel()
	params = model.init(rng, 5000 * jnp.ones([1, 30]))['params']
	tx = optax.adam(config['learning_rate'])
	return train_state.TrainState.create(
		apply_fn=model.apply, params=params, tx=tx)

rng = jax.random.PRNGKey(0)
dataset, eval_dataset = load_dataset()

rng, init_rng = jax.random.split(rng)
state = create_train_state(init_rng, CONFIG)

for epoch in range(1, CONFIG['num_epochs'] + 1):
	rng, input_rng = jax.random.split(rng)
	state, train_loss, eval_loss, accuracy = train_epoch(state, dataset, eval_dataset, CONFIG['batch_size'], input_rng)

	print(
		'epoch:% 3d, train_loss: %.4f, eval_loss: %.4f, accuracy: %.4f'
		% (epoch, train_loss, eval_loss, accuracy))







