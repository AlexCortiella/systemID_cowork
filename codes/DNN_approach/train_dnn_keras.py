import numpy as np
import os, sys
sys.path.append('./util/')
from DNN_basic_utils import *
import copy
import pickle

from sklearn.metrics import r2_score
import keras.backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LeakyReLU, PReLU
from keras import losses, initializers, optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint

def dnn_train_keras(DATA, Archi, Opt):
	DATA = copy.deepcopy(DATA)

	dim = {}
	for io in ['in', 'out']:
		DATA[io] = Data_Reshape(DATA[io])
		dim[io] = DATA[io].shape[-1]

	#########################################################################################################################
	##### Normalize data 
	#########################################################################################################################
	Min_prev, Max_prev = {},{}
	if Opt.want_normalize == 'yes':
		for io in ['in', 'out']:
			Min_prev[io], Max_prev[io], _, _, DATA[io] = Normalization(DATA[io], [0., 1.])
	else:
		for io in ['in', 'out']:
			Min_prev[io], Max_prev[io], _, _, _ = Normalization(DATA[io], [0., 1.])

	#########################################################################################################################
	##### Shuffle & Define 'Valid' set 
	#########################################################################################################################
	N, shuffle = {},{}
	N['all'] = DATA['in'].shape[0]

	r = Opt.train_rate

	Train, Valid = {},{}
	if r == 1.:
		Train = copy.deepcopy(DATA)
		Valid = copy.deepcopy(Train) ##Valid == Train
		N['train'] = N['all']
		N['valid'] = Valid['in'].shape[0]
	else:
		assert(r < 1.)
		N['train'] = int(N['all']*r)
		N['valid'] = N['all'] - N['train']

	# np.random.seed(100)
	shuffle['all'] = np.random.permutation(N['all'])
	shuffle['train'] = shuffle['all'][:N['train']]
	shuffle['valid'] = shuffle['all'][N['train']:]

	for io in ['in', 'out']:
		Train[io] = DATA[io][shuffle['train'],:]
		if r < 1.:
			Valid[io] = DATA[io][shuffle['valid'],:]

	#########################################################################################################################
	##### Train DNN model with Keras
	#########################################################################################################################
	# model_name = Folder.dnn + current_output(oj) + '.h5'

	model_name = Archi.model_folder + Archi.model_name
	HiddenLayer, Nodes, BatchSize, Epoch, Acti_ft = Archi.HiddenLayer, Archi.Nodes, Archi.BatchSize, Archi.Epoch, Archi.Acti_ft
	lr = Archi.lr
	if hasattr(Opt, 'want_bias') == False: Opt.want_bias = True ##default want_bias is True

	###############################################
	model = Sequential()

	if HiddenLayer == 0:
		model.add(Dense(dim['out'], input_shape=(dim['in'],), use_bias=Opt.want_bias))
	else:
		model.add(Dense(Nodes, input_shape=(dim['in'],), use_bias=Opt.want_bias))
		# model.add(Dense(Nodes, input_shape=(dim['in'],), \
		# 	kernel_initializer=initializers.glorot_uniform(seed=0)))
		# # model.add(Dense(Nodes, input_shape=(Dim['x'],)), kernel_initializer='uniform'))
		###############################################
		nL = HiddenLayer
		if not Acti_ft == 'identity':
			model.add(Activation(Acti_ft))
			while nL > 1:
				model.add(Dense(Nodes))
				model.add(Activation(Acti_ft))
				nL -= 1
		else:
			assert(Acti_ft == 'identity')
			while nL > 1:
				model.add(Dense(Nodes))
				nL -= 1

		if Opt.want_normalize == 'yes':
			model.add(Dense(dim['out'], activation='sigmoid')) ###To make the prediction nonnegative
		else:
			model.add(Dense(dim['out']))
	
	##############################################################################################
	model.compile(loss='mse', optimizer=optimizers.Adam(lr=lr), metrics=['accuracy'])

	es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
	mc = ModelCheckpoint(model_name, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

	history = model.fit(Train['in'], Train['out'], validation_data=(Valid['in'], Valid['out']),\
						epochs=Epoch, batch_size=BatchSize, verbose=Opt.Verbose, callbacks=[es, mc])

	model = load_model(model_name) ### call the best model saved
	with open(model_name.replace('.h5', '.pkl'), 'wb') as f:  # Python 3: open(..., 'wb')
		pickle.dump([Min_prev, Max_prev], f)

	return model, Min_prev, Max_prev



