import numpy as np
# import torch
import copy
from sklearn.metrics import r2_score
import keras.backend as K


def RelE_Lp(ref, approx, order=2.):
	assert(ref.shape == approx.shape)
	ref = ref.reshape(-1)
	approx = approx.reshape(-1)
	
	return np.sum(abs(ref - approx)**order)/np.sum(abs(ref)**order)


def RelE_pw(ref,approx, order=2.):
	assert(ref.shape == approx.shape)
	ref = ref.reshape(-1)
	approx = approx.reshape(-1)

	return abs(ref - approx)**order/abs(ref)**order

def RelE_ratio(RelE, threshold=0.1):
	n_threshold = len(RelE[RelE < threshold])
	n_sample = len(RelE)
	
	r = n_threshold / n_sample
	# r_format = '{:2.1E}'
	r_format = '{:1.5f}'
	# # if r < 0.01: r_format = '{:2.1E}'
	# # else: r_format = '{:2.1f}'
	return r, r_format, n_threshold, n_sample

def performance_tags (L, P):
	assert(L.shape == P.shape)
	assert(np.prod(L.shape) == np.prod(L.shape[:2])) ##the matrix should be 2d matrix

	tag = {}
	tag['R2'] = 'R2 = {:2.8f}'.format(r2_score(L.reshape(-1), P.reshape(-1)))

	RelE = {}
	RelE['pw(order=1)'] = RelE_pw(L, P, order=1.)
	RelE['pw(order=2)'] = RelE_pw(L, P, order=2.)
	RelE['L1'] = RelE_Lp(L, P, order=1.)
	RelE['L2'] = RelE_Lp(L, P, order=2.)
	
	for case in list(RelE.keys()):
		OUT = RelE[case]

		if 'pw' in case: 
			ratio, ratio_format, n_threshold, n_sample = RelE_ratio(OUT)
			tag[case] = 'RelE [{:s}] < 0.1 ({:d}/{:d} = {:s})'.format(case, n_threshold, n_sample, str(ratio_format).format(ratio))
		else:
			tag[case] = 'RelE [{:s}] = {:E}'.format(case, OUT)

	return tag, RelE

def performance_tags_grid (L, P):
	assert(L.shape == P.shape)
	assert(np.prod(L.shape) == np.prod(L.shape[:2])) ##the matrix should be 2d matrix

	tag = {}
	tag['R2'] = 'R2 = {:2.8f}'.format(r2_score(L, P))

	n_grid = L.shape[1]

	RelE = {}
	RelE['L1'] = np.zeros(n_grid)
	RelE['L2'] = np.zeros(n_grid)
	RelE['pw(order=1)'] = np.zeros(L.shape)
	RelE['pw(order=2)'] = np.zeros(L.shape)

	for gi in range(n_grid):
		LL = L[:, gi]
		PP = P[:, gi]

		RelE['L1'][gi] = RelE_Lp(LL, PP, order=1.)
		RelE['L2'][gi] = RelE_Lp(LL, PP, order=2.)

		RelE['pw(order=1)'][:,gi] = RelE_pw(LL, PP, order=1.)
		RelE['pw(order=2)'][:,gi] = RelE_pw(LL, PP, order=2.)
	
	for case in list(RelE.keys()):
		if 'pw' in case: 
			OUT = RelE[case].reshape(-1)

			ratio, ratio_format, n_threshold, n_sample = RelE_ratio(OUT)
			tag[case] = 'RelE [{:s}] < 0.1 ({:d}/{:d} = {:s})'.format(case, n_threshold, n_sample, str(ratio_format).format(ratio))
	###	else:
	### 		tag[case] = 'RelE [{:s}] = {:E}'.format(case, OUT)

	return tag, RelE


# def custom_mse(y_true, y_pred):
# 	# calculating squared difference between target and predicted values 
# 	loss = K.square(y_pred - y_true)  # (batch_size, 2)
# 	# multiplying the values with weights along batch dimension
# 	loss = loss * [0.3, 0.7]          # (batch_size, 2)
# 	# summing both loss values along batch dimension 
# 	loss = K.sum(loss, axis=1)        # (batch_size,)
# 	return loss

def custom_loss_mse(y_true, y_pred):
	loss = K.mean(K.square(y_pred - y_true))
	return loss

def custom_loss_RelE(y_true, y_pred):
	loss = K.sum(K.square(y_pred - y_true))/K.sum(K.square(y_true)+1e-30)
	return loss
	
def custom_loss_mse(y_true, y_pred):
	loss = K.square(y_pred - y_true)  # (batch_size, 2)
	loss = K.sum(loss, axis=1)        # (batch_size,)
	return loss

def Data_Reshape (data):
	data = copy.deepcopy(data)
	if np.prod(data.shape) == data.shape[0]:
		data = data.reshape(-1,1)
	return data

# def Get_Label_Pred_torch (model, Input, Output=[]):
# 	IN = torch.from_numpy(Input).float()
# 	Label = Output
# 	Pred = model(IN).data.numpy()
# 	return IN, Label, Pred

def Normalize_ft(data, m_prev, M_prev, m=0., M=1.):
	return (M - m)/(M_prev - m_prev)*(data - m_prev) + m ##normalization into [m, M]

def Inv_Normalize_ft(data, m_prev, M_prev, m=0., M=1.):
	return (M_prev - m_prev)/(M - m)*(data - m) + m_prev ##re-scale date from [m,M] to the previous one [m_prev, M_prev]


def Normalization(DATA_orig, minmax=[]):
	DATA = copy.deepcopy(DATA_orig)

	dim = DATA.shape[1]
	
	Min, Max, Min_prev, Max_prev = [], [], [], []

	for d in range(dim):
		data = DATA[:,d]

		m_prev = np.min(data) ##min of previous data
		M_prev = np.max(data) ##max of previous data

		if len(minmax) == 0:
			m, M = 0., 1.
		else:
			m, M = minmax 
		
		# print("m = {:f}, M = {:f}".format(m, M))
		if m_prev != M_prev:
			DATA[:,d] = Normalize_ft(data, m_prev, M_prev, m, M)
		else:
			m, M = 0.5, 0.5
			DATA[:,d] = 0.5*np.ones(DATA[:,d].shape)
 	
		Min.append(m)
		Max.append(M)
		Min_prev.append(m_prev)
		Max_prev.append(M_prev)

	Min = np.array(Min)
	Max = np.array(Max)
	Min_prev = np.array(Min_prev)
	Max_prev = np.array(Max_prev)
	
	return Min_prev, Max_prev, Min, Max, DATA


def Inverse_Normalization (data, min_max):
	m, M = min_max
	return m + (M - m)*data
