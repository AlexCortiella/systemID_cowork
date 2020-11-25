import numpy as np
# import numpy.linalg as LA
# import pandas as pd
import matplotlib.pylab as plt
from matplotlib import rc
rc('font', **{'family':'Times', 'size':13})
# rc('text', usetex=True)
rc('savefig', **{'transparent':False})
from colorama import init, Fore, Back, Style
init(autoreset=True)


import os, sys
sys.path.append('./util/')
from DNN_basic_utils import *

from importlib import reload

import math
import pickle
from sklearn.metrics import r2_score
import keras.backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LeakyReLU, PReLU
from keras import losses, initializers, optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint

# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter

from scipy.integrate import odeint
# from scipy.linalg import lstsq

# ####### SPGL1 package #######
# from spgl1 import spg_bpdn
# from spgl1.lsqr import lsqr
# from spgl1 import spgl1, spg_lasso, spg_bp, spg_bpdn, spg_mmv
# from spgl1.spgl1 import norm_l1nn_primal, norm_l1nn_dual, norm_l1nn_project
# from spgl1.spgl1 import norm_l12nn_primal, norm_l12nn_dual, norm_l12nn_project


#####################################################################################
from sklearn.preprocessing import PolynomialFeatures

def multi_index(poly_order, dim):
	Poly = PolynomialFeatures(degree=poly_order)
	x = np.zeros((2, dim))
	Poly.fit_transform(x)

	index = Poly.powers_
	P = index.shape[0]
	return index, P


def Phi_ft(x, poly_order):
	x = np.array(x)
	nx = x.shape[0]
	dim = x.shape[1]

	Alpha, P = multi_index(poly_order, dim) ## multi-index 

	phi_mat = np.zeros((P, nx))
	for pi in range(P):
		val = np.array([x[:,j]**Alpha[pi][j] for j in range(dim)])
		phi_mat[pi] = np.prod(val, axis=0)
	
	phi_mat = phi_mat.T

	return phi_mat

#####################################################################################
class Obj():
	def __init__(self):
		return

class FOLDER():
	def __init__(self):
		self.data = './data/duffling/'
		self.dnn = './dnn/'

	def MakeDir(self):
		for k in self.__dict__.keys():
			myfile = self.__dict__[k]
			if isinstance(myfile, str) == True:
				if not os.path.exists(myfile):
					os.makedirs(myfile)

Folder = FOLDER()
Folder.MakeDir()

#####################################################################################
##### Dynamic systems
#####################################################################################
# def duffing(x, t, gamma=0.1, kappa=1, epsilon=5):
def duffing(x, t, gamma, kappa, epsilon):
	### Compute dynamics
	dsdt = [x[1], -gamma*x[1] - kappa*x[0] - epsilon*x[0]**3.]
	return dsdt

#####################################################################################
##### Parameters 
#####################################################################################
gamma = 0.1
kappa = 1.
epsilon = 5.
poly_order = 3  ##1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3

x_initial = [0., 1.]
noise_scale = 0.01 ## constant to adjust the error scale 

t0, tf = 0, 10  # start and end
uniform_dt = 'yes'

#####################################################################################
##### Make true & noisy data using a dynamic model 
#####################################################################################
DATA = {}

##### Time instants 
if uniform_dt == 'yes':
	dt = 0.001  # time step
	Nt = int(np.floor(tf-t0)/dt + 1) #Number of time instances
	t_span = np.linspace(t0, tf, Nt)
else:
	Nt = 1000 ##the number of time instants
	t_span = np.sort(np.random.uniform(t0, tf, Nt)) ##Choose random time instants 

dt = np.diff(t_span) ##time step 
DATA['t'] = t_span
nsample = DATA['t'].shape[0]

##### Initial Conditions/Values
DATA['x:init'] =np.array(x_initial)
n_var = len(DATA['x:init'])

##### True data from a dynamic model
DATA['x:true'] = odeint(lambda x, t:duffing(x, t, gamma, kappa, epsilon), DATA['x:init'], t_span)
assert(DATA['x:true'].shape[1] == n_var)

##### True dx solutions in time 
DATA['dx:true'] = np.zeros(DATA['x:true'].shape)
for i in range(nsample):
	DATA['dx:true'][i] = duffing(DATA['x:true'][i], t_span[i], gamma, kappa, epsilon)
assert(np.sum(abs(DATA['x:true'][:,1] - DATA['dx:true'][:,0])) == 0.)

##### Corrupt states by adding noise --> Observation model x_noise(t) = x_true(t) + e(t)
Noise = {}
Noise['true'] = noise_scale*np.random.randn(Nt, n_var) ## Additive zero-mean white noise (Assumed Gaussian)
ti = 0
Noise['true'][ti] = 0.  ## Assume there is no noise at the initial time ti=0

DATA['x:noisy'] = DATA['x:true'] + Noise['true'] ## Add noise to the true data 


#####################################################################################
##### Train DNN model for Xtrue
#####################################################################################
import train_dnn_keras
reload(train_dnn_keras)
from train_dnn_keras import *

Ar = {k:Obj() for k in ['Xtrue', 'Coef']}
Model = {k:0. for k in ['Xtrue', 'Coef']}
Train = {k:{} for k in ['Xtrue', 'Coef']}
Min_prev, Max_prev = {k:0. for k in ['Xtrue','Coef']},  {k:0. for k in ['Xtrue','Coef']}
Nsample = {}
Opt = Obj()


mc = 'Xtrue' ###model_case

Train[mc]['in'] = DATA['t']
Train[mc]['out'] = DATA['x:noisy']
Nsample[mc] = Train[mc]['in'].shape

hl = 4
node = 16
batch = 100
epoch = 200
acti = 'tanh'
lr = 1e-3
Ar[mc].HiddenLayer, Ar[mc].Nodes, Ar[mc].BatchSize, Ar[mc].Epoch, Ar[mc].Acti_ft, Ar[mc].lr = hl, node, batch, epoch, acti, lr
Ar[mc].model_name = mc + '_hl{:d}_node{:d}_batch{:d}_ep{:d}_{:s}.h5'.format(hl, node, batch, epoch, acti)
Ar[mc].model_folder = Folder.dnn

Opt.train_rate = 0.8
Opt.want_normalize = 'no'
Opt.Verbose = True

# Model[mc], Min_prev[mc], Max_prev[mc] = dnn_train_keras(Train[mc], Ar[mc], Opt)

Model[mc] = load_model(Ar[mc].model_folder + Ar[mc].model_name)
with open(Ar[mc].model_folder + Ar[mc].model_name.replace('.h5', '.pkl'), 'rb') as f:
	[Min_prev[mc], Max_prev[mc]] = pickle.load(f)

# IN, IN_normalized, Label, Pred = {},{},{},{}
# IN[mc] = Train['in']
# Label[mc] = Train['out']
# if Opt.want_normalize == 'yes':
# 	IN_normalized[mc] = Normalize_ft(IN[mc], Min_prev[mc]['in'], Max_prev[mc]['in'])
# 	Pred[mc] = Model[mc].predict(IN_normalized[mc])
# 	Pred[mc] = Inv_Normalize_ft(Pred[mc], Min_prev[mc]['out'], Max_prev[mc]['out'])
# else:
# 	Pred[mc] = Model[mc].predict(IN[mc])

#####################################################################################
##### Int(phi(x)) & [X(b) - X(a)] matrices 
# #####################################################################################
mc = 'Coef'
pickle_Phi = './data/' + 'Int_Phi.pkl'

Nsample[mc] = 1000 ## Nsample for int(phi) implies the number of training samples 

ab_sample = np.hstack([np.random.uniform(t0, tf, (Nsample[mc],1)), np.random.uniform(t0, tf, (Nsample[mc],1))])

Xb_Xa = np.zeros((Nsample[mc], n_var))
Int_Phi = [0.]*Nsample[mc]

for s in range(Nsample[mc]):
	##### Get samples for (a, b) pair = integration boundary 
	int_boundary = ab_sample[s]
	a, b = np.min(int_boundary), np.max(int_boundary)
	
	diff_ab_tol = (tf - t0)/20.
	if abs(a - b) < diff_ab_tol:
		a = np.random.uniform(t0, (t0+tf)/2.-diff_ab_tol/2.)
		b = np.random.uniform((t0+tf)/2.+diff_ab_tol/2., tf)
		ab_sample[s] = np.array([a, b])

	##### Xb_Xa := [X(b) - X(a)] matrix
	if Opt.want_normalize == 'no':
		Xb_Xa[s] = Model['Xtrue'].predict([b]) - Model['Xtrue'].predict([a])

	##### Calculate Int_Phi := Int_a^b phi(x(t)) dt 
	ts = np.arange(a, b, 1e-3)
	ts[-1] = b
	ts_mid = (ts[1:] + ts[:-1])/2.

	x_val = Model['Xtrue'].predict(ts_mid)
	phi_mat = Phi_ft(x_val, poly_order)

	val = phi_mat.T*np.diff(ts)  ### val.T = phi_mat*np.tile(np.diff(ts).reshape(-1,1), [1, P])
	Int_Phi[s] = np.sum(val.T, axis=0)

	print("Int_Phi calculated for #sample = {:d}/{:d}".format(s, Nsample[mc]))
Int_Phi = np.array(Int_Phi)

with open(pickle_Phi, 'wb') as f:
	pickle.dump([ab_sample, Xb_Xa, Int_Phi], f)

with open(pickle_Phi, 'rb') as f:
	[ab_sample, Xb_Xa, Int_Phi] = pickle.load(f)
Nsample[mc] = ab_sample.shape[0]

#####################################################################################
##### Train 
#####################################################################################
import train_dnn_keras
reload(train_dnn_keras)
from train_dnn_keras import *

Train[mc]['in'] = Int_Phi
Train[mc]['out'] = Xb_Xa

hl = 0
node = 0
batch = 10
epoch = 300
acti = 'tanh'
lr = 1e-3
Ar[mc].HiddenLayer, Ar[mc].Nodes, Ar[mc].BatchSize, Ar[mc].Epoch, Ar[mc].Acti_ft, Ar[mc].lr = hl, node, batch, epoch, acti, lr
Ar[mc].model_name = mc + '_hl{:d}_node{:d}_batch{:d}_ep{:d}_{:s}.h5'.format(hl, node, batch, epoch, acti)
Ar[mc].model_folder = Folder.dnn

Opt.want_bias = False
Opt.train_rate = 0.8
Opt.want_normalize = 'no'
Opt.Verbose = True

# Model[mc], Min_prev[mc], Max_prev[mc] = dnn_train_keras(Train[mc], Ar[mc], Opt)

Model[mc] = load_model(Ar[mc].model_folder + Ar[mc].model_name)
with open(Ar[mc].model_folder + Ar[mc].model_name.replace('.h5', '.pkl'), 'rb') as f:
	[Min_prev[mc], Max_prev[mc]] = pickle.load(f)


#####################################################################################
##### Performance of Coef model
#####################################################################################
IN = Train[mc]['in']
Label = Train[mc]['out']
Pred = Model[mc].predict(IN)
print(r2_score(Label[:,0], Pred[:,0]))
print(r2_score(Label[:,1], Pred[:,1]))


P = Int_Phi.shape[1]
Coef = {}
Coef['true'] = np.zeros((P, n_var))
Coef['true'][2,0] = 1
Coef['true'][1,1] = -kappa
Coef['true'][2,1] = -gamma
Coef['true'][6,1] = -epsilon

mc = 'Coef'
Coef['dnn'] = Model[mc].get_weights()[0]

#####################################################################################
ax = {}
fig, (ax[0], ax[1]) = plt.subplots(1, 2,figsize=(16,5))
for d in range(n_var):
	ax[d].plot(Coef['true'][:,d], 'k', lw=1.5)
	ax[d].plot(Coef['dnn'][:,d], 'ro')
plt.show()


# ##### Calculate approximation for dx from the coefficients achieved
Phi = {}
method = 'dnn'
######
dt = 0.001  # time step
Nt = int(np.floor(tf-t0)/dt + 1) #Number of time instances
t_span = np.linspace(t0, tf, Nt)

x = Model['Xtrue'].predict(Data_Reshape(t_span))
Phi[method] = Phi_ft(x, poly_order)


DATA['dx:' + method] = np.zeros(DATA['dx:true'].shape)
for i in range(n_var):
	DATA['dx:' + method][:,i] = np.dot(Phi[method],  Coef[method][:,i])


################################################################################
ax = {}
fig, (ax[0], ax[1]) = plt.subplots(1, 2, figsize=(13,6))

for i in range(n_var):
	for method in ['dnn']:
		cc = {'ls':'b', 'bpdn':'r', 'dnn':'g'}[method]
		mk = {'ls':'^', 'bpdn':'x', 'dnn':'o'}[method]
		ax[i].plot(t_span, DATA['dx:'+method][:,i], cc+'-', lw=1.5)
		jump = 50
		ax[i].plot(t_span[::jump], DATA['dx:'+method][:,i][::jump], cc+mk, label=method.upper(), mew=1.5, mfc='none')
	ax[i].plot(t_span, DATA['dx:true'][:,i], 'k-', label='True', lw=2)

	ax[i].set_xlabel('t')
	ax[i].set_ylabel('dx{:d}/dt'.format(i))
	ax[i].set_title('Error scale = {:f}'.format(noise_scale))
	ax[i].legend(loc='upper left', borderpad=0.3)

plt.show()


























