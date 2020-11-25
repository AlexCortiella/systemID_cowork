import os
import sys

import numpy as np
import math
import numpy.linalg as LA
import numpy.matlib
import pandas as pd
import matplotlib.pylab as plt
from matplotlib import rc
rc('font', **{'family':'Times', 'size':13})
# rc('text', usetex=True)
rc('savefig', **{'transparent':False})


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from scipy.integrate import odeint
from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import lstsq

####### SPGL1 package #######
from spgl1 import spg_bpdn
# from spgl1.lsqr import lsqr
# from spgl1 import spgl1, spg_lasso, spg_bp, spg_bpdn, spg_mmv
# from spgl1.spgl1 import norm_l1nn_primal, norm_l1nn_dual, norm_l1nn_project
# from spgl1.spgl1 import norm_l12nn_primal, norm_l12nn_dual, norm_l12nn_project

class Obj():
	def __init__(self):
		return

class FOLDER():
	def __init__(self):
		self.data = './data/duffling/'

		# self.post = './postprocessing/'
		# self.ex = './example/'

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
##### Algorithms
#####################################################################################

##### Initialize noise
Noise['approx'] = np.zeros((nsample, n_var))

##### Subtract assumed(=initialized) noise from observations (= the first approximation for x_true)
DATA['x:approx'] = DATA['x:noisy'] - Noise['approx']

##### Generate multi-variate Vandermonde matrix of degree poly_order in n variables
Poly = PolynomialFeatures(degree=poly_order)
Phi = {}
for i in ['true', 'approx']:
	Phi[i] = Poly.fit_transform(DATA['x:' + i])
P = Phi['approx'].shape[1] ## Number of basis functions
assert(P == math.factorial(poly_order+n_var)/math.factorial(poly_order)/math.factorial(n_var))

# xx = DATA['x:true'][:,0]
# yy = DATA['x:true'][:,1]
# phi_ft = [lambda x,y:np.ones(x.shape), lambda x,y:x, lambda x,y:y, lambda x,y:x**2., lambda x,y:x*y, lambda x,y:y**2., lambda x,y:x**3., lambda x,y:x**2.*y, lambda x,y:x*y**2., lambda x,y:y**3.]
# Phi['true'] = np.zeros((nsample, P))
# for i in range(len(phi_ft)):
# 	Phi['true'][:,i] = phi_ft[i](xx, yy)

##### Check Polynomial plots 
# xx, yy = np.meshgrid(DATA['x:approx'][:, 0],  DATA['x:approx'][:, 1])
# xxyy = np.concatenate((xx.reshape(-1,1), yy.reshape(-1,1)), axis=1)
# for iPoly in range(p):
# 	zz = Poly.fit_transform(xxyy)[:, iPoly]
# 	zz = zz.reshape(xx.shape)

# 	fig = plt.figure()
# 	ax = fig.gca(projection='3d')

# 	# zz2 = [np.ones(zz.shape), xx, yy, xx**2., xx*yy, yy**2., xx**3., xx**2.*yy, xx*yy**2., yy**3.][iPoly]
# 	# surf = ax.plot_surface(xx, yy, zz2-zz, cmap=cm.hot, linewidth=1, antialiased=False)
# 	surf = ax.plot_surface(xx, yy, zz, cmap=cm.hot, linewidth=1, antialiased=False)

# 	fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()


##### True coefficients
Coef = {}
Coef['true'] = np.zeros((P, n_var))
Coef['true'][2,0] = 1
Coef['true'][1,1] = -kappa
Coef['true'][2,1] = -gamma
Coef['true'][6,1] = -epsilon

# for i in range(n_var):
# 	val = DATA['dx:true'][:,i] - np.dot(Phi['true'],  Coef['true'][:,i]) 
# 	assert(np.sum(abs(val)) < 5e-14) ## should be zero theoretically 

#####################################################################################
##### Solve Equations for Coefficients
#####################################################################################

##### Generate LHS and integral matrix of the integral formulation using simple quadrature
Xb_Xa = DATA['x:approx'] - np.matlib.repmat(DATA['x:init'], nsample, 1)# LHS
dt_mat = np.tile(dt.reshape(-1,1), (1, P)) ### Reshape for convenience

ninteg = nsample  ##the number of integrations "int_a^b phi_i(x) dt" = the number of (aj, bj) instants
if ((uniform_dt == 'yes') & (dt[0] == 0.0001)):
	Int_Phi = np.loadtxt('./data/Int_Phi_dt0.0001.txt')
else:
	Int_Phi = np.zeros((ninteg, P))
	for i in range(ninteg):
		Int_Phi[i] = np.sum(dt_mat[0:i]*Phi['approx'][0:i], axis=0)

##### Solve the coefficient with a simple least-square problem
Coef['ls'] = np.zeros((P, n_var))
for i in range(n_var):
	b = Xb_Xa[:,i]
	A = Int_Phi

	Coef['ls'][:,i], res, rnk, s = lstsq(A, b)

##### Solve the coefficient with a regulaized least-square problem 
##### with a constraint  "min|Coef|_1 s.t. b - A*Coef <= tolerance"
Coef['bpdn'] = np.zeros((P, n_var)) ### basis pursuit denoising (BPDN)
tol_bpdn = 1e-3
for i in range(n_var):
	b = Xb_Xa[:,i]
	A = Int_Phi

	Coef['bpdn'][:,i], resid, grad, info = spg_bpdn(A, b, tol_bpdn, iter_lim=200, verbosity=0)

##### Calculate approximation for dx from the coefficients achieved
for method in ['ls', 'bpdn']:
	DATA['dx:' + method] = np.zeros(DATA['dx:true'].shape)
	for i in range(n_var):
		DATA['dx:' + method][:,i] = np.dot(Phi['approx'],  Coef[method][:,i])


##### Plot: Compare coefficients 
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(16,5))
ax1.plot(Coef['true'][:,0],'k')
ax1.plot(Coef['ls'][:,0],'ro')
ax1.plot(Coef['bpdn'][:,0],'bo')
ax1.legend(('True','LS','BPDN'))
ax1.set_title('State x1 (Error scale = {:f})'.format(noise_scale))
ax1.set_xlabel('basis index')
ax1.set_ylabel('coefficient value')

ax2.plot(Coef['true'][:,1],'k')
ax2.plot(Coef['ls'][:,1],'ro')
ax2.plot(Coef['bpdn'][:,1],'bo')
ax2.legend(('True','LS','BPDN'))
ax2.set_title('State x2 (Error scale = {:f})'.format(noise_scale))
ax2.set_xlabel('basis index')
ax2.set_ylabel('coefficient value')


##### Plot: Compare dx/dt solutions from different coefficient solutions
ax = {}
fig, (ax[0], ax[1]) = plt.subplots(1, 2, figsize=(13,6))

for i in range(n_var):
	for method in ['ls']:#, 'bpdn']:
		cc = {'ls':'b', 'bpdn':'r', 'dnn':'g'}[method]
		mk = {'ls':'o', 'bpdn':'x', 'dnn':'o'}[method]
		ax[i].plot(t_span, DATA['dx:'+method][:,i], cc+'-', lw=1.5)
		jump = 50
		ax[i].plot(t_span[::jump], DATA['dx:'+method][:,i][::jump], cc+mk, label=method.upper(), mew=1.5, mfc='none')
	ax[i].plot(t_span, DATA['dx:true'][:,i], 'k-', label='True', lw=2)

	ax[i].set_xlabel('t')
	ax[i].set_ylabel('dx{:d}/dt'.format(i))
	ax[i].set_title('Error scale = {:f}'.format(noise_scale))
	ax[i].legend(loc='upper left', borderpad=0.3)

plt.show()

######################################################################################################
###### Plot: dynamic responses 
for method in ['ls', 'bpdn']:
	DATA['x:approx:'+method] = DATA['x:init'] + np.dot(Int_Phi, Coef[method]) 
######################################################################################################


ax = {}
fig, (ax[0], ax[1]) = plt.subplots(1, 2, figsize=(13,6))

############################
i = 0
ref = DATA['x:true']
approx = DATA['x:noisy']

ax[i].plot(approx[:,0], approx[:,1], 'g.', label='Observations', ms=3)
ax[i].plot(ref[:,0], ref[:,1], 'k.', label='Exact dynamics', ms=3)

############################
i = 1
method = ['ls', 'bpdn'][0]
approx = DATA['x:approx:'+method]

jump = 50
ax[i].plot(ref[:,0][::jump], ref[:,1][::jump], 'k+', ms=8, label='Exact dynamics')
ax[i].plot(approx[:,0], approx[:,1], 'r.', ms=3,\
	label='Approximation ({:s})'.format(method.upper()))

xLim = ax[0].get_xlim()
yLim = ax[0].get_ylim()
for i in range(2):
	ax[i].set_xlim(xLim)
	ax[i].set_ylim(yLim)
	ax[i].legend(loc='upper left', borderpad=0.25, fontsize=13)
	ax[i].set_title('Duffing oscillator\n(Noise scale constant = {:1.3f})'.format(noise_scale))
	ax[i].set_ylabel('x2 (The second variable)')
	ax[i].set_xlabel('x1 (The first variable)')
	

###### Plot: dx/dt
method = ['ls', 'bpdn'][1]
approx = DATA['x:approx:'+method]

dxdt = {}
for i in range(n_var):
	y = approx[:,0]
	dxdt[i] = (y[1:] - y[:-1])/dt

ig, ax = plt.subplots()
t_mid = (t_span[1:] + t_span[:-1])/2.

ax.plot(t_mid, dxdt[0])
ax.plot(t_span, approx[:,1], lw=2)
# ax.plot(t_span[1:-1], dxdt[1])

####################################################################################
plt.show()
plt.close('all')








	