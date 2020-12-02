import numpy as np
from numpy.linalg import norm
import cvxpy as cp
import matplotlib.pyplot as plt

# Define local functions
def loss_fn(X, Y, beta):
    return cp.norm2(X @ beta - Y)**2

def regularizer(beta):
    return cp.norm1(beta)

def objective_fn(X, Y, beta, lambd):
    return loss_fn(X, Y, beta) + lambd * regularizer(beta)

def mse(X, Y, beta):
    return (1.0 / X.shape[0]) * loss_fn(X, Y, beta).value

def solveBP(X, y, w = None, lambda_reg = 0, verbose = False):

    """
    SolveBP: Solves a Basis Pursuit problem
    Usage
    ----------------------------------------------------------------------
    sol = SolveBP(X, y, w, lambda_reg, verbose)

    Input
    ----------------------------------------------------------------------
    X:             Either an explicit nxN matrix, with rank(A) = min(N,n) 
                   by assumption, or a string containing the name of a 
                   function implementing an implicit matrix.
                   vector of length n.
                   length of solution vector. 

    y:             input/RHS vector
    w:             weight vector so that len(w) = # columns of X
    lambda_reg:    If 0 or omitted, Basis Pursuit is applied to the data, 
                   otherwise, Basis Pursuit Denoising is applied with 
                   parameter lambda (default 0). 
    OptTol:        Error tolerance, default 1e-3

    Outputs
    ----------------------------------------------------------------------
    sol             solution of BP

    Description

    SolveBP solves the Basis Pursuit Denoising (BPDN) problem

    min  1/2||b - A*x||_2^2 + lambda*||x||_1

    using the library cvxpy  
    """
    
    #Normalize columns of X
    p_basis = X.shape[1]
    c_norm = np.linalg.norm(X, axis=0)
    Wn = np.diag(1 / c_norm)#column-normalization matrix
   
    
    #Weight the columns of the basis matrix
    if w is None:
        Ww = np.eye(p_basis)
    else:
        if len(w) != p_basis:
            raise Exception("The weight vector provided do not match the dimensions of X."
                        "Please, provide a weight vector so that len(w) = # columns of X")
        else:
            Ww = np.diag(1 / w)
    
    Wnw = np.dot(Wn, Ww)
    Xnw = np.dot(X, Wnw)
   
    #Set up variables
    beta_tilde = cp.Variable(p_basis)
    lambd = cp.Parameter(nonneg=True)
    #Set up problem
    problem = cp.Problem(cp.Minimize(objective_fn(Xnw, y, beta_tilde, lambd)))
    #Define lambda
    lambd.value = lambda_reg
    #Solve BPDN using CVXOPT
    problem.solve(solver = cp.CVXOPT, verbose=verbose)
    #problem.solve()
    #De-normalize solution
    sol = np.dot(Wnw, beta_tilde.value)
    
    #Compute residual
    residual = norm(np.dot(X, sol) - y)
    reg_residual = norm(beta_tilde.value, ord = 1)
    
    
    return [sol, (residual, reg_residual)]

def menger(P1, P2, P3):
    
    u = P1 - P2
    v = P3 - P2

    return 2 * (v[0] * u[1] - u[0] * v[1]) / ( norm(u) * norm(v) * norm(u - v) )

def check_lambdas(lambda_min, lambda_max, X, y, w = None, tol = 1e-8, verbose = False):
    
#     #Make sure that the solver does not fail for the range of lambdas 
#     while True:
#         try:
#             bpdn = solveBP(X, y, lambda_reg = lambda_min)
#             residual_min, reg_residual_max = bpdn[1]
#         except:
#             print('lambda_min too small. Increasing it 10 times...')
#             lambda_min = lambda_min * 10
#             continue
#         else:
#             break
    
#     while True:
#         try:
#             bpdn = solveBP(X, y, lambda_reg = lambda_max)
#             residual_max, reg_residual_min = bpdn[1]
#         except:
#             print('lambda_max too large. Reducing it 10 times...')
#             lambda_max = lambda_max / 10
#             continue
#         else:
#             break

    bpdn = solveBP(X, y, w, lambda_reg = lambda_max)
    residual_max, reg_residual_min = bpdn[1]
    bpdn = solveBP(X, y, w, lambda_reg = lambda_min)
    residual_min, reg_residual_max = bpdn[1]
    
    #Make sure the residuals are not too small to help lcurve_corner converge
    while residual_min < tol:
        if verbose:
            print('lambda_min too small. Increasing it 10 times...')
            
        lambda_min = lambda_min * 10
        bpdn = solveBP(X, y, w, lambda_reg = lambda_min)
        residual_min, reg_residual_max = bpdn[1]

    while reg_residual_min < tol:
        if verbose:
            print('lambda_max too large. Reducing it 10 times...')
            
        lambda_max = lambda_max / 10
        bpdn = solveBP(X, y, w, lambda_reg = lambda_max)
        residual_max, reg_residual_min = bpdn[1]
            
    return (lambda_max, lambda_min)

def normalize_get(residual, reg_residual):
    
    residual_max = np.max(residual)
    reg_residual_max = np.max(reg_residual)
    
    residual_min = np.min(residual)
    reg_residual_min = np.min(reg_residual)
    
    #Compute normalization coefficients to transform the L-curve into a [-1, 1] range
    cres0 = 1 - 2/(np.log10(residual_max) - np.log10(residual_min)) * np.log10(residual_max)
    cres1 = 2 / (np.log10(residual_max) - np.log10(residual_min))
    creg0 = 1 - 2/(np.log10(reg_residual_max) - np.log10(reg_residual_min)) * np.log10(reg_residual_max)
    creg1 = 2 / (np.log10(reg_residual_max) - np.log10(reg_residual_min))
    
    
    return (cres0, cres1, creg0, creg1)

def normalize_fit(residual, reg_residual, normalization_coefs):
    
    #Unpack coefficients
    cres0, cres1, creg0, creg1 = normalization_coefs
    
    
    xi = cres0 + cres1 * np.log10(residual)
    eta = creg0 + creg1 * np.log10(reg_residual)
    
    return (xi, eta)

def lcurve_corner(X, y, w = None, lambda_min = 1e-10, lambda_max = 1e10, epsilon = 0.001, max_iter = 50, tol = 1e-8, normalize = False, verbose = False):
    
    n = len(y)
    pGS = (1 + np.sqrt(5))/2 #Golden search parameter
    gap = 1
    itr = 0
    lambda_itr = []

    #Check the range of lambdas and compute normalization coefficients
    lambda_max, lambda_min = check_lambdas(lambda_min, lambda_max, X, y, w, tol, verbose)

    lambda_vec = np.array([lambda_min, lambda_max, 0, 0])

    lambda_vec[2] = 10 ** ( (np.log10(lambda_vec[1]) + pGS * np.log10(lambda_vec[0])) / (1 + pGS) ) 
    lambda_vec[3] = 10 ** ( np.log10(lambda_vec[0]) + np.log10(lambda_vec[1]) - np.log10(lambda_vec[2]) )

    ress = np.zeros(4)#residuals
    regs = np.zeros(4)#regularization residuals

    while (gap >= epsilon) and (itr <= max_iter):

        if itr == 0:

            for s in range(4):

                current_lambda = lambda_vec[s]

                #Run trend filter with current lambda

                bpdn = solveBP(X, y, w, lambda_reg = current_lambda, verbose = verbose)
                ress[s], regs[s] = bpdn[1]
            
            normalization_coefs = normalize_get(ress, regs)
            xis, etas = normalize_fit(ress, regs, normalization_coefs)
            
            if normalize:
                lc_res = list(xis)
                lc_reg = list(etas)
            else:
                lc_res = list(ress)
                lc_reg = list(regs)

            P = np.array([[xis[0],xis[1],xis[2],xis[3]], [etas[0],etas[1],etas[2],etas[3]]])           
            indx = np.argsort(lambda_vec)

            #Sort lambdas
            lambda_vec = lambda_vec[indx]
            P = P[:,indx]

        # Compute curvatures of the current points
        C2 = menger(P[:,0], P[:,1], P[:,2])
        C3 = menger(P[:,1], P[:,2], P[:,3])

        # Check if the curvature is negative and update values
        while C3 < 0:

            #Reassign maximum and interior lambdas and Lcurve points (Golden search interval)
            lambda_vec[3] = lambda_vec[2]
            P[:,3] = P[:,2]
            lambda_vec[2] = lambda_vec[1]
            P[:,2] = P[:,1]

            #Update interior lambda and interior point
            lambda_vec[1] = 10 ** ( (np.log10(lambda_vec[3]) + pGS * np.log10(lambda_vec[0])) / (1 + pGS) ) 

            bpdn = solveBP(X, y, w, lambda_reg = lambda_vec[1], verbose = verbose)
            res, reg = bpdn[1]
            
            xi, eta = normalize_fit(res, reg, normalization_coefs)
            
            P[:,1] = [xi,eta]

            C3 = menger(P[:,1], P[:,2], P[:,3])

        # Update values depending on the curvature at the new points
        if C2 > C3:

            current_lambda = lambda_vec[1]

            #Reassign maximum and interior lambdas and Lcurve points (Golden search interval)
            lambda_vec[3] = lambda_vec[2]
            P[:,3] = P[:,2]

            lambda_vec[2] = lambda_vec[1]
            P[:,2] = P[:,1]

            #Update interior lambda and interior point
            lambda_vec[1] = 10 ** ( (np.log10(lambda_vec[3]) + pGS * np.log10(lambda_vec[0])) / (1 + pGS) ) 
            
            bpdn = solveBP(X, y, w, lambda_reg = lambda_vec[1], verbose = verbose)
            res, reg = bpdn[1]
            
            xi, eta = normalize_fit(res, reg, normalization_coefs)
        
            P[:,1] = [xi,eta]

        else:

            current_lambda = lambda_vec[2]

            #Reassign maximum and interior lambdas and Lcurve points (Golden search interval)
            lambda_vec[0] = lambda_vec[1]
            P[:,0] = P[:,1]

            lambda_vec[1] = lambda_vec[2]
            P[:,1] = P[:,2]

            #Update interior lambda and interior point
            lambda_vec[2] = 10 ** ( np.log10(lambda_vec[0]) + np.log10(lambda_vec[3]) - np.log10(lambda_vec[1]) )
            
            bpdn = solveBP(X, y, w, lambda_reg = lambda_vec[2], verbose = verbose)
            res, reg = bpdn[1]
            
            xi, eta = normalize_fit(res, reg, normalization_coefs)

            P[:,2] = [xi, eta]

        gap = ( lambda_vec[3] - lambda_vec[0] ) / lambda_vec[3]

        if gap < epsilon:
            if verbose:
                print(f'  Convergence criterion reached in {itr} iterations.')

        lambda_itr.append(current_lambda)
        
        if normalize:
            lc_res.append(xi)
            lc_reg.append(eta)
        else:
            lc_res.append(res)
            lc_reg.append(reg)

        itr += 1

        if itr == max_iter:
            if verbose:
                print(f'  Maximum number of {itr} iterations reached.')

    #Solve for optimal lambda
    bpdn = solveBP(X, y, w, lambda_reg = current_lambda, verbose = verbose)
    sol = bpdn[0]
    res, reg = bpdn[1]
    
    if normalize:
        xi, eta = normalize_fit(res, reg, normalization_coefs)
        lc_res.append(xi)
        lc_reg.append(eta)
    else:
        lc_res.append(res)
        lc_reg.append(reg)

    return [sol, current_lambda, lambda_itr, (lc_res, lc_reg)]

def full_lcurve(X, y, w = None, lambda_min = 1e-10, lambda_max = 1e10, n_lambdas = 100, tol = 1e-8, normalize = False, plot_lc = False, verbose = False):
    
    p_basis = X.shape[1]
    sol = np.zeros((p_basis, n_lambdas))
    residual_lc = np.zeros(n_lambdas)
    reg_residual_lc = np.zeros(n_lambdas)
    
    #Check the range of lambdas and compute normalization coefficients
    lambda_max, lambda_min = check_lambdas(lambda_min, lambda_max, X, y, w, tol, verbose)
    
    #Generate array of lambdas
    lambdas = np.logspace(np.log10(lambda_min), np.log10(lambda_max), n_lambdas)
    
    if lambdas[0] < lambda_min:
        lambdas[0] = lambda_min 
        
    if lambdas[-1] > lambda_max:
        lambdas[-1] = lambda_max 
    
    #Loop over lambdas
    for i, lambd in enumerate(lambdas):
    
        bpdn = solveBP(X, y, w, lambda_reg = lambd, verbose = verbose)
        sol[:,i] = bpdn[0]
        residual_lc[i], reg_residual_lc[i] = bpdn[1]
    
    if normalize:
        
        normalization_coefs = normalize_get(residual_lc, reg_residual_lc)
        xi, eta = normalize_fit(residual_lc, reg_residual_lc, normalization_coefs)
        
        if plot_lc:
            plt.plot(xi, eta)
            plt.xlabel(r'$||y - X\beta||_2$')
            plt.ylabel(r'$||\beta||_1$')
        
        return [sol, (xi, eta)]
    
    else:
        
        if plot_lc:
            plt.loglog(residual_lc, reg_residual_lc)
            plt.xlabel(r'$||y - X\beta||_2$')
            plt.ylabel(r'$||\beta||_1$')
    
        return [sol, (residual_lc, reg_residual_lc)]