# import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D

def objective_f(FRACTION, COEFFICIENT, STOCHASTIC, MU, SIGMA):
    """
    Parameters
    ----------
    FRACTION : Fraction of total capital
    COEFFICIENT : The coefficients of fractions in objective function, namely, Exp[Y_i * 1_(Xi>=xi)]
    STOCHASTIC : True or False
    MU : Mean value of the distrbution of the noise to be added
    SIGMA : Standard deviation of the noise

    Returns
    -------
    function : Value of objective function
    """
    function = np.sum(np.multiply(FRACTION, COEFFICIENT))
    if STOCHASTIC:
        function = function + np.random.normal(MU, SIGMA)
    return function

# estimate the gradient using SPSA
def estimate_gradient(FRACTION, i, COEFFICIENT, STOCHASTIC, MU, SIGMA, BATCH, NR_ESTIMATES):
    """
    Parameters
    ----------
    FRACTION : Fraction of capital
    i : The number of times of estimations
    COEFFICIENT : The coefficients of fractions in objective function, namely, Exp[Y_i * 1_(Xi>=xi)]
    STOCHASTIC : True or False
    MU : Mean value of the distrbution of the noise to be added
    SIGMA : Standard deviation of the noise
    BATCH : True of False. Whether to estimate the gradient from a batch of data
    NR_ESTIMATES : size of the batch

    Returns
    -------
    gradient_estimate : the value of estimated gradient

    """
    eta_i = 1/(i+1)
    delta_i = np.random.choice((-0.01, 0.01), size = 3)
    if BATCH:
        gradient_estimates = np.zeros((NR_ESTIMATES, 3))
        for n in range(NR_ESTIMATES):
            delta_i = np.random.choice((-0.01, 0.01), size = 3)
            perturbation_high = objective_f(FRACTION + eta_i * delta_i, COEFFICIENT, STOCHASTIC, MU, SIGMA)
            perturbation_low = objective_f(FRACTION - eta_i * delta_i, COEFFICIENT, STOCHASTIC, MU, SIGMA)
            numerator = perturbation_high-perturbation_low
            denominator = 2*eta_i*delta_i
            gradient_estimates[n] = np.divide(numerator, denominator)
        gradient_estimate = np.mean(gradient_estimates, axis=0)
    else:
        perturbation_high = objective_f(FRACTION + eta_i * delta_i, COEFFICIENT, STOCHASTIC, MU, SIGMA)
        perturbation_low = objective_f(FRACTION - eta_i * delta_i, COEFFICIENT, STOCHASTIC, MU, SIGMA)
        numerator = perturbation_high-perturbation_low
        denominator = 2*eta_i*delta_i
        gradient_estimate = np.divide(numerator, denominator)
    
    return gradient_estimate

def SPSA(FRACTION_0, INVEST_TYPE, EPSILON_TYPE, EPSILON_VALUE, NR_ITERATIONS, STOCHASTIC, MU, SIGMA, BATCH, NR_ESTIMATES, NR_SAMPLES):
    """
    Parameters
    ----------
    FRACTION_0 : Initial value of the fraction
    INVEST_TYPE : "big" or "small", to identify the question type
    EPSILON_TYPE : "fixed" or "decreasing"
    EPSILON_VALUE : value of EPSILON of the fixed type
    NR_ITERATIONS : number of iterations, which is the number of updating the fractions
    STOCHASTIC : True or False
    MU : Mean value of the distrbution of the noise to be added
    SIGMA : Standard deviation of the noise
    BATCH : True of False. Whether to estimate the gradient from a batch of data
    NR_ESTIMATES : size of the batch
    NR_SAMPLES : number of samples used to estimate the coefficients

    Returns
    -------
    fractions : numpy arrays of size (NR_ITERATIONS+1, 3)
        store the fractions in each iteration
    gradients : numpy arrays of size (NR_ITERATIONS, 3)
        store the gradients estimated in each iteration
    objective_values : numpy arrays of size(NR_ITERATIONS+1, 3)
        store the objective function values in each iteration

    """
    fractions = np.zeros((NR_ITERATIONS+1, 3))
    gradients = np.zeros((NR_ITERATIONS, 3))
    coefficients = np.zeros((NR_ITERATIONS+1,3))
    objective_values = np.zeros(NR_ITERATIONS)
    fractions[0] = FRACTION_0
    coefficients[0] = estimate_expectation(NR_SAMPLES, fractions[0], INVEST_TYPE)
    for i in range(NR_ITERATIONS):
        if(INVEST_TYPE == 'small'):
            g = estimate_gradient(fractions[i], i, coefficients[0], STOCHASTIC, MU, SIGMA, BATCH, NR_ESTIMATES)
        else:
            g = estimate_gradient(fractions[i], i, coefficients[i], STOCHASTIC, MU, SIGMA, BATCH, NR_ESTIMATES)
        gradients[i] = g
        if EPSILON_TYPE == 'fixed':
            if OPTIMIZATION_TYPE == 'minimization':
                fractions[i+1] = fractions[i] - EPSILON_VALUE * g
            if OPTIMIZATION_TYPE == 'maximization':
                fractions[i+1] = fractions[i] + EPSILON_VALUE * g
        if EPSILON_TYPE == 'decreasing':
            if OPTIMIZATION_TYPE == 'minimization':
                fractions[i+1] = fractions[i] - 1/(i+1) * g
            if OPTIMIZATION_TYPE == 'maximization':
                fractions[i+1] = fractions[i] + 1/(i+1) * g
        # Projection
        fractions[i+1] = [ max(0,fractions[i+1][j]) for j in range(3)]
        nz = np.count_nonzero(fractions[i+1])
        for j in range(3):
            if (fractions[i+1][j] > 0 ):
                fractions[i+1][j] -= (np.sum(fractions[i+1])-1)/nz    
        while(np.min(fractions[i+1])<0 or np.max(fractions[i+1])>1):
            fractions[i+1] = [ max(0,fractions[i+1][j]) for j in range(3)]
            nz = np.count_nonzero(fractions[i+1])
            for j in range(3):
                if (fractions[i+1][j] > 0 ):
                    fractions[i+1][j] -= (np.sum(fractions[i+1])-1)/nz
        
        if(INVEST_TYPE == 'big'):
            coefficients[i+1] = estimate_expectation(NR_SAMPLES, fractions[i+1], INVEST_TYPE)
            objective_values[i] = objective_f(fractions[i+1], coefficients[i+1], STOCHASTIC, MU, SIGMA)
        else:
            objective_values[i] = objective_f(fractions[i+1], coefficients[0], STOCHASTIC, MU, SIGMA)
        
    return fractions, gradients, objective_values


def estimate_expectation(n,fraction,INVEST_TYPE):
    """
    Parameters
    ----------
    n : number of samples
    fraction : current fraction
    INVEST_TYPE : "big" or "small", to identify the question type

    Returns
    -------
    exp[n] : the value of estimated expectation of Y_i * 1_(Xi>=xi)

    """
    x = np.zeros((n,3))
    samples = np.zeros((n,3))
    exp = np.zeros((n+1,3)) 
    for i in range(n):
        RHO = 0.6
        V = np.random.normal(0,1,1)
        ETA = np.random.normal([0,0,0],[1,2**0.5,3**0.5],3)
        W = np.random.exponential(0.3,1)
        X = np.divide(RHO*V+((1-RHO**2)**0.5)*ETA,max(W,1))
        x[i] = X
        if(INVEST_TYPE == 'small'):
            Y = np.random.uniform([0,0,0],X,3)
        else:
            Y = np.random.uniform([0,0,0],[1,2,3] + X * fraction,3)
        threshold = [2,3,1]
        temp = X-threshold
        indicator = np.zeros(3)
        for j in range(3):
            if(temp[j]!=0):
                indicator[j] = temp[j]/abs(temp[j])
            else:
                indicator[j] = 1
        samples[i] = Y * [max(indicator[i],0) for i in range(3)]
        exp[i+1] = exp[i] + (samples[i]-exp[i])/(i+1)   
# =============================================================================
#     # Plot the process volumes
#     fig, axs = plt.subplots(2,2, figsize=(10,10))
#     axs[0,0].hist(samples.T[0], bins = 20, range = (0,0.05),density = True)
#     axs[0,0].set_xlabel("sample profit from company1")
#     axs[0,1].hist(samples.T[1], bins = 20, range = (0,1),density = True)
#     axs[0,1].set_xlabel("sample profit from company2")
#     axs[1,0].hist(samples.T[2], bins = 20, range = (0,1),density = True)
#     axs[1,0].set_xlabel("sample profit from company3")
#     axs[1,1].plot(exp.T[0], color = "green", label = r'$e_1$')
#     axs[1,1].plot(exp.T[1], color = "blue", label = r'$e_2$')
#     axs[1,1].plot(exp.T[2], color = "grey", label = r'$e_3$')
#     axs[1,1].set_ylabel("Estimated expected value")
#     axs[1,1].set_xlabel("sample times")
#     fig.clear()
# =============================================================================
    return exp[n]
     
    
# The number of iterations in SPSA.
NR_ITERATIONS = 100
# 'fixed' or 'decreasing'
EPSILON_TYPE = 'decreasing' 
# In case of fixed epsilon, the value for epsilon.
EPSILON_VALUE = 0.1
# 'minimization' or 'maximization' for obejctive function
OPTIMIZATION_TYPE = 'maximization' 
# Whether to add stochasticity
STOCHASTIC = True # False or True
MU = 0 # mu parameter of the normal distribution for the noise.
SIGMA = 0.0001 # sigma parameter of the normal distribution for the noise.
# In case of stochasticity, whether to estimate the gradient multiple times and use the average as update
BATCH = True #False or True
# The size of batch
NR_ESTIMATES = 36
# The sample size for estimating the coeffients
NR_SAMPLES = 100
# The initial values
COEFFICIENT = np.zeros(3)
FRACTION_0 = np.ones(3)
FRACTION_0 = FRACTION_0/np.sum(FRACTION_0)

# question type
question = input("Type 'a','b' or 'c'\n")
if(question == 'c'):
    # The algorithm for streaming data
    INVEST_TYPE = 'small'
    length = 100
    threshold =[2,3,1]
    X = np.zeros((length+1,3))
    Y = np.zeros((length+1,3))
    RESULT = np.zeros((length+1,3))
# =============================================================================
#         input_str = input("Enter x1,x2,x3 (separated by spaces): ")
#         X = [float(x) for x in input_str.split()]
#         X = np.array(X)
#         input_str = input("Enter y1,y2,y3 (separated by spaces): ")
#         Y = [float(x) for x in input_str.split()]
#         Y = np.array(Y)
# =============================================================================
    for _ in range(length//2):
        X[_] = np.random.normal([2,3,1],[1,1.7,1.2],3)
        Y[_] = np.random.uniform(0,X[_],size=3)
        X[length-_] = np.random.normal([2.5,4,0],[1,1,1],3)
        Y[length-_] = np.random.uniform(0,X[_],size=3)
    for i in range(length):
        temp = X[i]-threshold
        indicator = np.zeros(3)
        for j in range(3):
            if(temp[j]!=0):
                indicator[j] = temp[j]/abs(temp[j])
            else:
                indicator[j] = 1
        parameter = Y[i] * [max(indicator[i],0) for i in range(3)]
        fraction = np.zeros(3)
        fraction[np.argmax(parameter)] = 1
        RESULT[i+1] = RESULT[i] + (fraction - RESULT[i])/(i+1)
    plt.plot(RESULT.T[0], label = r'$p_1$')
    plt.plot(RESULT.T[1], label = r'$p_2$')
    plt.plot(RESULT.T[2], label = r'$p_3$')
    plt.legend()
else:
    if(question == 'a'):
        INVEST_TYPE = 'small'
    if(question == 'b'):
        INVEST_TYPE = 'big'
    
    # nn is the times of doing optimization for output analysis 
    nn = 1
    ## final result of fraction in each optimization 
    fraction_final = np.zeros((nn,3))
    objective_final = np.zeros(nn)
    for i in range(nn):
        fractions, gradients, objective_values = SPSA(FRACTION_0, INVEST_TYPE, EPSILON_TYPE,  EPSILON_VALUE, NR_ITERATIONS, STOCHASTIC, MU, SIGMA, BATCH, NR_ESTIMATES, NR_SAMPLES)
        fraction_final[i] = fractions[-1]
        objective_final[i] = objective_values[-1]
    
    print('Iterations ',NR_ITERATIONS,'\nAverage p1,p2,p3: ', np.mean(fraction_final,axis = 0))
    
    # output analysis for more than 1 run
    if nn > 1 :
        fig, axs = plt.subplots(4,figsize=(10,15))
        fraction_final = fraction_final.T
        axs[0].set_xlabel(r"$p_1$")
        axs[0].hist(fraction_final[0],10,histtype='stepfilled')
        axs[1].set_xlabel(r"$p_2$")
        axs[1].hist(fraction_final[1],10,histtype='stepfilled')
        axs[2].set_xlabel(r"$p_3$")
        axs[2].hist(fraction_final[2],10,histtype='stepfilled')
        axs[3].set_xlabel(" optimized value")
        axs[3].hist(objective_final,20,color='red')
        print("The average optimized objective function value is",np.mean(objective_final))
    
    # Plot the updating process for the last one
    fig, axs = plt.subplots(2, figsize=(10,10))
    axs[0].plot(fractions.T[0], color="darkblue", label = r'$p_1$')
    axs[0].plot(fractions.T[1], color="deepskyblue", label = r'$p_2$')
    axs[0].plot(fractions.T[2], color="green", label = r'$p_3$')
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel(r"$p_n$")
    axs[0].set_title(r"$p_n$")
    axs[0].legend()
    
    objective_mean = np.mean(objective_values[-50:])
    axs[1].plot(objective_values, color="darkblue")
    print("The optimized value of the objective function is",objective_mean,"for the last optimization")
    axs[1].axhline(y=objective_mean, linestyle='--', c='r')
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel(r"f($p_n$)")
    axs[1].set_title(r"f($p_n$)")

#Validation
obj = np.zeros((20,20))
obj2 = np.zeros((20,20))
max_return = 0
coefficient_a = estimate_expectation(10000, (0,0,1), "small")
for i in range(21):
    p1 = i/20
    for j in range(20-i):
        p2 = j/20
        p3 = 1-p1-p2
        frac = np.zeros(3)
        frac[0] = p1
        frac[1] = p2
        frac[2] = p3
        coefficient_b = estimate_expectation(500, (0,0,1), "big")
        obj[i][j] = objective_f(frac, coefficient_a, False, MU, SIGMA)
        obj2[i][j] = objective_f(frac, coefficient_b, False, MU, SIGMA)
        
for i in range(20):
    for j in range (20):
        print(np.round(obj[i][j],3),end=" ")
    print()
for i in range(20):
    for j in range (20):
        print(np.round(obj2[i][j],3),end=" ")
    print()
