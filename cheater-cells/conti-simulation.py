import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_ivp

# fermentation time [h]:
fermentation_time = 400

# defining parameters
# Sf: substrate concentration in feed [g/L]
# X0: inoculation biomass concentration [g/L]
# Ks: Monod constant [g/L]
# mumax: max growth rate [1/h]
# Yxs: biomass/substrate yield [g/g]
# qpmax: max production rate [1/h]
# Yps: product/substrate yield [g/g]
# D: dilution rate [1/h]

Sf = 15.0
X0 = 0.1
Ks = 3.0
mumax = 0.3
Yxs = 0.9
qpmax = 0.5
Yps = 0.7
Kp = 1.0
D = 0.14

# Xn/X: normal cells
# Xm/M: mutated cells
# example: mutationrate 1:1,000,000 every 20 min (e.g. E.coli)
# assumption: cell concentration [g/L] ~ amount of cells

mut_rate_interval = 1/10**6
mut_interval = 1/3 #[h]
mut_rate_hour = mut_rate_interval/mut_interval
mut_fitness = 1.2 # fitness advantage of 20 %

def mu(S):
    return mumax*S/(Ks+S)
def mu_new(S):
    return mumax*mut_fitness*S/(Ks+S)
def Xn(X,S):
    return mu(S)*X
def Xm(M,S):
    return mu_new(S)*M
def qp(S):
    return qpmax*S/(Kp+S)
def Pn(X,S):
    return qp(S)*X

def ODE_conti(t,x):
    M,X,S,P = x
    dM = Xm(M,S)+X*mut_rate_hour-D*M
    dX = Xn(X,S)-D*X-X*mut_rate_hour
    dS = (-1/Yxs)*(Xn(X,S)+Xm(M,S))+D*(Sf-S)-(1/Yps)*qp(S)*X
    dP = Pn(X,S)-D*P
    return [dM,dX,dS,dP]

y0 = [0,X0,Sf,0] # no mutants and product at t = 0 h
t = np.arange(0,fermentation_time,0.01)
t_span = (0.0,fermentation_time)

sol = solve_ivp(ODE_conti,t_span,y0,t_eval=t)
plt.plot(sol.t,sol.y[0])
plt.plot(sol.t,sol.y[1])
plt.plot(sol.t,sol.y[2])
plt.plot(sol.t,sol.y[3])
plt.xlabel('Duration [h]')
plt.ylabel('Concentration [g/L]')
plt.legend(['Mutated cells','Normal cells','Substrate', 'Product'])
plt.show()
