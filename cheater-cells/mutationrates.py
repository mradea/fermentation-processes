import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_ivp

# fermentation time:
fermentation_time = 400 # [h]

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
# assumption: cell concentration [g/L] ~ amount of cells

# simulate fermentations with different mutation rates:
rates = list(np.arange(1/10**6, 1/10**3, 1/10**5))
mut_fitness = 1.2 # 20% fitness advantage

# list of time points when threshold is reached
threshold_reached = []

for mut_rate in rates:

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
        dM = Xm(M,S)+X*mut_rate-D*M
        dX = Xn(X,S)-D*X-X*mut_rate
        dS = (-1/Yxs)*(Xn(X,S)+Xm(M,S))+D*(Sf-S)-(1/Yps)*qp(S)*X
        dP = Pn(X,S)-D*P
        return [dM,dX,dS,dP]

    t = list(np.arange(0,fermentation_time,0.01))

    y0 = [0,X0,Sf,0] # no mutants and product at t = 0 h

    t_span = (0.0,fermentation_time)

    sol = solve_ivp(ODE_conti,t_span,y0,t_eval=t)

    mutants = list(sol.y[0])

    normals = list(sol.y[1])
    
    # threshold: if amount of mutant cells > amount of normal cells; you could also choose a certain product concentration, etc,
    for i in mutants:
        j = mutants.index(i)
        if i > normals[j]:
            th = t[j]
            threshold_reached.append(th)
            break
            
if len(threshold_reached) == len(rates):
    # prevents the case that threshold is not reached for every mutation rate
    plt.plot(rates, threshold_reached)
    plt.xlabel('Mutation rate [1/h]')
    plt.ylabel('Ferm. time until threshold reached [h]')
    plt.show()
else:
    print('No threshold reached within the given fermentation time. Prolong the fermentation time or imcrease mutation rate please.')
