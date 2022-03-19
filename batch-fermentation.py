import numpy as np
import matplotlib.pyplot as plt
from scipy_integrate import solve_ivp

# defining some parameters for batch fermentation process with product production but without maintenance metabolism
# S0: substrate concentration in reactor at t = 0 [g/L]
# X0: incoulation biomass [g/L]
# Ks: Monod constant [g/L]
# Kp: product constant [g/L]
# mumax: max growth rate [1/h]
# qpmax: max product rate [1/h]
# Yxs: biomass/substrate yield [g/g]
# Yps: produc/substrate yield [g/g]
S0 = 15.0
X0 = 0.2
Ks = 2.0
Kp = 1.0
mumax = 0.3
qpmax = 0.4
Yxs = 0.5
Yps = 0.4

# equations to describe growth of organism in batch fermentation 
# mu: growth rate 
# Xn: new biomass concentration 
# qp: product rate
# Pn: new product concentration
def mu(S):
  return mumax*S/(Ks+S)
def Xn(X,S):
  return X*mu(S)
def qp(S):
  return qpmax*S/(Kp+S)
def Pn(X,S):
  return X*qp(S)

# ordinary differential equations to describe change of biomass and substrate concentration in batch fermentation
def ODE_batch(t,x):
  X,P,S = x
  dX = Xn(X,S)
  dP = Pn(X,S)
  dS = (-1/Yxs)*Xn(X,S)+(-1/Yps)*Pn(X,S)
  return [dX,dP,dS]

# starting vector w/ concentrations at t = 0 
y0 = [X0,0,S0]

# t = duration of fermentation 
# t_span has to be defined for solve_ivp
t = np.arange(0,50,0.01)
t_span = (0.0,50.0)

# solving ODEs and plotting concentration of X and S using solve_ivp from scipy
sol = solve_ivp(ODE_batch,t_span,y0,t_eval=t)
plt.plot(sol.t,sol.y[0])
plt.plot(sol.t,sol.y[1])
plt.plot(sol.t,sol.y[2])
plt.xlabel('Duration [h]')
plt.ylabel('Concentration [g/L]')
plt.legend(['Cell concentration','Product concentration', 'Substrate concentration']
plt.show()
