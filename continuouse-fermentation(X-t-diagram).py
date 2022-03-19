import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# simulating X-t-diagram to determine when steady-state is reached using a certain dilution rate; without maintenance metabolism
# Sf: substrate concentration in feed [g/L]
# X0: inoculation biomass concentration [g/L]
# Ks: Monod constant [g/L]
# mumax: max growth rate [1/h]
# Yxs: biomass/substrate yield [g/g]
# qpmax: max production rate [1/h]
# Yps: product/substrate yield [g/g]
# Kp: production constant [g/L]
# D: dilution rate [1/h]
# S0: starting substrate concentration [g/L]

Sf = 10.0
X0 = 0.2
Ks = 3.0
mumax = 0.3
Yxs = 0.6
qpmax = 0.4
Yps = 0.7
Kp = 1.0
D = 0.14
S0 = 5.0

def mu(S):
  return mumax*S/(Ks+S)
def Xn(X,S):
  return mu(S)*X
def qp(S):
  return qpmax*S/(S+Kp)
def Pn(X,S):
  return qp(S)*X

def ODE_conti(t,x):
    X,S,P = x
    dX = Xn(X,S)-D*X
    dS = (-1/Yxs)*Xn(X,S)+D*(Sf-S)+(-1/Yps)*Pn(X,S)
    dP = qp(S)*X-D*P
    return [dX,dS,dP]

y0 = [X0,S0,0]
t = np.arange(0,50,0.01)
t_span = (0.0,50.0)

sol = solve_ivp(ODE_conti,t_span,y0,t_eval=t)
plt.plot(sol.t,sol.y[0])
plt.plot(sol.t,sol.y[1])
plt.plot(sol.t,sol.y[2])
plt.xlabel('Duration [h]')
plt.ylabel('Concentration [g/L]')
plt.legend(['Cell concentration','Substrate concentration','Product concentration'])
plt.show()
