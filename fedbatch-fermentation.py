import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# defining parameters for fedbatch fermentation; linear feed starts with fermentation
# Sf: substrate concentration in feed [g/L]
# S0: starting substrate concentration [g/L]
# X0: inoculation biomass concentration [g/L]
# Ks: Monod constant [g/L]
# mumax: max growth rate [1/h]
# Yxs: biomass/substrate yield [g/g]
# F: feed [L/h]
# V0: starting volume [L]
# qpmx: max product rate [1/h]
# Yps: product/substrate yield [g/g]
# Kp: product constant [g/L]

Sf = 5.0
S0 = 0.5
X0 = 0.3
Ks = 2.0
mumax = 0.3
Yxs = 0.6
F = 0.05
V0 = 1.0
qpmax = 0.5
Yps = 0.5
Kp = 1.0

# defining equations for growth and product formation
def mu(S):
  return mumax*S/(Ks+S)
def Xn(X,S):
  return mu(S)*X
def qp(S):
  return qpmax*S(Kp+S)
def Pn(X,S):
  return qp(S)*X

# defining differential equations for fed batch
def ODE_fedbatch(t,x):
  X,S,P,V = x
  dX = Xn(X,S)-(F/V)*X
  dS = (-1/Yxs)*Xn(X,S)+(F/V)*(Sf-S)+(-1/Yps)*Pn(X,S)
  dP = Pn(X,S)-(F/V)*P
  dV = F
  return [dX,dS,dP,dV]

# defining starting vector
y0 = [X0,S0,0,V0]

# duration of fermentation t = 50
t = np.arange(0,50,0.01)
t_span = (0.0,50.0)

sol = solve_ivp(ODE_fedbatch,t_span,y0,t_eval=t)

# subplots to demonstrate change of volume on second y axis
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(sol.t,sol.y[0],'g-')
ax1.plot(sol.t,sol.y[1],'b-')
ax1.plot(sol.t,sol.y[2], 'y')
ax2.plot(sol.t,sol.y[3],'r-')
ax1.set_xlabel('Duration [h]')
ax1.set_ylabel('Concentration [g/L]')
ax2.set_ylabel('Volume [L]')
ax1.legend(['Cell concentration','Substrate concentration','Product concentration'])
ax2.legend(['Volume'])
