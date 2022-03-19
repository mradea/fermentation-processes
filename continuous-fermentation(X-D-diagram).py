import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# defining parameters
# Sf: substrate concentration in feed [g/L]
# X0: inoculation biomass concentration [g/L]
# Ks: Monod constant [g/L]
# mumax: max growth rate [1/h]
# Yxs: biomass/substrate yield [g/g]
# qpmax: max production rate [1/h]
# Yps: product/substrate yield [g/g]
# Kp: product constant [g/L]
# D: dilution rate [1/h]
# STY: space time yield [g/L*h]

# first, we are simulating a X-D-diagram
Sf = 10.0
X0 = 0.2
Ks = 3.0
mumax = 0.3
Yxs = 0.6
qpmax = 0.4
Yps = 0.7
Kp = 1.0

# reactor is in steady-state, so dX/dt = dS/dt = dP/dt = 0
# the diagram shows the concentrations while a certain dilution rate is used 
D = np.arange(0,mumax,0.001)
X = Yxs*(Sf-((D*Ks)/(mumax-D)))
# since the substrate concentration cannot be higher than Sf, use np.minimum to set it to Sf if it exceeds this value
S = np.minimum((D*Ks)/(mumax-D),Sf)
P = Yps*(Sf-((D*Kp)/(qpmax-D)))
STY = P*D

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(D,X,'g')
ax1.plot(D,P,'y')
ax1.plot(D,S,'r')
ax2.plot(D,STY)
ax1.set_ylim([0,20.0])
ax2.set_ylim([0,5.0])
ax1.set_xlabel('Dilution rate [1/h]')
ax1.set_ylabel('Concentration [g/L]')
ax2.set_ylabel('Space-time-yield [g/L*h]')
ax1.legend(['Cell concentration','Product concentration','Substrate concentration'])
ax2.legend(['STY'])
plt.show()
