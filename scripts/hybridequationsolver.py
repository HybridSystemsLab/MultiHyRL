"""
A python implementation of the Hybrid Systems Simulation Toolbox Solver
"""

import numpy as np
from scipy.integrate import solve_ivp
from functools import partial
import matplotlib.pyplot as plt 


class hybridsystem(object):
    def __init__(self, f, C, g, D, x0, rule = 0, j_max=100, t_max=100, atol=1e-14, rtol=1e-14, maxS=0.001):
        # Initial condtions
        self.x0 = x0
        self.state_dimension = ( len(x0), 1)
        self.states = x0
        # System model
        self.f = f
        self.C = C
        self.g = g
        self.D = D
        # maximum hybrid time
        self.t_max = t_max
        self.j_max = j_max
        # Initialize hybrid time
        self.t =np.array([0])
        self.j = np.array([0])

        # Solver parameters
        self.rule = rule
        self.rtol = rtol
        self.atol = atol
        self.maxS = maxS

    def _jump(self):
        nStep = np.array( self.g( self.states[:,-1] ) ).reshape(self.state_dimension)
        self.states = np.hstack( (self.states, nStep) )
        self.j = np.append( self.j, self.j[-1]+1 )
        self.t = np.append( self.t, self.t[-1] )

    def _flow(self):
        dyna = partial(self._dynamics)
        event = lambda t,x: self._cross_events(t, x)
        event.terminal = True
        sol = solve_ivp(dyna, [self.t[-1], self.t_max], self.states[:,-1].flatten(), max_step=self.maxS, atol=self.atol, rtol=self.rtol, events=event, method='LSODA', dense_output=True)
        self.states = np.hstack( ( self.states, sol.y ) )
        jSeq = sol.t*0 + self.j[-1]
        self.j = np.hstack( (self.j, jSeq) )
        self.t = np.hstack( (self.t, sol.t) )
        pass

    def _cross_events(self, t, x):
        if self.D(x) == 1 and self.rule == "jump":
            return 0
        if self.C(x) == 0:
            return 0
        return 1

    def jump_priority(self):
        if self.D(self.states[:,-1]) == 1:
            self._jump()
        elif self.C(self.states[:,-1]) == 1:
            self._flow()

    def flow_priority(self):
        if self.C(self.states[:,-1]) == 1:
            self._flow()
        elif self.D(self.states[:,-1]) == 1:
            self._jump()

    def solve(self):
        while  self.j[-1] < self.j_max and self.t[-1] < self.t_max:
            if self.rule == "jump":
                self.jump_priority()
            else:
                self.flow_priority()
            if self.D(self.states[:,-1]) == 0 and self.C(self.states[:,-1]) == 0:
                break

        return self.states, self.t, self.j

    def _dynamics(self, t, state):
        flow = np.zeros(len(state))
        fl = self.f(state)
        for i in range(len(state)):
            flow[i] = fl[i]
        return flow

def flow( x, gamma ):
    z1 = x[0]
    z2 = x[1]
    z3 = x[2]
    z4 = x[3]
    xdot = np.zeros( len(x) )
    xdot[0] = z2
    xdot[1] = -gamma
    xdot[2] = z4
    xdot[3] = gamma
    return xdot

def jump( x, lmbda, mbar ):
    z1 = x[0]
    z2 = x[1]
    z3 = x[2]
    z4 = x[3]
    Gamma0 = np.array([ [mbar - (1 - mbar) * lmbda, (1 - mbar) * (1 + lmbda)],
        [mbar * (1 + lmbda), 1 - mbar - mbar * lmbda] ])
    
    z_2_4 = np.array( [z2,z4] )
    ztemp = np.dot( Gamma0, z_2_4 )
    return [z1, ztemp[0], z1, ztemp[1]] 

def inside_C(x):
    z11 = x[0]
    z12 = x[1]
    z21 = x[2]
    z22 = x[3]
    if (z11 >= z21):
        inside = 1
    else:
        inside = 0
    return inside

def inside_D(x, gamma):
    z11 = x[0]
    z12 = x[1]
    z21 = x[2]
    z22 = x[3]
    if (z11 <= z21 and z12 <= z22):
        inside = 1
    else:
        inside = 0
    return inside

def plot_2d(t, j, x):
    fig, ax = plt.subplots( figsize=(8, 3) )
    j0 = 0
    i_ini = int(0)
    i_end = int(0)
    for j0 in range( int(j[-1])+1 ):
        i_ini = i_end
        j_current = j[i_ini]
        while j_current <= j0 and i_end + 1 < len( x ):
            i_end = int(i_end + 1)
            j_current = j[i_end]
        ax.plot(t[i_ini:i_end], x[i_ini:i_end], color = "C0")
        ax.plot(t[i_end-1: i_end + 1], x[i_end-1: i_end + 1], linestyle = ":", color = "C0")
    ax.grid()
    return ax