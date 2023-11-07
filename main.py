import numpy as np
from numpy import linalg as LA
from scipy import optimize
from collections import namedtuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

DesiredState = namedtuple('DesiredState', 'pos vel acc jerk yaw yawdot')

def polyder(t, k = 0, order = 10):
    if k == 'all':
        terms = np.array([polyder(t,k,order) for k in range(1,5)])
    else:
        terms = np.zeros(order)
        coeffs = np.polyder([1]*order,k)[::-1]
        pows = t**np.arange(0,order-k,1)
        terms[k:] = coeffs*pows
    return terms

def Hessian(T,order = 10,opt = 4):
    n = len(T)
    Q = np.zeros((order*n,order*n))
    for k in range(n):
        m = np.arange(0,opt,1)
        for i in range(order):
            for j in range(order):
                if i >= opt and j >= opt:
                    pow = i+j-2*opt+1
                    Q[order*k+i,order*k+j] = 2*np.prod((i-m)*(j-m))*T[k]**pow/pow
    return Q

class trajGenerator:
    def __init__(self,waypoints,max_vel = 5,gamma = 100):
        self.waypoints = waypoints
        self.max_vel = max_vel
        self.gamma = gamma
        self.order = 10
        len,dim = waypoints.shape
        self.dim = dim
        self.len = len
        self.TS = np.zeros(self.len)
        self.optimize()
        self.yaw = 0
        self.heading = np.zeros(2)

    def get_cost(self,T):
        coeffs,cost = self.MinimizeSnap(T)
        cost = cost + self.gamma*np.sum(T)
        return cost

    def optimize(self):
        diff = self.waypoints[0:-1] - self.waypoints[1:]
        Tmin = LA.norm(diff,axis = -1)/self.max_vel
        T = optimize.minimize(self.get_cost,Tmin, method="COBYLA",constraints= ({'type': 'ineq', 'fun': lambda T: T-Tmin}))['x']

        self.TS[1:] = np.cumsum(T)
        self.coeffs, self.cost = self.MinimizeSnap(T)


    def MinimizeSnap(self,T):
        unkns = 4*(self.len - 2)

        Q = Hessian(T)
        A,B = self.get_constraints(T)

        invA = LA.inv(A)

        if unkns != 0:
            R = invA.T@Q@invA

            Rfp = R[:-unkns,-unkns:]
            Rpp = R[-unkns:,-unkns:]

            B[-unkns:,] = -LA.inv(Rpp)@Rfp.T@B[:-unkns,]

        P = invA@B
        cost = np.trace(P.T@Q@P)

        return P, cost

    def get_constraints(self,T):
        n = self.len - 1
        o = self.order

        A = np.zeros((self.order*n, self.order*n))
        B = np.zeros((self.order*n, self.dim))

        B[:n,:] = self.waypoints[ :-1, : ]
        B[n:2*n,:] = self.waypoints[1: , : ]

        #waypoints contraints
        for i in range(n):
            A[i, o*i : o*(i+1)] = polyder(0)
            A[i + n, o*i : o*(i+1)] = polyder(T[i])

        #continuity contraints
        for i in range(n-1):
            A[2*n + 4*i: 2*n + 4*(i+1), o*i : o*(i+1)] = -polyder(T[i],'all')
            A[2*n + 4*i: 2*n + 4*(i+1), o*(i+1) : o*(i+2)] = polyder(0,'all')

        #start and end at rest
        A[6*n - 4 : 6*n, : o] = polyder(0,'all')
        A[6*n : 6*n + 4, -o : ] = polyder(T[-1],'all')

        #free variables
        for i in range(1,n):
            A[6*n + 4*i : 6*n + 4*(i+1), o*i : o*(i+1)] = polyder(0,'all')

        return A,B

    def get_des_state(self,t):

        if t > self.TS[-1]: t = self.TS[-1] - 0.001

        i = np.where(t >= self.TS)[0][-1]

        t = t - self.TS[i]
        coeff = (self.coeffs.T)[:,self.order*i:self.order*(i+1)]

        pos  = coeff@polyder(t)
        vel  = coeff@polyder(t,1)
        accl = coeff@polyder(t,2)
        jerk = coeff@polyder(t,3)

        #set yaw in the direction of velocity
        yaw, yawdot = self.get_yaw(vel[:2])

        return DesiredState(pos, vel, accl, jerk, yaw, yawdot)

    def get_yaw(self,vel):
        curr_heading = vel/LA.norm(vel)
        prev_heading = self.heading
        cosine = max(-1,min(np.dot(prev_heading, curr_heading),1))
        dyaw = np.arccos(cosine)
        norm_v = np.cross(prev_heading,curr_heading)
        self.yaw += np.sign(norm_v)*dyaw

        if self.yaw > np.pi: self.yaw -= 2*np.pi
        if self.yaw < -np.pi: self.yaw += 2*np.pi

        self.heading = curr_heading
        yawdot = max(-30,min(dyaw/0.005,30))
        return self.yaw,yawdot
    
if __name__ == "__main__":
    waypoints = np.array([[0,0,1], [1,0,3], [5,1,2], [3,4,2]])

    #Generate trajectory through waypoints
    traj = trajGenerator(waypoints, max_vel = 5, gamma = 1e2)
    dt = 0.05
    iter = 0
    path = []
    while traj.TS[-1] - dt*iter > 0:
        des_state = traj.get_des_state(dt*iter)
        # print(des_state.pos)
        path.append(des_state.pos)
        iter += 1
    path = np.array(path)
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(waypoints[:,0], waypoints[:,1], waypoints[:,2], label="Path")
    ax.scatter(path[:,0], path[:,1], path[:,2], label="Trajectory")
    ax.legend()
    plt.show()