from phantomx_CPG import HOPF
from BodyCPG import BodyCPG
import numpy as np
from scipy import integrate
import time
import matplotlib.pyplot as plt

class LimbCPG(object):
    def __init__(self, motor_num = 3, coupling_factor = 0.35,
                 coupling = True, time_step = 0.01, leg_id = 0):
        self._oscillator = []
        self._motor_num = motor_num
        self._coupling_factor = coupling_factor
        self._coupling = coupling
        self._time_step = time_step
        self._p0 = [-1.0] * (self._motor_num * 2)
        self._leg_id = leg_id
        self._parentCPG = BodyCPG()
        if self._leg_id != 0:
            self._hip_joint_x, self._hip_joint_y = self._parentCPG.get_hip_joint(i = leg_id)
        self._oscillator.append(self.init_oscillator(omega_sw=2*np.pi))
        self._oscillator.append(self.init_oscillator(omega_sw=2*np.pi))
        
    def init_oscillator(self, eplison=100, sigma=100, a=50, 
                 mu=1, beta=0.5, omega_sw=2*np.pi,
                 time_step=0.01, coupling = False, 
                 coupling_factor = 0.1, y0=[0.1, 0.1]):
        return HOPF(eplison, sigma, a, mu, beta, omega_sw, time_step, coupling, coupling_factor, y0)
    

    def coupling_hopf(self, pos, time_step):
        x1, y1, x2, y2, x3, y3 = pos
        eplison=[]
        sigma=[]
        a=[]
        mu=[]
        beta=[]
        omega_sw = []
        omega_st = []
        omega = []
        x=[x1, x2, x3]
        y=[y1, y2, y3]
        r_square = []
        dx=[]
        dy=[]
        w = [1, 1, 1]

        for i in range(self._motor_num - 1):
            eplison.append(self._oscillator[i]._eplison)
            sigma.append(self._oscillator[i]._sigma)
            a.append(self._oscillator[i]._a)
            mu.append(self._oscillator[i]._mu)
            beta.append(self._oscillator[i]._beta)
            omega_sw.append(self._oscillator[i]._omega_sw)

        for i in range(self._motor_num - 1):
            omega_st.append(((1 - beta[i]) / beta[i]) * omega_sw[i])
            omega.append(omega_st[i] / (np.e ** (-a[i] * x[i + 1]) + 1) + omega_sw[i] / (np.e ** (a[i] * x[i + 1]) + 1))
            r_square.append(x[i + 1]**2 + y[i + 1]**2)

        dx.append(self._hip_joint_x)
        dy.append(self._hip_joint_y)

        for i in range(self._motor_num - 1):
            dx.append(eplison[i] * (mu[i] - r_square[i]) * x[i + 1] - omega[i] * y[i + 1])
            dy.append(sigma[i] * (mu[i] - r_square[i]) * y[i + 1] + omega[i] * x[i + 1])

        theta = np.pi/2

        dx[0] += self._coupling_factor * 0
        dy[0] += self._coupling_factor * 0
        dx[1] += self._coupling_factor * (x[0] * np.cos(theta) - y[0] * np.sin(theta) + x[2] * np.cos(0) - y[2] * np.sin(0))
        dy[1] += self._coupling_factor * (x[0] * np.sin(theta) + y[0] * np.cos(theta) + x[2] * np.sin(0) + y[2] * np.cos(0))
        dx[2] += self._coupling_factor * (x[0] * np.cos(theta) - y[0] * np.sin(theta) + x[1] * np.cos(0) - y[1] * np.sin(0))
        dy[2] += self._coupling_factor * (x[0] * np.sin(theta) + y[0] * np.cos(theta) + x[1] * np.sin(0) + y[1] * np.cos(0))

        return dx[0], dy[0], dx[1], dy[1], dx[2], dy[2]
    
    def calculate(self, t):
        data = integrate.odeint(self.coupling_hopf, self._p0, t)
        return data
    
    def show(self, data, t):
        # plt.plot(t, data[:, 0], label="x1")
        plt.plot(t, data[:, 1], label="y1")
        # plt.plot(t, data[:, 2], label="x2")
        plt.plot(t, data[:, 3], label="y2")
        # plt.plot(t, data[:, 4], label="x3")
        plt.plot(t, data[:, 5], label="y3")
        plt.legend()
        plt.show()

if __name__ == '__main__':
    start = time.time()
    t = np.arange(0, 100, 0.01)
    legCPG = LimbCPG(leg_id=1)
    data = legCPG.calculate(t)
    legCPG.show(data, t)
    end = time.time()
    print("time: ", end - start)
        

        