from phantomx_CPG import HOPF
import numpy as np
from scipy import integrate
import time
import matplotlib.pyplot as plt

class BodyCPG(object):
    """description of class"""
    def __init__(self, leg_num = 6, coupling_factor = 0.35, 
                coupling = True, time_step = 0.01):
        self._oscillator = []
        self._leg_num = leg_num
        self._coupling_factor = coupling_factor
        self._coupling = coupling
        self._time_step = time_step
        self._p0 = [-1.0] * (self._leg_num * 2)
        if self._coupling == True:
            self._oscillator = [self.init_oscillator(omega_sw=2*np.pi)] * self._leg_num 
            
        self._dx = []
        self._dy = []

    def init_oscillator(self, eplison=100, sigma=100, a=50, 
                 mu=1, beta=0.5, omega_sw=2*np.pi,
                 time_step=0.01, coupling = False, 
                 coupling_factor = 0.1, y0=[0.1, 0.1]):
        return HOPF(eplison, sigma, a, mu, beta, omega_sw, time_step, coupling, coupling_factor, y0)
    
    def coupling_hopf(self, pos, time_step):
        '''
        耦合项
        '''
        x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6 = pos
        eplison=[]
        sigma=[]
        a=[]
        mu=[]
        beta=[]
        omega_sw = []
        omega_st = []
        omega = []
        x=[x1, x2, x3, x4, x5, x6]
        y=[y1, y2, y3, y4, y5, y6]
        r_square = []
        dx=[]
        dy=[]
        w = [1, 1, 1, 1, 1, 1]

        for i in range(self._leg_num):
            eplison.append(self._oscillator[i]._eplison)
            sigma.append(self._oscillator[i]._sigma)
            a.append(self._oscillator[i]._a)
            mu.append(self._oscillator[i]._mu)
            beta.append(self._oscillator[i]._beta)
            omega_sw.append(self._oscillator[i]._omega_sw)

        # for i in range(self._leg_num):
            omega_st.append(((1 - beta[i]) / beta[i]) * omega_sw[i])
            omega.append(omega_st[i] / (np.e ** (-a[i] * x[i]) + 1) + omega_sw[i] / (np.e ** (a[i] * x[i]) + 1))
            r_square.append(x[i]**2 + y[i]**2)

        # for i in range(self._leg_num):
            dx.append(eplison[i] * (mu[i] - r_square[i]) * x[i] - omega[i] * y[i])
            dy.append(sigma[i] * (mu[i] - r_square[i]) * y[i] + omega[i] * x[i])

            # dx.append(eplison[i] * (mu[i] - r_square[i]) * x[i] - 2*np.pi*w[i] * y[i])
            # dy.append(sigma[i] * (mu[i] - r_square[i]) * y[i] + 2*np.pi*w[i] * x[i])
        
        theta = np.pi
        # 1、 3、 5同相， 2、 4、 6与1反相
        # 0-2-4 || 1-3-5
        dx[0] += self._coupling_factor * (x[1]*np.cos(theta) - y[1]*np.sin(theta) + x[2]*np.cos(0) - y[2]*np.sin(0) + x[3]*np.cos(theta) - y[3]*np.sin(theta)
                                            +x[4]*np.cos(0) - y[4]*np.sin(0) + x[5]*np.cos(theta) - y[5]*np.sin(theta))
        dy[0] += self._coupling_factor * (x[1]*np.sin(theta) + y[1]*np.cos(theta) + x[2]*np.sin(0) + y[2]*np.cos(0) + x[3]*np.sin(theta) + y[3]*np.cos(theta)
                                            +x[4]*np.sin(0) + y[4]*np.cos(0) + x[5]*np.sin(theta) + y[5]*np.cos(theta))
        dx[1] += self._coupling_factor * (x[0]*np.cos(-theta) - y[0]*np.sin(-theta) + x[2]*np.cos(-theta) - y[2]*np.sin(-theta) + x[3]*np.cos(0) - y[3]*np.sin(0)
                                            +x[4]*np.cos(-theta) - y[4]*np.sin(-theta) + x[5]*np.cos(0) - y[5]*np.sin(0))
        dy[1] += self._coupling_factor * (x[0]*np.sin(-theta) + y[0]*np.cos(-theta) + x[2]*np.sin(-theta) + y[2]*np.cos(-theta) + x[3]*np.sin(0) + y[3]*np.cos(0)
                                            +x[4]*np.sin(-theta) + y[4]*np.cos(-theta) + x[5]*np.sin(0) + y[5]*np.cos(0))
        dx[2] += self._coupling_factor * (x[0]*np.cos(0) - y[0]*np.sin(0) + x[1]*np.cos(theta) - y[1]*np.sin(theta) + x[3]*np.cos(theta) - y[3]*np.sin(theta)
                                            +x[4]*np.cos(0) - y[4]*np.sin(0) + x[5]*np.cos(theta) - y[5]*np.sin(theta))
        dy[2] += self._coupling_factor * (x[0]*np.sin(0) + y[0]*np.cos(0) + x[1]*np.sin(theta) + y[1]*np.cos(theta) + x[3]*np.sin(theta) + y[3]*np.cos(theta)
                                            +x[4]*np.sin(0) + y[4]*np.cos(0) + x[5]*np.sin(theta) + y[5]*np.cos(theta))
        dx[3] += self._coupling_factor * (x[0]*np.cos(-theta) - y[0]*np.sin(-theta) + x[1]*np.cos(0) - y[1]*np.sin(0) + x[2]*np.cos(-theta) - y[2]*np.sin(-theta)
                                            +x[4]*np.cos(-theta) - y[4]*np.sin(-theta) + x[5]*np.cos(0) - y[5]*np.sin(0))
        dy[3] += self._coupling_factor * (x[0]*np.sin(-theta) + y[0]*np.cos(-theta) + x[1]*np.sin(0) + y[1]*np.cos(0) + x[2]*np.sin(-theta) + y[2]*np.cos(-theta)
                                            +x[4]*np.sin(-theta) + y[4]*np.cos(-theta) + x[5]*np.sin(0) + y[5]*np.cos(0))
        dx[4] += self._coupling_factor * (x[0]*np.cos(0) - y[0]*np.sin(0) + x[1]*np.cos(theta) - y[1]*np.sin(theta) + x[2]*np.cos(0) - y[2]*np.sin(0)
                                            +x[3]*np.cos(theta) - y[3]*np.sin(theta) + x[5]*np.cos(theta) - y[5]*np.sin(theta))
        dy[4] += self._coupling_factor * (x[0]*np.sin(0) + y[0]*np.cos(0) + x[1]*np.sin(theta) + y[1]*np.cos(theta) + x[2]*np.sin(0) + y[2]*np.cos(0)
                                            +x[3]*np.sin(theta) + y[3]*np.cos(theta) + x[5]*np.sin(theta) + y[5]*np.cos(theta))
        dx[5] += self._coupling_factor * (x[0]*np.cos(-theta) - y[0]*np.sin(-theta) + x[1]*np.cos(0) - y[1]*np.sin(0) + x[2]*np.cos(-theta) - y[2]*np.sin(-theta)
                                            +x[3]*np.cos(0) - y[3]*np.sin(0) + x[4]*np.cos(-theta) - y[4]*np.sin(-theta))
        dy[5] += self._coupling_factor * (x[0]*np.sin(-theta) + y[0]*np.cos(-theta) + x[1]*np.sin(0) + y[1]*np.cos(0) + x[2]*np.sin(-theta) + y[2]*np.cos(-theta)
                                            +x[3]*np.sin(0) + y[3]*np.cos(0) + x[4]*np.sin(-theta) + y[4]*np.cos(-theta))
        
        for i in range(self._leg_num):
            self._dx.append(dx[i])
            self._dy.append(dy[i])

        return dx[0], dy[0], dx[1], dy[1], dx[2], dy[2], dx[3], dy[3], dx[4], dy[4], dx[5], dy[5]
    
    def calculate(self, t):
        '''
        计算hopf振荡器的运动
        '''
        data = integrate.odeint(self.coupling_hopf, self._p0, t)
        return data
    
    def show(self, data, t):
        '''
        显示hopf振荡器的运动
        '''
        # plt.plot(t, data[:, 0], label="x1")
        plt.plot(t, data[:, 1], label="y1")
        # plt.plot(t, data[:, 2], label="x2")
        plt.plot(t, data[:, 3], label="y2")
        # plt.plot(t, data[:, 4], label="x3")
        plt.plot(t, data[:, 5], label="y3")
        # plt.plot(t, data[:, 6], label="x4")
        plt.plot(t, data[:, 7], label="y4")
        # plt.plot(t, data[:, 8], label="x5")
        plt.plot(t, data[:, 9], label="y5")
        # plt.plot(t, data[:, 10], label="x6")
        plt.plot(t, data[:, 11], label="y6")
        plt.legend()
        plt.show()

    def get_hip_joint(self, i):
        # return hipXYlist[2*(i-1)], hipXYlist[2*(i-1)+1]
        t = np.arange(0, 100, 0.01)
        data = self.calculate(t)
        # print("self._dx", self._dx)
        return self._dx[i-1], self._dy[i-1]
        

if __name__ == "__main__":
    BodyCPGMOdule = BodyCPG()

    t = np.arange(0, 100, 0.01)
    data = BodyCPGMOdule.calculate(t)
    BodyCPGMOdule.show(data[8000:10000], t[8000:10000])