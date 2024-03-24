from src.CentralPatternGenerators.phantomx_CPG import HOPF
import numpy as np
from scipy import integrate
import time
import matplotlib.pyplot as plt

class PhantomxCPG(object):
    """description of class"""
    def __init__(self, leg_num = 6, motor_num = 3, coupling_factor = 0.75, 
                coupling = True, time_step = 0.01):
        self._body_oscillator = []
        self._limb_oscillator = []
        self._leg_num = leg_num
        self.__motor_num = motor_num
        self._coupling_factor = coupling_factor
        self._coupling = coupling
        self._time_step = time_step
        self._p0 = [-1.0] * ((self._leg_num * self.__motor_num) * 2)
        
        if self._coupling == True:
            self._body_oscillator = [self.init_oscillator(omega_sw=3/2*np.pi)] * self._leg_num #6个hip joint
            self._limb_oscillator = [self.init_oscillator(mu=4, omega_sw=3/2*np.pi)] * (self._leg_num - 1) * self.__motor_num #12个limb joint\
            self._oscillator = self._body_oscillator + self._limb_oscillator
        
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
        x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, \
        x7, y7, x8, y8, x9, y9, x10, y10, x11, y11, x12, y12, \
        x13, y13, x14, y14, x15, y15, x16, y16, x17, y17, x18, y18 = pos
        eplison=[]
        sigma=[]
        a=[]
        mu=[]
        beta=[]
        omega_sw = []
        omega_st = []
        omega = []
        bx=[x1, x2, x3, x4, x5, x6]
        by=[y1, y2, y3, y4, y5, y6]
        lx = [x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18]
        ly = [y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18]

        x = bx + lx
        y = by + ly

        r_square = []
        dx=[]
        dy=[]
        # w = [1, 1, 1, 1, 1, 1]

        for i in range(self._leg_num * self.__motor_num):
            eplison.append(self._oscillator[i]._eplison)
            sigma.append(self._oscillator[i]._sigma)
            a.append(self._oscillator[i]._a)
            mu.append(self._oscillator[i]._mu)
            beta.append(self._oscillator[i]._beta)
            omega_sw.append(self._oscillator[i]._omega_sw)

            omega_st.append(((1 - beta[i]) / beta[i]) * omega_sw[i])
            omega.append(omega_st[i] / (np.e ** (-a[i] * x[i]) + 1) + omega_sw[i] / (np.e ** (a[i] * x[i]) + 1))
            r_square.append(x[i]**2 + y[i]**2)

            dx.append(eplison[i] * (mu[i] - r_square[i]) * x[i] - omega[i] * y[i])
            dy.append(sigma[i] * (mu[i] - r_square[i]) * y[i] + omega[i] * x[i])
        
        btheta = np.pi
        # ltheta = np.pi/2
        ltheta = np.pi/3*2
        const = 0
        # const2 = 0
        # 1、 3、 5同相， 2、 4、 6与1反相
        # 0-2-4 || 1-3-5

        # 髋关节间耦合
        for i in range(self._leg_num):
            mark = 2 * i + self._leg_num
            dx[i] += self._coupling_factor * (x[mark]*np.cos(-ltheta) - y[mark]*np.sin(-ltheta) + x[mark + 1]*np.cos(-ltheta) - y[mark + 1]*np.sin(-ltheta))
            dy[i] += self._coupling_factor * (x[mark]*np.sin(-ltheta) + y[mark]*np.cos(-ltheta) + x[mark + 1]*np.sin(-ltheta) + y[mark + 1]*np.cos(-ltheta))
            for j in range(self._leg_num):
                if i == j:
                    continue
                elif (i%2 == 0 and j%2 == 0) or (i%2 == 1 and j%2 == 1):
                    dx[i] += self._coupling_factor * (x[j]*np.cos(0) - y[j]*np.sin(0))
                    dy[i] += self._coupling_factor * (x[j]*np.sin(0) + y[j]*np.cos(0))
                elif i%2 == 0 and j%2 == 1:
                    dx[i] += self._coupling_factor * (x[j]*np.cos(btheta) - y[j]*np.sin(btheta))
                    dy[i] += self._coupling_factor * (x[j]*np.sin(btheta) + y[j]*np.cos(btheta))
                elif i%2 == 1 and j%2 == 0:
                    dx[i] += self._coupling_factor * (x[j]*np.cos(-btheta) - y[j]*np.sin(-btheta))
                    dy[i] += self._coupling_factor * (x[j]*np.sin(-btheta) + y[j]*np.cos(-btheta))
        # dx[0] += self._coupling_factor * (x[1]*np.cos(btheta) - y[1]*np.sin(btheta) + x[2]*np.cos(const) - y[2]*np.sin(const) + x[3]*np.cos(const) - y[3]*np.sin(const)
        #                                     + x[4]*np.cos(btheta) - y[4]*np.sin(btheta) + x[5]*np.cos(const) - y[5]*np.sin(const))
        # dy[0] += self._coupling_factor * (x[1]*np.sin(btheta) + y[1]*np.cos(btheta) + x[2]*np.sin(const) + y[2]*np.cos(const) + x[3]*np.sin(const) + y[3]*np.cos(const)
        #                                     + x[4]*np.sin(btheta) + y[4]*np.cos(btheta) + x[5]*np.sin(const) + y[5]*np.cos(const))
        # dx[1] += self._coupling_factor * (x[0]*np.cos(-btheta) - y[0]*np.sin(-btheta) + x[2]*np.cos(-btheta) - y[2]*np.sin(-btheta) + x[3]*np.cos(-btheta) - y[3]*np.sin(-btheta)
        #                                     + x[4]*np.cos(const) - y[4]*np.sin(const) + x[5]*np.cos(-btheta) - y[5]*np.sin(-btheta))
        # dy[1] += self._coupling_factor * (x[0]*np.sin(-btheta) + y[0]*np.cos(-btheta) + x[2]*np.sin(-btheta) + y[2]*np.cos(-btheta) + x[3]*np.sin(-btheta) + y[3]*np.cos(-btheta)
        #                                     + x[4]*np.sin(const) + y[4]*np.cos(const) + x[5]*np.sin(-btheta) + y[5]*np.cos(-btheta))
        # dx[2] += self._coupling_factor * (x[0]*np.cos(const) - y[0]*np.sin(const) + x[1]*np.cos(btheta) - y[1]*np.sin(btheta) + x[3]*np.cos(const) - y[3]*np.sin(const)
        #                                     + x[4]*np.cos(btheta) - y[4]*np.sin(btheta) + x[5]*np.cos(const) - y[5]*np.sin(const))
        # dy[2] += self._coupling_factor * (x[0]*np.sin(const) + y[0]*np.cos(const) + x[1]*np.sin(btheta) + y[1]*np.cos(btheta) + x[3]*np.sin(const) + y[3]*np.cos(const)
        #                                     + x[4]*np.sin(btheta) + y[4]*np.cos(btheta) + x[5]*np.sin(const) + y[5]*np.cos(const))
        # dx[3] += self._coupling_factor * (x[0]*np.cos(const) - y[0]*np.sin(const) + x[1]*np.cos(btheta) - y[1]*np.sin(btheta) + x[2]*np.cos(const) - y[2]*np.sin(const)
        #                                     + x[4]*np.cos(-btheta) - y[4]*np.sin(-btheta) + x[5]*np.cos(const) - y[5]*np.sin(const))
        # dy[3] += self._coupling_factor * (x[0]*np.sin(const) + y[0]*np.cos(const) + x[1]*np.sin(btheta) + y[1]*np.cos(btheta) + x[2]*np.sin(const) + y[2]*np.cos(const)
        #                                     + x[4]*np.sin(-btheta) + y[4]*np.cos(-btheta) + x[5]*np.sin(const) + y[5]*np.cos(const))
        # dx[4] += self._coupling_factor * (x[0]*np.cos(-btheta) - y[0]*np.sin(-btheta) + x[1]*np.cos(const) - y[1]*np.sin(const) + x[2]*np.cos(-btheta) - y[2]*np.sin(-btheta)
        #                                     + x[3]*np.cos(btheta) - y[3]*np.sin(btheta) + x[5]*np.cos(btheta) - y[5]*np.sin(btheta))
        # dy[4] += self._coupling_factor * (x[0]*np.sin(-btheta) + y[0]*np.cos(-btheta) + x[1]*np.sin(const) + y[1]*np.cos(const) + x[2]*np.sin(-btheta) + y[2]*np.cos(-btheta)
        #                                     + x[3]*np.sin(btheta) + y[3]*np.cos(btheta) + x[5]*np.sin(btheta) + y[5]*np.cos(btheta))
        # dx[5] += self._coupling_factor * (x[0]*np.cos(const) - y[0]*np.sin(const) + x[1]*np.cos(btheta) - y[1]*np.sin(btheta) + x[2]*np.cos(const) - y[2]*np.sin(const)
        #                                     + x[3]*np.cos(const) - y[3]*np.sin(const) + x[4]*np.cos(-btheta) - y[4]*np.sin(-btheta))
        # dy[5] += self._coupling_factor * (x[0]*np.sin(const) + y[0]*np.cos(const) + x[1]*np.sin(btheta) + y[1]*np.cos(btheta) + x[2]*np.sin(const) + y[2]*np.cos(const)
        #                                     + x[3]*np.sin(const) + y[3]*np.cos(const) + x[4]*np.sin(-btheta) + y[4]*np.cos(-btheta))
        # 单条腿关节间耦合
        for i in range(self._leg_num):
            mark = 2 * i + self._leg_num
            dx[mark] += self._coupling_factor * (x[i]*np.cos(ltheta) - x[i]*np.sin(ltheta) + x[mark + 1]*np.cos(0) - y[mark + 1]*np.sin(0))
            dy[mark] += self._coupling_factor * (x[i]*np.sin(ltheta) + x[i]*np.cos(ltheta) + x[mark + 1]*np.sin(0) + y[mark + 1]*np.cos(0))
            dx[mark + 1] += self._coupling_factor * (x[i]*np.cos(ltheta) - x[i]*np.sin(ltheta) + x[mark]*np.cos(0) - y[mark]*np.sin(0))
            dy[mark + 1] += self._coupling_factor * (x[i]*np.sin(ltheta) + x[i]*np.cos(ltheta) + x[mark]*np.sin(0) + y[mark]*np.cos(0))
        # 不同腿关节间耦合

        return dx[0], dy[0], dx[1], dy[1], dx[2], dy[2], dx[3], dy[3], dx[4], dy[4], dx[5], dy[5], \
                dx[6], dy[6], dx[7], dy[7], dx[8], dy[8], dx[9], dy[9], dx[10], dy[10], dx[11], dy[11], \
                dx[12], dy[12], dx[13], dy[13], dx[14], dy[14], dx[15], dy[15], dx[16], dy[16], dx[17], dy[17]
    
    def calculate(self, t):
        '''
        计算hopf振荡器的运动
        '''
        data = integrate.odeint(self.coupling_hopf, self._p0, t)
        return data
    
    def body_show(self, data, t):
        '''
        显示hopf振荡器的运动
        '''
        # plt.plot(t, data[:, 0], label="x1")
        plt.plot(t, data[:, 1], label="y1")
        plt.plot(t, data[:, 3], label="y2")
        plt.plot(t, data[:, 5], label="y3")
        plt.plot(t, data[:, 7], label="y4")
        plt.plot(t, data[:, 9], label="y5")
        plt.plot(t, data[:, 11], label="y6")

        # plt.plot(t, data[:, 2], label="x2")
        
        # plt.plot(t, data[:, 4], label="x3")
        
        # plt.plot(t, data[:, 6], label="x4")

        # plt.plot(t, data[:, 8], label="x5")

        # plt.plot(t, data[:, 10], label="x6")

        plt.legend()
        plt.show()

    def limb_show(self, data, t, leg_id):
        '''
        显示hopf振荡器的运动
        '''
        # plt.plot(t, data[:, 0], label="x1")
        plt.plot(t, data[:, leg_id*2-1], label="hip-y")
        plt.plot(t, data[:, self._leg_num*2 + leg_id*4-3], label="knee-y")
        plt.plot(t, data[:, self._leg_num*2 + leg_id*4-1], label="ankle-y")
        # plt.plot(t, data[:, leg_id*2], label="knee-y")
        # plt.plot(t, data[:, leg_id*2+1], label="ankle-y")
        plt.legend()
        plt.show()

    def TripodGait(self, leg_id, t, data):
        # data = self.calculate(t)

        for i in range(len(data[:, leg_id*2-1])):
            if data[i, self._leg_num*2 + leg_id*4-3] < 0.0:
                data[i, self._leg_num*2 + leg_id*4-3] = 0.0
                data[i, self._leg_num*2 + leg_id*4-1] = 0.0
        # plt.plot(t[5000:6000], data[:, leg_id*2-1][5000:6000], label="hip-y")
        # plt.plot(t[5000:6000], data[:, self._leg_num*2 + leg_id*4-3][5000:6000], label="knee-y")
        # plt.plot(t[5000:6000], data[:, self._leg_num*2 + leg_id*4-1][5000:6000], label="ankle-y")
        # plt.legend()
        # plt.show()
        return data[:, leg_id*2-1], data[:, self._leg_num*2 + leg_id*4-3], data[:, self._leg_num*2 + leg_id*4-1]
    
    def CouplingHpofShow(self, t, data, leg_id):
        dy_hip = np.diff(data[:, leg_id*2-1])/np.diff(t)
        dy_knee = np.diff(data[:, self._leg_num*2 + leg_id*4-3])/np.diff(t)
        dy_ankle = np.diff(data[:, self._leg_num*2 + leg_id*4-1])/np.diff(t)
        plt.plot(t[5000:6000], dy_hip[5000:6000], label="hip-y")
        plt.plot(t[5000:6000], dy_knee[5000:6000], label="knee-y")
        plt.plot(t[5000:6000], dy_ankle[5000:6000], label="ankle-y")
        plt.legend()
        plt.show()
        return dy_hip, dy_knee, dy_ankle
    

if __name__ == "__main__":
    BodyCPGMOdule = PhantomxCPG()

    t = np.arange(0, 100, 0.01)
    data = BodyCPGMOdule.calculate(t)
    BodyCPGMOdule.body_show(data[6000:8000], t[6000:8000])
    # BodyCPGMOdule.limb_show(data[9000:10000], t[9000:10000], 6)
    BodyCPGMOdule.TripodGait(1, t, data)
    BodyCPGMOdule.TripodGait(2, t, data)
    BodyCPGMOdule.TripodGait(3, t, data)
    BodyCPGMOdule.TripodGait(4, t, data)
    BodyCPGMOdule.TripodGait(5, t, data)
    BodyCPGMOdule.TripodGait(6, t, data)
    BodyCPGMOdule.CouplingHpofShow(t, data, 1)