import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import time

class HOPF(object):
    def __init__(self, eplison=100, sigma=100, a=50, 
                 mu=1, beta=0.5, omega_sw=2*np.pi,
                 time_step=0.01, coupling = False, 
                 coupling_factor = 0.1, y0=[0.1, 0.1]) :
        '''
        初始化参数
        '''
        self._time_step = time_step
        self._eplison = eplison
        self._sigma = sigma
        self._a = a
        self._mu = mu
        self._beta = beta
        self._omega_sw = omega_sw
        self._p0 = y0
        # 耦合项
        self._coupling = coupling
        self._coupling_factor = 0.1

    def hopf(self, pos, time_step):
        '''
        hopf振荡器数学模型
        '''
        x, y = pos
        eplison, sigma, a, mu, beta, omega_sw = self._get_param()
        r_square = x**2 + y**2

        omega_st = ((1 - beta) / beta) * self._omega_sw
        omega = omega_st / (np.e ** (-a * y) + 1) + omega_sw / (np.e ** (a * y) + 1)

        # if self._coupling == True:
        #     pass

        dx = eplison * (mu - r_square) * x - omega * y
        dy = sigma * (mu - r_square) * y + omega * x

        return dx, dy
    
    def _get_param(self):
        '''
        获取参数
        '''
        return self._eplison, self._sigma, self._a, self._mu, self._beta, self._omega_sw
    
    def calculate(self, t):
        '''
        计算hopf振荡器的运动
        '''
        data = integrate.odeint(self.hopf, self._p0, t)
        return data
    
    def show(self, data, t):
        '''
        显示结果
        '''
        plt.plot(t, data[:, 0], label="x")
        plt.plot(t, data[:, 1], label="y")
        plt.legend()
        plt.show()

    def calandshow(self):
        '''
        绘制图像
        '''
        t = np.arange(0, 10, self._time_step)
        t0 = time.time()
        data = self.calculate(t)
        t1 = time.time()
        fig1 = plt.figure()
        print("time cost:", t1 - t0)
        plt.plot(t, data[:, 0], label="x")
        plt.plot(t, data[:, 1], label="y")
        plt.legend()
        plt.show()

class SingleLegCPG(object):
    def __init__(self, motor_num=4, coupling_factor=0.35,
                 coupling=True, time_step=0.01) :
        '''
        初始化参数
        '''
        self._motor_num = motor_num
        self._coupling_factor = coupling_factor
        self._coupling = coupling
        self._time_step = time_step
        self._oscillator = []
        self._p0 = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        self._oscillator.append(HOPF(omega_sw=1*np.pi))
        self._oscillator.append(HOPF(omega_sw=2*np.pi))
        self._oscillator.append(HOPF(omega_sw=2*np.pi))
        self._oscillator.append(HOPF(omega_sw=2*np.pi)) # 用于调节2、3的相位

    def coupling_hopf(self, pos, time_step):
        '''
        hopf振荡器数学模型
        '''
        x1, y1, x2, y2, x3, y3, x4, y4 = pos
        eplison=[]
        sigma=[]
        a=[]
        mu=[]
        beta=[]
        omega_sw = []
        omega_st = []
        omega = []
        x=[x1, x2, x3, x4]
        y=[y1, y2, y3, y4]
        r_square = []
        dx=[]
        dy=[]
        w = [1, 1, 1, 1]

        for i in range(self._motor_num):
            eplison.append(self._oscillator[i]._eplison)
            sigma.append(self._oscillator[i]._sigma)
            a.append(self._oscillator[i]._a)
            mu.append(self._oscillator[i]._mu)
            beta.append(self._oscillator[i]._beta)
            omega_sw.append(self._oscillator[i]._omega_sw)
        
        for i in range(self._motor_num):
            omega_st.append(((1 - beta[i]) / beta[i]) * omega_sw[i])
            omega.append(omega_st[i] / (np.e ** (-a[i] * x[i]) + 1) + omega_sw[i] / (np.e ** (a[i] * x[i]) + 1))
            r_square.append(x[i]**2 + y[i]**2)
        
        for i in range(self._motor_num):
            # dx.append(eplison[i] * (mu[i] - r_square[i]) * x[i] - omega[i] * y[i])
            # dy.append(sigma[i] * (mu[i] - r_square[i]) * y[i] + omega[i] * x[i])
            dx.append(eplison[i] * (mu[i] - r_square[i]) * x[i] - 2*np.pi*w[i] * y[i])
            dy.append(sigma[i] * (mu[i] - r_square[i]) * y[i] + 2*np.pi*w[i] * x[i])

        delta_theta = np.pi/4 * 3
        dx[0] += self._coupling_factor * (x[1]*np.cos(delta_theta) - y[1]*np.sin(delta_theta) + x[2]*np.cos(delta_theta) - y[2]*np.sin(delta_theta))
        dy[0] += self._coupling_factor * (x[1]*np.sin(delta_theta) + y[1]*np.cos(delta_theta) + x[2]*np.sin(delta_theta) + y[2]*np.cos(delta_theta))
        dx[1] += self._coupling_factor * (x[0]*np.cos(-delta_theta) - y[0]*np.sin(-delta_theta) + x[2]*np.cos(0) - y[2]*np.sin(0))
        dy[1] += self._coupling_factor * (x[0]*np.sin(-delta_theta) + y[0]*np.cos(-delta_theta) + x[2]*np.sin(0) + y[2]*np.cos(0))
        dx[2] += self._coupling_factor * (x[0]*np.cos(-delta_theta) - y[0]*np.sin(-delta_theta) + x[1]*np.cos(0) - y[1]*np.sin(0))
        dy[2] += self._coupling_factor * (x[0]*np.sin(-delta_theta) + y[0]*np.cos(-delta_theta) + x[1]*np.sin(0) + y[1]*np.cos(0))

        # theta = np.pi - np.pi/3 + np.pi/12
        # dx[3] += self._coupling_factor * (x[1]*np.cos(theta) - y[1]*np.sin(theta) + x[2]*np.cos(0) - y[2]*np.sin(0))
        # dy[3] += self._coupling_factor * (x[1]*np.sin(theta) + y[1]*np.cos(theta) + x[2]*np.sin(0) + y[2]*np.cos(0))
        # dx[2] += self._coupling_factor * (x[1]*np.cos(theta) - y[1]*np.sin(theta) + x[3]*np.cos(0) - y[3]*np.sin(0))
        # dy[2] += self._coupling_factor * (x[1]*np.sin(theta) + y[1]*np.cos(theta) + x[3]*np.sin(0) + y[3]*np.cos(0))

        # dy[0] += self._coupling_factor * ((y[1]*np.cos(np.pi) + x[1]*np.sin(np.pi)) + (y[2]*np.cos(np.pi) + x[2]*np.sin(np.pi)))
        # dy[1] += self._coupling_factor * (y[0]*np.cos(-np.pi) + x[0]*np.sin(-np.pi) + (y[2]*np.cos(0) + x[2]*np.sin(0)))
        # dy[2] += self._coupling_factor * (y[0]*np.cos(-np.pi) + x[0]*np.sin(-np.pi) + (y[1]*np.cos(0) + x[1]*np.sin(0)))

        return dx[0], dy[0], dx[1], dy[1], dx[2], dy[2], dx[3], dy[3]
    
    def calculate(self, t):
        '''
        计算hopf振荡器的运动
        '''
        data = integrate.odeint(self.coupling_hopf, self._p0, t)
        return data
    
    def show(self, data, t):
        '''
        显示结果
        '''
        # plt.plot(t, data[:, 0], label="x1")
        plt.plot(t, data[:, 1], label="y1")
        # plt.plot(t, data[:, 2], label="x2")
        plt.plot(t, data[:, 3], label="y2")
        # plt.plot(t, data[:, 4], label="x3")
        plt.plot(t, data[:, 5], label="y3")
        # plt.plot(t, data[:, 6], label="x4")
        # plt.plot(t, data[:, 7], label="y4")
        plt.legend()
        plt.show()

    def draw_circle(self, data):
        plt.plot(data[:, 0], data[:, 1], label="id=1")
        # plt.plot(data[:, 2], data[:, 3], label="id=2")
        # plt.plot(data[:, 4], data[:, 5], label="id=3")
        plt.legend()
        plt.show()

    def getYdata(self, tmin=0, tmax=100, ratio=0.01):
        t = np.arange(tmin, tmax, ratio)
        data = self.calculate(t)
        return data[:, 1], data[:, 5], data[:, 7]

    def TripodGait(self, tmin=0, tmax=100, ratio=0.01):
        t = np.arange(tmin, tmax, ratio)
        data = self.calculate(t)
        for i in range(len(data[:, 1])):
            delta_data = 0
            if i != 0:
                delta_data = data[i, 1] - data[i - 1, 1]
            if delta_data < 0:
                data[i, 5] = 0
                data[i, 7] = 0
        
        plt.plot(t[8000:10000], data[:, 1][8000:10000], label="y1")
        plt.plot(t[8000:10000], data[:, 5][8000:10000], label="y3")
        plt.legend()
        plt.show()
        return data[:, 1], data[:, 3], data[:, 5], data[:, 7]
    
    def ConvertCPGToJoint(self, tmin=0, tmax=100, ratio=0.01):
        t = np.arange(tmin, tmax, ratio)
        data = self.calculate(t)
        for i in range(len(data[:, 1])):
            delta_data = 0
            if i != 0:
                delta_data = data[i, 1] - data[i - 1, 1]
            if delta_data < 0:
                data[i, 5] = 1
                data[i, 7] = 1
        
        plt.plot(t[8000:10000], data[:, 1][8000:10000], label="y1")
        plt.plot(t[8000:10000], data[:, 5][8000:10000], label="y3")
        plt.legend()
        plt.show()
        return data[:, 1], data[:, 5], data[:, 7]



if __name__ == "__main__" :
    Hopf = HOPF()
    Leg1 = SingleLegCPG()
    # print(Leg1._oscillator)
    t = np.arange(0, 100, 0.01)
    data = Leg1.calculate(t)
    data_h = Hopf.calculate(t)
    # print(data_h)
    # print(len(data))

    Leg1.show(data[8000:10000], t[8000:10000])
    # Leg1.draw_circle(data)
    # Leg1.TripodGait()
    Leg1.ConvertCPGToJoint()

    # Hopf.show(data_h, t) 
    # Hopf.calandshow()
