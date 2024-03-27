# from src.CentralPatternGenerators.phantomx_CPG import HOPF
import numpy as np
from scipy import integrate
import time
import matplotlib.pyplot as plt

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

class OnlinePhantomxCPG(object):
    """description of class"""
    def __init__(self, leg_num = 6, motor_num = 3, coupling_factor = 2, 
                coupling = True, time_step = 0.01):
        self._body_oscillator = []
        self._limb_oscillator = []
        self._leg_num = leg_num
        self.__motor_num = motor_num
        self._coupling_factor = coupling_factor
        self._coupling = coupling
        self._time_step = time_step
        # self._p0 = [-1.0] * ((self._leg_num * self.__motor_num) * 2)
        self._p0 = [0.29616917,  1.04204406, -0.37516158, -1.01642539,  0.29616917,  1.04204406,
                    -0.37516158, -1.01642539,  0.29616917,  1.04204406, -0.37516158, -1.01642539,
                    -1.97166912,  0.36889423, -1.97166912,  0.36889423,  1.86400333, -0.74193836,
                    1.86400333, -0.74193836, -1.97166912,  0.36889423, -1.97166912,  0.36889423,
                    1.86400333, -0.74193836,  1.86400333, -0.74193836, -1.97166912,  0.36889423,
                    -1.97166912, 0.36889423,  1.86400333, -0.74193836,  1.86400333, -0.74193836]
        
        if self._coupling == True:
            self._body_oscillator = [self.init_oscillator(omega_sw=2*3)] * self._leg_num #6个hip joint
            self._limb_oscillator = [self.init_oscillator(mu=4, omega_sw=2*3)] * (self._leg_num - 1) * self.__motor_num #12个limb joint\
            self._oscillator = self._body_oscillator + self._limb_oscillator
        
        self._dx = []
        self._dy = []

    def init_oscillator(self, **kwargs):
        default_params = {
            'eplison': 100,
            'sigma': 100,
            'a': 50,
            'mu': 1,
            'beta': 0.5,
            'omega_sw': 2 * np.pi,
            'time_step': 0.01,
            'coupling': False,
            'coupling_factor': 0.1,
            'y0': [0.1, 0.1]
        }
        for key, value in kwargs.items():
            if key in default_params:
                default_params[key] = value
        return HOPF(**default_params)
    
    def update_oscillator(self, oscillatorID=0, **kwargs):
        default_params = {
            'eplison': 100,
            'sigma': 100,
            'a': 50,
            'mu': 1,
            'beta': 0.5,
            'omega_sw': 2 * np.pi,
            'time_step': 0.01,
            'coupling': False,
            'coupling_factor': 0.1,
            'y0': [0.1, 0.1]
        }
        for key, value in kwargs.items():
            if key in default_params:
                default_params[key] = value
        self._oscillator[oscillatorID] = HOPF(**kwargs)

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
        ltheta = np.pi/2
        # ltheta = np.pi/3*2
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

        # 单条腿关节间耦合
        for i in range(self._leg_num):
            mark = 2 * i + self._leg_num
            dx[mark] += self._coupling_factor * (x[i]*np.cos(ltheta) - x[i]*np.sin(ltheta) + x[mark + 1]*np.cos(0) - y[mark + 1]*np.sin(0))
            dy[mark] += self._coupling_factor * (x[i]*np.sin(ltheta) + x[i]*np.cos(ltheta) + x[mark + 1]*np.sin(0) + y[mark + 1]*np.cos(0))
            dx[mark + 1] += self._coupling_factor * (x[i]*np.cos(ltheta) - x[i]*np.sin(ltheta) + x[mark]*np.cos(0) - y[mark]*np.sin(0))
            dy[mark + 1] += self._coupling_factor * (x[i]*np.sin(ltheta) + x[i]*np.cos(ltheta) + x[mark]*np.sin(0) + y[mark]*np.cos(0))
        # 不同腿关节间耦合
            
        # 将dy[0]保存到文件中
        # with open("output.txt", 'a') as file:
        #     file.write(str(dy[0]))
        #     file.write("\n")
        return dx[0], dy[0], dx[1], dy[1], dx[2], dy[2], dx[3], dy[3], dx[4], dy[4], dx[5], dy[5], \
                dx[6], dy[6], dx[7], dy[7], dx[8], dy[8], dx[9], dy[9], dx[10], dy[10], dx[11], dy[11], \
                dx[12], dy[12], dx[13], dy[13], dx[14], dy[14], dx[15], dy[15], dx[16], dy[16], dx[17], dy[17]
    
    def online_calculate(self, t, initial_values=None, **kwargs):
        if initial_values is None:
            initial_values = self._p0
            print("NONE INITIAL VALUES")
        initial_values = initial_values.tolist()
        # 基于kwargs更新振荡器参数
        for i in range(18):
            # self.update_oscillator(oscillatorID=i, y0 = [initial_values[i], initial_values[i+1]])
            # self.update_oscillator(oscillatorID=i, mu=4, beta=0.3)
            pass

        data = integrate.odeint(self.coupling_hopf, initial_values, t)

        return data
    
    def solve_cpg(self, total_time=10, time_step=0.01):
        t = np.arange(0, total_time, time_step)
        initial_values = [-1.0] * ((self._leg_num * self.__motor_num) * 2)
        history_data = np.zeros((len(t), len(initial_values)))

        for i in range(len(t)):
            if i == 0:
                # 初始条件
                history_data[i, :] = initial_values
            else:
                # 调用 odeint 进行数值求解
                data = integrate.odeint(self.coupling_hopf, history_data[i - 1, :], [t[i - 1], t[i]])
                history_data[i, :] = data[-1, :]
        plt.figure()
        plt.plot(t, history_data[:, 1], label="y1")
        plt.plot(t, history_data[:, 3], label="y2")
        plt.plot(t, history_data[:, 5], label="y3")
        plt.plot(t, history_data[:, 7], label="y4")
        plt.plot(t, history_data[:, 9], label="y5")
        plt.plot(t, history_data[:, 11], label="y6")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.title("CPG Simulation")
        plt.show()

        return t, history_data


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

        # for i in range(len(data[:, leg_id*2-1])):
        if data[self._leg_num*2 + leg_id*4-3] < 0.0:
            data[self._leg_num*2 + leg_id*4-3] = 0.0
            data[self._leg_num*2 + leg_id*4-1] = 0.0
        # plt.plot(t[5000:6000], data[:, leg_id*2-1][5000:6000], label="hip-y")
        # plt.plot(t[5000:6000], data[:, self._leg_num*2 + leg_id*4-3][5000:6000], label="knee-y")
        # plt.plot(t[5000:6000], data[:, self._leg_num*2 + leg_id*4-1][5000:6000], label="ankle-y")
        # plt.legend()
        # plt.show()
        return data[leg_id*2-1], data[self._leg_num*2 + leg_id*4-3], data[self._leg_num*2 + leg_id*4-1]
    
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
    BodyCPGMOdule = OnlinePhantomxCPG()

    # t = np.arange(0, 100, 0.01)
    # data = BodyCPGMOdule.calculate(t)
    # BodyCPGMOdule.body_show(data[6000:8000], t[6000:8000])
    # # BodyCPGMOdule.limb_show(data[9000:10000], t[9000:10000], 6)
    # BodyCPGMOdule.TripodGait(1, t, data)
    # BodyCPGMOdule.TripodGait(2, t, data)
    # BodyCPGMOdule.TripodGait(3, t, data)
    # BodyCPGMOdule.TripodGait(4, t, data)
    # BodyCPGMOdule.TripodGait(5, t, data)
    # BodyCPGMOdule.TripodGait(6, t, data)
    # BodyCPGMOdule.CouplingHpofShow(t, data, 1)
    t, data = BodyCPGMOdule.solve_cpg(total_time=10, time_step=0.01)