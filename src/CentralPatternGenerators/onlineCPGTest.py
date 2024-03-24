
from OnlineCPG import OnlinePhantomxCPG
import numpy as np
import matplotlib.pyplot as plt

CPGModule = OnlinePhantomxCPG()
# t = np.arange(0, 100, 0.01)
# data = CPGModule.calculate(t)
# leg1_hip, leg1_knee, leg1_ankle = CPGModule.TripodGait(1, t, data)
# leg2_hip, leg2_knee, leg2_ankle = CPGModule.TripodGait(2, t, data)
# leg3_hip, leg3_knee, leg3_ankle = CPGModule.TripodGait(3, t, data)
# leg4_hip, leg4_knee, leg4_ankle = CPGModule.TripodGait(4, t, data)
# leg5_hip, leg5_knee, leg5_ankle = CPGModule.TripodGait(5, t, data)
# leg6_hip, leg6_knee, leg6_ankle = CPGModule.TripodGait(6, t, data)
# data = np.array([-1.0] * ((6 * 3) * 2)).reshape(1, -1)
# history_data = np.array([-1.0] * ((6 * 3) * 2)).reshape(1, -1)
data = np.array([0.29616917,  1.04204406, -0.37516158, -1.01642539,  0.29616917,  1.04204406,
                    -0.37516158, -1.01642539,  0.29616917,  1.04204406, -0.37516158, -1.01642539,
                    -1.97166912,  0.36889423, -1.97166912,  0.36889423,  1.86400333, -0.74193836,
                    1.86400333, -0.74193836, -1.97166912,  0.36889423, -1.97166912,  0.36889423,
                    1.86400333, -0.74193836,  1.86400333, -0.74193836, -1.97166912,  0.36889423,
                    -1.97166912, 0.36889423,  1.86400333, -0.74193836,  1.86400333, -0.74193836]).reshape(1, -1)
history_data = np.array([0.29616917,  1.04204406, -0.37516158, -1.01642539,  0.29616917,  1.04204406,
                    -0.37516158, -1.01642539,  0.29616917,  1.04204406, -0.37516158, -1.01642539,
                    -1.97166912,  0.36889423, -1.97166912,  0.36889423,  1.86400333, -0.74193836,
                    1.86400333, -0.74193836, -1.97166912,  0.36889423, -1.97166912,  0.36889423,
                    1.86400333, -0.74193836,  1.86400333, -0.74193836, -1.97166912,  0.36889423,
                    -1.97166912, 0.36889423,  1.86400333, -0.74193836,  1.86400333, -0.74193836]).reshape(1, -1)

# t = np.arange(0.01, 0.01 + 0.01, 0.01)
# print(CPGModule._oscillator[0]._mu)
# CPGModule.update_oscillator(oscillatorID=0, mu=4)
# print(CPGModule._oscillator[0]._mu)
initionalpos = []
for i in range(4000) :
    t = np.linspace(i * 0.01 + 20, i * 0.01 + 0.01 + 20, 2)

    data = CPGModule.online_calculate(t, initial_values=history_data[-1, :])
    if i == 2000:
        for j in range(18):
            CPGModule.update_oscillator(oscillatorID=j, mu=4)
    
    if (len(data)==2):
        data = data[1:].reshape(1, -1)

    history_data = np.vstack((history_data, data)) 

    if i == 1500:
        initionalpos = data
        print(initionalpos)


print(len(history_data))
# 绘制历史数据
plt.figure()
plt.plot(history_data[:, 1], label="y1")
plt.plot(history_data[:, 3], label="y2")
plt.plot(history_data[:, 5], label="y3")
plt.plot(history_data[:, 7], label="y4")
plt.plot(history_data[:, 9], label="y5")
plt.plot(history_data[:, 11], label="y6")
plt.legend()
plt.show()
