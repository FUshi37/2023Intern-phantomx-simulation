
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
data = np.array([-1.0] * ((6 * 3) * 2)).reshape(1, -1)
history_data = np.array([-1.0] * ((6 * 3) * 2)).reshape(1, -1)
# data = np.array([0.29616917,  1.04204406, -0.37516158, -1.01642539,  0.29616917,  1.04204406,
#                     -0.37516158, -1.01642539,  0.29616917,  1.04204406, -0.37516158, -1.01642539,
#                     -1.97166912,  0.36889423, -1.97166912,  0.36889423,  1.86400333, -0.74193836,
#                     1.86400333, -0.74193836, -1.97166912,  0.36889423, -1.97166912,  0.36889423,
#                     1.86400333, -0.74193836,  1.86400333, -0.74193836, -1.97166912,  0.36889423,
#                     -1.97166912, 0.36889423,  1.86400333, -0.74193836,  1.86400333, -0.74193836]).reshape(1, -1)
history_data = np.array([-1.08321732, -0.00918567,  1.08317018, -0.02320317, -1.08321732, -0.00918567,
                                      1.08317018, -0.02320317, -1.08321732, -0.00918567,  1.08317018, -0.02320317,
                                      0.28975204, -1.98700634,  0.28975204, -1.98700634, -0.13758287,  2.00311752,
                                      -0.13758287,  2.00311752,  0.28975204, -1.98700634,  0.28975204, -1.98700634,
                                      -0.13758287,  2.00311752, -0.13758287,  2.00311752,  0.28975204, -1.98700634,
                                      0.28975204, -1.98700634, -0.13758287,  2.00311752, -0.13758287,  2.00311752]).reshape(1, -1)
initionalpos = np.array([-1.08321732, -0.00918567,  1.08317018, -0.02320317, -1.08321732, -0.00918567,
                                      1.08317018, -0.02320317, -1.08321732, -0.00918567,  1.08317018, -0.02320317,
                                      0.28975204, -1.98700634,  0.28975204, -1.98700634, -0.13758287,  2.00311752,
                                      -0.13758287,  2.00311752,  0.28975204, -1.98700634,  0.28975204, -1.98700634,
                                      -0.13758287,  2.00311752, -0.13758287,  2.00311752,  0.28975204, -1.98700634,
                                      0.28975204, -1.98700634, -0.13758287,  2.00311752, -0.13758287,  2.00311752]).reshape(1, -1)

# t = np.arange(0.01, 0.01 + 0.01, 0.01)
# print(CPGModule._oscillator[0]._mu)
# CPGModule.update_oscillator(oscillatorID=0, mu=4)
# print(CPGModule._oscillator[0]._mu)
Tpos = []
Tpos.append(history_data)
for i in range(1000) :
    t = np.linspace(i * 0.01 + 20, i * 0.01 + 0.01 + 20, 2)

    data = CPGModule.online_calculate(t, initial_values=history_data[-1, :])
    # if i == 2000:
    #     for j in range(18):
    #         CPGModule.update_oscillator(oscillatorID=j, mu=4)
    
    if (len(data)==2):
        data = data[1:].reshape(1, -1)
    # print(type(data))
    history_data = np.vstack((history_data, data)) 

    if i <= 175: 
        # 将一个周期的数据保存到Tpos中
        # print(data)
        Tpos.append(data)
        


    # if (abs(data[0, 1] - initionalpos[0, 1]) < 1e-2).all():
    #     print(i)
        # break
    # if i == 100:
    #     print(data)

    # if i > 1750:
    #     if abs(data[0, 1]) < 1e-2 and len(initionalpos) == 0:
    #         initionalpos = data
    #         print(initionalpos)


# print(len(history_data))
print(len(Tpos))
with open("../../history_data2.txt", "w") as f:
    for i in range(len(Tpos)):
        f.write("[")
        for j in range(36):
            f.write((str(Tpos[i][0, j])) + (" " if j == 35 else ", "))
        f.write("]\n")
# 绘制y历史数据
plt.figure()
plt.plot(history_data[:, 1], label="y1")
plt.plot(history_data[:, 3], label="y2")
plt.plot(history_data[:, 5], label="y3")
plt.plot(history_data[:, 7], label="y4")
plt.plot(history_data[:, 9], label="y5")
plt.plot(history_data[:, 11], label="y6")
plt.legend()
plt.show()
# 绘制x历史数据
plt.figure()
plt.plot(history_data[:, 0], label="x1")
plt.plot(history_data[:, 2], label="x2")
plt.plot(history_data[:, 4], label="x3")
plt.plot(history_data[:, 6], label="x4")
plt.plot(history_data[:, 8], label="x5")
plt.plot(history_data[:, 10], label="x6")
plt.legend()
plt.show()

file_path = "../../history_data2.txt"
with open(file_path, "r") as f:
    lines = f.readlines()
    # lines = np.array(eval(lines))
    for line in lines:
        line = np.array(eval(line)).reshape(1, -1)
        # print(type(line))
        print(line)
    # print(len(lines))