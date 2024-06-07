# import numpy as np
# import os

# history_data = np.array([0.29616917,  1.04204406, -0.37516158, -1.01642539,  0.29616917,  1.04204406,
#                     -0.37516158, -1.01642539,  0.29616917,  1.04204406, -0.37516158, -1.01642539,
#                     -1.97166912,  0.36889423, -1.97166912,  0.36889423,  1.86400333, -0.74193836,
#                     1.86400333, -0.74193836, -1.97166912,  0.36889423, -1.97166912,  0.36889423,
#                     1.86400333, -0.74193836,  1.86400333, -0.74193836, -1.97166912,  0.36889423,
#                     -1.97166912, 0.36889423,  1.86400333, -0.74193836,  1.86400333, -0.74193836]).reshape(1, -1)
# print(history_data)
# data = np.array([history_data[0][0], history_data[0][1], history_data[0][2], history_data[0][3]])
# print("data", data)
# path = os.getcwd()
# print(path)

# 读取x_velocity.txt文件，并绘制曲线
import numpy as np
import matplotlib.pyplot as plt
import os

path = os.getcwd()
x_velocity = np.loadtxt(path + "/x_velocity.txt")
# print(x_velocity)
plt.plot(x_velocity)
plt.show()