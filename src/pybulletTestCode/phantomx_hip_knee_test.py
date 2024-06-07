import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义初始参数
length_1 = 1.0  # 杆1的长度
height = 1.0    # 杆1与地面的距离
length_2 = 0.7  # 杆2的长度
theta_initial = np.radians(30)  # 初始角度，这里设定为30度

# 创建一个3D坐标系
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 计算运动轨迹
theta_range = np.linspace(0, 2 * np.pi, 100)
x_trajectory = length_1 * np.cos(theta_range)
y_trajectory = length_1 * np.sin(theta_range)
z_trajectory = np.ones_like(theta_range) * height

# 绘制杆1的运动轨迹
ax.plot(x_trajectory, y_trajectory, z_trajectory, label='Rod 1')

# 计算杆2的末端运动轨迹
x_end = length_1 * np.cos(theta_range)
y_end = length_1 * np.sin(theta_range)
z_end = np.ones_like(theta_range) * height

# 绘制杆2的末端运动轨迹
ax.plot(x_end, y_end, z_end, label='Rod 2 End Point', linestyle='dashed')

# 设置图形参数
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('Motion Trajectory of Rods')
ax.legend()

# 显示图形
plt.show()
