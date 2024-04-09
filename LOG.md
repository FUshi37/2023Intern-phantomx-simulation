2024.4.3 训练机器人跟随x方向大小为0.4-0.6的任意速度。所有reward系数都是1，yaw角reward factor为1，x、y方向速度reward factor为0.25，训练效果一般，16M后训练曲线没收敛，LOG文件PPO—TrackVelocity-100HZ-xvelP，最后保存模型modelVxP2_16000000。机器人直走过程中还会转。
2024.4.4 继续上一次的训练，训练到了38400000，发现问题是action经常取道边界值。
2024.4.5 修改激活函数为RELU，action取边界值的效果有所改善但还是存在。
2024.4.6 修改了reward_factor，拉大了x方向速度和关节作功的reward factor十倍，y方向速度和yaw角速度拉大了两倍，还是训练不到预期效果。 LOG-9
2024.4.7 继续上次训练到3M，reward收敛到1w8附近，没有收敛到预期值。机器人在走的时候还是会在yaw角方向有速度，且action的取值比较随即。
2024.4.8 reward function增加负向惩罚，修改系数y速度和yaw速度为5，修改激活函数为ELU函数。 LOG-10
2024.4.9 LOG-10训练到19M后收敛到1w左右，然而理想reward应收敛到2w，action取值还较为随机，机器人仍然会有转向（yaw角速度），增加即时速度惩罚项，reward function里的负向惩罚取消，修改激活函数为tanh。 LOG-11

todo
1.尝试用SAC代替PPO，看是否能解决action取值在边界值的问题
2.修改网络结构，现在隐藏层结构为[512, 256, 128]
