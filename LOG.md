2024.4.3 训练机器人跟随x方向大小为0.4-0.6的任意速度。所有reward系数都是1，yaw角reward factor为1，x、y方向速度reward factor为0.25，训练效果一般，16M后训练曲线没收敛，LOG文件PPO—TrackVelocity-100HZ-xvelP，最后保存模型modelVxP2_16000000。机器人直走过程中还会转。
2024.4.4 继续上一次的训练，训练到了38400000，发现问题是action经常取道边界值。
2024.4.5 修改激活函数为RELU，action取边界值的效果有所改善但还是存在。
2024.4.6 修改了reward_factor，拉大了x方向速度和关节作功的reward factor十倍，y方向速度和yaw角速度拉大了两倍，还是训练不到预期效果。 LOG-9
2024.4.7 继续上次训练到3M，reward收敛到1w8附近，没有收敛到预期值。机器人在走的时候还是会在yaw角方向有速度，且action的取值比较随即。
2024.4.8 reward function增加负向惩罚，修改系数y速度和yaw速度为5，修改激活函数为ELU函数。 LOG-10
2024.4.9 LOG-10训练到19M后收敛到1w左右，然而理想reward应收敛到2w，action取值还较为随机，机器人仍然会有转向（yaw角速度），增加即时速度惩罚项，reward function里的负向惩罚取消，修改激活函数为tanh。 LOG-11
2024.4.10 训练局部收敛（假收敛），收敛到1w8附近，期望值为2w8左右，action取值还是较为随机。尝试用SAC训练，action取值随机的情况有所改善。 LOG-12
2024.4.11 发现yaw_average_velocity有BUG，修正后重新用PPO、Tanh训练，，训练了13M，还位收敛但已经有收敛的趋势、未收敛13M的reward值约为1.8w，理想reward为2w以上。 LOG-13 modelVxPPPOaction
2024.4.12 增加action rate惩罚项，对action的变化值做惩罚，训练效果一般，未收敛到预期值，action取值有收缩趋势但仍然较为随机。 LOG-14 modelVxPPPOA
2024.4.13 修改reward function为线性函数，修改网络结构为[128 256 512 256 128]，缩短CPG周前为100，修改单次训练episode为300，修改leaky_relu作为激活函数。基本收敛，action取值出现取值在边界值的情况，且action取值还是较为随机。 LOG-15 modelVxLeakyRelu
2024.4.14 reward增加即时速度惩罚和action rate惩罚。未收敛到目标值，action取值较为随机。 LOG-16 modelVxLeakyRelu2
2024.4.15 reward去掉了即时速度惩罚，修改了一下PPO参数设置，尝试用VecNormalize，但是由于一直存在读取模型和环境测试时无法直接调用环境里的成员和函数遂放弃使用。训练结果一般，未收敛到目标reward值，action取值还较为随机。 LOG-17 modelVxLeakyRelu3
2024.4.16 在17的基础上增加了Vecnormalize，效果和每加差不多。 LOG-18 modelVxLeakyRelu4
2024.4.17 在17的基础上去掉了action rate，收敛到的值没什么变化。LOG-19 modelVxLeakyRelu5
2024.4.18 调节了reward系数，前10步energy除以10 LOG-20 6
2024.4.19 调节了reward function的系数，调大了10倍，训练了一晚上貌似收敛到了负值，action收敛接近0 LOG-21 7


todo
1.尝试用SAC代替PPO，看是否能解决action取值在边界值的问题 ok
2.修改网络结构，现在隐藏层结构为[512, 256, 128] ok
3.接入rsl-rl的ppo代替stable_baselines3的ppo
4.奖励项增加action rate，对action的变化值做惩罚，尝试是否能让action收敛 ok
5.改reward函数 ok
6.减obs 
7.改相位 
8.缩短周期 ok
9.跟随方向 
10.debug梯度测试判断训练效果，避免梯度爆炸，clip反向传播的范围
11.leaky_relu作激活函数 ok
12.修改网络结构为菱形[128, 256, 512, 256, 128] ok
13.obs数量级 ok
