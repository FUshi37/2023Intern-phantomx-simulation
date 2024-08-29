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
2024.4.19 obs去掉了关节力矩，调节了reward function的系数，调大了10倍，训练了一晚上貌似收敛到了负值，action收敛接近0 LOG-21 7
2024.4.21 继续上次训练，效果没有提升。 LOG-21 7
2024.4.22 增加角度penalty，训练效果一般。 LOG-22 8 发现penalty函数有BUG，penalty的取值没有先取abs。
此次训练系数：
x_velocity_weight = 26.0,# 26
y_velocity_weight = 2.00,# 2
yaw_velocity_weight = 2.0,# 2
height_weight = 5.0,# 5
shakevel_weight = 5.0,# 5
energy_weight = 100.0,# 100

# intime_x_velocity = 3.0,

# intime_y_velocity = 3.0,

# intime_yaw_velocity = 3.0,

action_rate = 1.0, # 1
yaw_weight = 3.0, # 3
2024.4.23 修改BUG后在LOG-22的基础上训练，训练了14M收敛到2000，训练初始值在-8000左右，特别大需要调整。 LOG-23 9
2024.4.29 obs去掉关节角速度，给reward各项增加下届值-5，对累加的reward除以300，一个epoch reward的值收缩到-20-30之间，loss数量级大幅度下降。 发现obs角速度处多除以了一个10，修改后训练，14M未收敛。但发现正常走的情况下reward振荡非常严重，发现是角速度相关奖励项和惩罚项的问题，对于跟姿态角速度相关的奖励都除以了10。 LOG-24 10
2024.4.30 调整角速度reward，在验证时对action增加正态分布噪声，不再在线计算cpg数据，改为直接读取一个周期的数值。 在env中去掉时间对齐的操作，训练速度提升近一倍。 一个epoch训练长度为94*4。 训练到18M收敛到20左右，期望收敛值为31左右，不确定收敛与否，但是曲线变平滑了。 LOG-25 11
2024.5.1 PPO config ent_coef设为0.01而不是0，训练先上升后下降，最大值也没有收敛到理想值。 LOG-26 12
修改网络结构256, 128，修改激活函数为Tanh，训练还是先上升后下降，最大值也没有收敛到理想值。LOG-27 13
2024.5.2 修改跟随x方向固定速度0.5，训练还是先上升后下降，最大值也没有收敛到理想值。 LOG-28 14. 此处推测为ent_coef的问题
2024.5.3 改了obs超范围的问题，网络结构改为128 256 512 256 128，训练还是先上升后下降。 LOG-29 15
2024.5.4 发现一直没加change_dynamics函数，即没加摩擦力，修改仿真帧率为1000HZ，控制帧率为100HZ，跟随固定速度0.24，修改reward函数为二次型，调整reward计算时height和shakevel的reward大小，修改energy weight为50。训练到24M未收敛。 LOG-30 16 训练到80M还没收敛，上升速度非常慢，停止训练。
2024.5.6 修改控制帧率1000/3HZ，摩擦力lateralFriction为0.8，平均速度为0.63左右，一个epoch为94*5，调整reward回到线性函数，因为在LOG30中收敛速度太慢。调整了部分obs范围，避免超出bound。激活函数LeakyRelu，发现Tanh训练出来的的action取值几乎在边界。ent_coef取值0。
LOG-31 17
是否要用ent_coef？
2024.5.7 修改网络结构，reward增加is fallen惩罚，如果is fallen返回-700/94/5。跟随速度设定为0.11，效果有所提升但是没有收敛到期望值。 LOG-32 18
2024.5.8 ent_coef改为0.01，训练上升速度非常慢。LOG-33 19
2024.5.10 ent_coef改回0.00，和LOG-32相同，因为LOG-32的模型没保存 LOG-34 20
2024.5.12 改为力矩控制替代为电机位置控制，参考CPG-RL项目里的力矩控制和电机力矩控制器，跟随速度0.60，能量损耗函数的计算也相应的修改，同时修改奖励全重。 LOG-35 21
2024.5.13 摩擦力修改为1，ACTION_REPEATE改为10，修改奖励函数为带平顶的线性函数，这是去针对CPG模式下速度振幅较大采取的措施，速度窗口改为94即一个周期以减少一定程度上的振荡。训练时速度给错成了0.6，应该给0.15，训练收敛但action取值有靠近边界的趋势。 LOG-36 22
2024.5.20 在22的基础上给速度0.15，reward收敛有很好的效果，不过并不是想要的CPG运动情形，增大了action rate惩罚项系数10倍，训练到40M训练曲线缓慢上升还未收敛。 LOG-37 23
2024.5.23 两次调大action rate的reward，还未收敛到目标值。 LOG-37 23
2024.5.24 计划直接重新训练一次action rate系数大的模型，修改为100，同时修改一下机械模型映射系数，把action4-6位的映射乘以-1，效果很一般，不如人LOG-37，把本次训练记录删除。
2024.5.25 action_rate系数直接改大到50重新训练，未收敛到目标值。 LOG-38 24
2024.5.26 ent_coef修改为0.01，还未收敛。 LOG-39 25
2024.5.27 发现之前的平均速度计算部分代码有BUG，修改后将reward函数改回线性函数。未收敛到目标值，action取值较为随机。  LOG-40 26
2024.5.28 在LOG-40的基础上修改ent_coef为0.019重新训练，action取值在边界值。 LOG-41 27 TrackTest新增区分绘制髋关节action图像和膝关节action图像，便于DEBUG
2024.6.1 修改激活函数为Tanh，未收敛到预期值。LOG-42 28。发现在计算reward时，计算energy reward的值为正值，符号取反了，修改bug。
2024.6.2 修改bug后训练，调整reward function系数，调大各种reward系数；observation增加触地的踝关节布尔值，进行训练，未收敛预期值。 LOG-43 29。
2024.6.4 调大energy reward系数训练，为收敛到预期值。 LOG-44 30。
2024.6.5/6 尝试用ppo_continuous_action的PPO代码进行训练，跑通了训练代码，但存在模型加载检验和继续训练的问题。
2024.6.7 取消对reward负值的限副，修改了一下ppo_config，训练开始时最小值大致为-100左右，训练了12M，有收敛趋势，目前值在16左右，期望值为24左右。 机器人表现正常了许多，action取值变得比较有规律。 LOG-45 31。

todo
1.尝试用SAC代替PPO，看是否能解决action取值在边界值的问题 ok
2.修改网络结构，现在隐藏层结构为[512, 256, 128] ok
3.接入rsl-rl的ppo代替stable_baselines3的ppo
4.奖励项增加action rate，对action的变化值做惩罚，尝试是否能让action收敛 ok
5.改reward函数 ok
6.减obs ok
7.改相位
8.缩短周期 ok 100步一个周期
9.跟随方向
10.debug梯度测试判断训练效果，避免梯度爆炸，clip反向传播的范围
11.leaky_relu作激活函数 ok
12.修改网络结构为菱形[128, 256, 512, 256, 128] ok
13.obs数量级 ok

14.直接读一个周期的CPG数据不再在线计算，这样改变相位也更加便利。ok
15.测试reward时加高斯分布的扰动，查看reward曲线
16.去掉obs中的角度角速度 ok
17.调整reward让最开始训练时ep_rew_mean绝对值不要太大

18.设定action初始值
19.控制帧率为仿真帧率的1/2or1/3 ok
20.记录observation是否超出bound ok

21.前240个时间步长不考虑速度差计算跟随速度，因为在加速，直接拉满（或者改成加速reward拉满）

TODO

跟随x方向固定速度，无action惩罚，奖励函数为线性函数，trackvel_3

1 action_rate改为1.0*DT或0.5*DT后查看效果

2 奖励函数改为指数函数查看效果&y、z方向奖励改为用reward_function

3 任务跟随随机速度，增加网络结构

4 改变任务目标，改为到点
