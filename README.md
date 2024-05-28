
# 六足机器人CPG+RL

### environment

```html
强化学习框架 
stable_baselines3

仿真环境 
pybullet

其他库函数 
numpy 
torch 
tensorboard 
matplotlib 
multiprocessing 
os 
datatime
gym
```

### files structure

```html
data
--90deg                         六足机器人髋关节相位差为90度时的18个关节CPG数据
--ResultPictures                各种DEBUG图像
hexapod_34                  hexapod_34六足机器人urdf及mesh等文件
model                       训练过程中储存的中间模型
ObstacleReg                 障碍物urdf及mesh等文件
phantomx_description        phantomx六足机器人urdf及mesh等文件
phantomx_env                训练环境代码（基于stable_baselines3环境框架和gym环境）
--hexapod_env.py                hexapod机器人的gym环境代码
--hexapod.py                    hexapod机器人的pybullet仿真交互代码
--phantomx_env.py               phantomx机器人的gym环境代码
--phantomx.py                   phantomx机器人的pybullet仿真交互代码
--phantomxCPG_env.py            增加在线CPG
--phantomxCPG.py
--phantomxCPG_Trackline_env.py  修改任务目标为跟踪任意速度
--phantomxCPG_Trackline.py
phantomx_tensorboard_test   训练过程中tensorboard log文件
src                         各种训练辅助源代码
--Algorithm                     算法源码
----PPO_Pytorch/PPO.py              github上的PPO代码
----PPOMAX                          取自知乎PPO相关代码                          
--AssistModulesCode
----ActionSelector.py               选取action的动作
----MatPlotAssistor.py              绘制图像
--CentralPatternGenerators      CPG相关源码
----OnlineCPG.py                    在线计算CPG数据
--pybulletTestCode              
zip                         储存历史模型
TrackRLmain.py              跟踪平面任意速度的训练/测试代码
RLmain.py                   跟踪直线的训练/测试代码
```
