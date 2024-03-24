import pybullet as p
import pybullet_data as pd
import time

p.connect(p.GUI)
p.setGravity(0,0,-9.8)
p.setAdditionalSearchPath(pd.getDataPath())
floor = p.loadURDF("plane.urdf")
startPos = [0,0,0.12]
startOrientation = [0.707, 0, 0., 0.707]
# robot = p.loadURDF("../../hexapod_34/urdf/hexapod_34.urdf", startPos)
robot = p.loadURDF("../../ObstacleReg/urdf/ObstacleReg.urdf", startPos, startOrientation, useFixedBase=True)

# numJoints = p.getNumJoints(robot)
# p.changeVisualShape(robot,-1,rgbaColor=[1,1,1,1])
# for j in range (numJoints):
# 	p.changeVisualShape(robot,j,rgbaColor=[1,1,1,1])
# 	force=200
# 	pos=0
# 	p.setJointMotorControl2(robot,j,p.POSITION_CONTROL,pos,force=force)
# dt = 1./240.
# p.setTimeStep(dt)

while (1):
  p.stepSimulation()
  time.sleep(0.01)



# import pybullet
# import pybullet_data
# import time

# if __name__ == 'main':
#     # Initialize the PyBullet environment
#     pybullet.connect(pybullet.GUI)
#     pybullet.setGravity(0, 0, -9.81)
#     pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

#     # pybullet.setRealTimeSimulation(1)

#     floor = pybullet.loadURDF('plane.urdf')
#     stratPos = [0, 0, 1]

#     # Load the URDF file
#     # pybullet.loadURDF('/yangzhe/Intern/simulation/RL_phantomx_pybullet/phantomx_description/phantomx.urdf', stratPos)
#     # pybullet.loadURDF('phantomx_description/phantomx_pybullet.urdf', stratPos)
#     pybullet.loadURDF('phantomx_description/new_hexapod.urdf.urdf', stratPos)

#     # Run the simulation
#     while (1):
#         print('Simulation running')
#         pybullet.stepSimulation()
#         time.sleep(1./240.)