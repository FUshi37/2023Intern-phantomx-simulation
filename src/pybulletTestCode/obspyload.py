import pybullet as p
import pybullet_data as pd
import time

p.connect(p.GUI)
p.setGravity(0,0,-9.8)
p.setAdditionalSearchPath(pd.getDataPath())
floor = p.loadURDF("plane.urdf")
startPos = [0,0,0.12]
startOrientation = [0.707, 0, 0., 0.707]

shift = [0, 0, 0]
scale = [1, 1, 1]

# robot = p.loadURDF("../../phantomx_description/phantomx.urdf", startPos, useFixedBase=True)

visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                    fileName="sphere_smooth.obj",
                                    rgbaColor=[1, 1, 1, 1],
                                    specularColor=[0.4, 0.4, 0],
                                    visualFramePosition=[0, 0, 0],
                                    meshScale=scale,
                                    )


collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                        fileName="sphere_smooth.obj",
                                        collisionFramePosition=[0, 0, 0],
                                        meshScale=scale
                                        )
p.createMultiBody(baseMass=0,
                    baseInertialFramePosition=[0, 0, 0],
                    baseCollisionShapeIndex=collisionShapeId,
                    baseVisualShapeIndex=visualShapeId,
                    basePosition=[0, 0, 0],
                    useMaximalCoordinates=True)

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

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
