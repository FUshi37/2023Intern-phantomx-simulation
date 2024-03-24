from phantomx_env.envs.phantomx import Phantomx
import pybullet
from pybullet_utils import bullet_client as bc
import pybullet_data
import time

def map_range(value, from_min, from_max, to_min, to_max):
        from_span = from_max - from_min
        to_span = to_max - to_min
        value_scaled = float(value - from_min) / float(from_span)
        return to_min + (value_scaled * to_span)

_pybullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
# _pybullet_client = bc.BulletClient()
_pybullet_client.configureDebugVisualizer(_pybullet_client.COV_ENABLE_RENDERING, 0)

_pybullet_client.resetSimulation()
_pybullet_client.setGravity(0, 0, -10)
_ground_id = _pybullet_client.loadURDF("%s/plane.urdf" % pybullet_data.getDataPath())

_phantomx_urdf_root = "/home/yangzhe/Intern/simulation/RL_phantomx_pybullet/phantomx_description"

phantomx = Phantomx(pybullet_client=_pybullet_client, urdf_root=_phantomx_urdf_root)

# _pybullet_client.resetDebugVisualizerCamera(1.0, 0, -30, [0, 0, 0])

_pybullet_client.configureDebugVisualizer(_pybullet_client.COV_ENABLE_RENDERING, 1)

motor_command = [0.0, 1.2, -1.2, 0.0, 1.2, -1.2, 0.0, 1.2, -1.2, 0.0, 1.2, -1.2, 0.0, 1.2, -1.2, 0.0, 1.2, -1.2]
# motor_command = [0.0, 0.12, 0.0, 0.0, 0.12, 0.0, 0.0, 0.12, 0.0, 0.0, 0.12, 0.0, 0.0, 0.12, 0.0, 0.0, 0.12, 0.0]

print("MAP-RNGE",map_range(0.5, -1.0, 1.0, -1.5708, 0.6179939))

while (0):
    pybullet.stepSimulation()
    base_position = phantomx.GetBasePosition()
    base_orientation = phantomx.GetBaseOrientation()
    base_height = phantomx.GetBaseHigh()
    motor_angle = phantomx.GetTrueMotorAngles()
    print("base_position=", base_position)
    print("base_orientation=", base_orientation)
    print("base_height=", base_height)
    print("motor_angle=", motor_angle)
    phantomx.ApplyAction(motor_command)
    print("motor_list=", phantomx._motor_id_list)
    print("joint_list=", phantomx._joint_name_to_id)
    print("joint_leg_id", phantomx.joint_leg_joint_id)
    # time.sleep(0.001)