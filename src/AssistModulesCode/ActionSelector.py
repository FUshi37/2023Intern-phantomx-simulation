import numpy as np
from src.CentralPatternGenerators.OnlineCPG import OnlinePhantomxCPG

class ActionModuleSelector(object):
    def __init__(self, RLenv=False, model=None, obs=None, action_mode=0, aim_leg_pos=[], index=0, command_start_index=0, TOTAL_TIME=0) :
        self._action_mode = action_mode
        self._RLenv = RLenv
        self._pos_command = aim_leg_pos
        self._obs = obs
        self._index = index
        self._command_start_index = command_start_index

        self._TOTAL_TIME = TOTAL_TIME

        self.CPG = OnlinePhantomxCPG()

    def SelectAction(self, RLenv=False, model=None, obs=None, action_mode=0, aim_leg_pos=[], index=0, command_start_index=0, TOTAL_TIME=0,
                     # params used in online mode #
                    t = [0. ], initial_value = [], data = []
                    ):
        self._action_mode = action_mode
        self._RLenv = RLenv
        self._pos_command = aim_leg_pos
        self._obs = obs
        self._command_start_index = command_start_index
        self._index = index

        self._TOTAL_TIME = TOTAL_TIME

        if RLenv == True:
            action, _states = model.predict(obs)
            return action, _states
        

        # 10 -> online moving mode
        if action_mode == 10:
            # print("data", data)
            leg1_hip, leg1_knee, leg1_ankle = self.CPG.TripodGait(1, t, data)
            leg2_hip, leg2_knee, leg2_ankle = self.CPG.TripodGait(2, t, data)
            leg3_hip, leg3_knee, leg3_ankle = self.CPG.TripodGait(3, t, data)
            leg4_hip, leg4_knee, leg4_ankle = self.CPG.TripodGait(4, t, data)
            leg5_hip, leg5_knee, leg5_ankle = self.CPG.TripodGait(5, t, data)
            leg6_hip, leg6_knee, leg6_ankle = self.CPG.TripodGait(6, t, data)
            LegData = [leg1_hip, leg1_knee, leg1_ankle, leg2_hip, leg2_knee, leg2_ankle, leg3_hip, leg3_knee, leg3_ankle,
                       leg4_hip, leg4_knee, leg4_ankle, leg5_hip, leg5_knee, leg5_ankle, leg6_hip, leg6_knee, leg6_ankle]
            
            action = np.array([leg1_hip, leg1_knee, leg1_ankle, leg2_hip, leg2_knee, leg2_ankle, leg3_hip, leg3_knee, leg3_ankle,
               leg4_hip, leg4_knee, leg4_ankle, leg5_hip, leg5_knee, leg5_ankle, leg6_hip, leg6_knee, leg6_ankle])
            
            return action
        
        
        # 0 -> all motor moved by aim_leg_pos
        if action_mode == 0:
            action = np.array([self._pos_command[0][index+self._command_start_index], self._pos_command[1][index+self._command_start_index], self._pos_command[2][index+self._command_start_index],
                               self._pos_command[3][index+self._command_start_index], self._pos_command[4][index+self._command_start_index], self._pos_command[5][index+self._command_start_index],
                               self._pos_command[6][index+self._command_start_index], self._pos_command[7][index+self._command_start_index], self._pos_command[8][index+self._command_start_index],
                               self._pos_command[9][index+self._command_start_index], self._pos_command[10][index+self._command_start_index], self._pos_command[11][index+self._command_start_index],
                               self._pos_command[12][index+self._command_start_index], self._pos_command[13][index+self._command_start_index], self._pos_command[14][index+self._command_start_index],
                               self._pos_command[15][index+self._command_start_index], self._pos_command[16][index+self._command_start_index], self._pos_command[17][index+self._command_start_index]])
            return action
        
        # -1 -> all motor moved by aim_leg_pos, but only one leg move at a time
        # TOTAL_TIME must be multiple of six
        elif action_mode == -1:
            if index <= TOTAL_TIME/6:
                action = np.array([self._pos_command[0][index+self._command_start_index], self._pos_command[1][index+self._command_start_index], self._pos_command[2][index+self._command_start_index],
                                   0.0, 0.0, 0.3, 0.0, 0.0, 0.3, 0.0, 0.0, 0.3, 0.0, 0.0, 0.3, 0.0, 0.0, 0.3])
            elif TOTAL_TIME/6 < index and index <= TOTAL_TIME/3:
                action = np.array([0.0, 0.0, 0.3, self._pos_command[3][index+self._command_start_index], self._pos_command[4][index+self._command_start_index], self._pos_command[5][index+self._command_start_index],
                                   0.0, 0.0, 0.3, 0.0, 0.0, 0.3, 0.0, 0.0, 0.3, 0.0, 0.0, 0.3])
            elif TOTAL_TIME/3 < index and index <= TOTAL_TIME/2:
                action = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.3, self._pos_command[6][index+self._command_start_index], self._pos_command[7][index+self._command_start_index], self._pos_command[8][index+self._command_start_index],
                                   0.0, 0.0, 0.3, 0.0, 0.0, 0.3, 0.0, 0.0, 0.3])
            elif TOTAL_TIME/2 < index and index <= TOTAL_TIME*2/3:
                action = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.3, 0.0, 0.0, 0.3, self._pos_command[9][index+self._command_start_index], self._pos_command[10][index+self._command_start_index], self._pos_command[11][index+self._command_start_index],
                                   0.0, 0.0, 0.3, 0.0, 0.0, 0.3])
            elif TOTAL_TIME*2/3 < index and index <= TOTAL_TIME*5/6:
                action = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.3, 0.0, 0.0, 0.3,
                                   0.0, 0.0, 0.3, self._pos_command[12][index+self._command_start_index], self._pos_command[13][index+self._command_start_index], self._pos_command[14][index+self._command_start_index],
                                   0.0, 0.0, 0.3])
            else :
                action = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.3, 0.0, 0.0, 0.3, 0.0, 0.0, 0.3, 0.0, 0.0, 0.3,
                                   self._pos_command[15][index+self._command_start_index], self._pos_command[16][index+self._command_start_index], self._pos_command[17][index+self._command_start_index]])
            return action
        
        # 1 -> stay
        elif action_mode == 1:
            action = np.array([0.0, 0.0, 0.3]*6)
            return action