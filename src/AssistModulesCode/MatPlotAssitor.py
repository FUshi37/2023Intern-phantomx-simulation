import matplotlib.pyplot as plt
import os
# Assit main.py save results to /data/ResultPictures

class PlotModuleAssistor(object):
    def __init__(self, data = [], plt_mode = 0) :

        self._data = data
        self._pltMode = plt_mode
        self._path = os.getcwd() 

    def plot(self, data = [], plt_mode = 0, title="default", xlabel="t", ylabel="theta", save_name="default", save_path = "/data/ResultPictures"):
        # plt.title(title)
        # plt.xlabel(xlabel)
        # plt.ylabel(ylabel)
        # plt_mode = 0 -> plot all hip data in one figure 
        if data != []:
            self._data = data
        self._pltMode = plt_mode
        self._path = os.getcwd()
        self._path = self._path + save_path
        plt.figure()
        if self._pltMode == 0:
            self.plotHip("HipJoint", "t", "theta", "HipJoint", self._path)
        # plt_mode = 1 -> plot all knee data in one figure
        elif self._pltMode == 1:
            self.plotKnee("KneeJoint", "t", "theta", "KneeJoint", self._path)
        # plt_mode = 2 -> plot all ankle data in one figure
        elif self._pltMode == 2:
            self.plotAnkle("AnkelJOint", "t", "theta", "AnkleJoint", self._path)
        # plt_mode = 10 -> plot reward data in one figure
        elif self._pltMode == 10:
            self.plotReward("Reward", "t", "reward", "Reward", self._path)
        # plt_mode = 11 -> plot particular reward data in one figure
        elif self._pltMode == 11:
            self.plotReward("Reward", "t", "reward", save_name, self._path)
        
        plt.close()

    def plotHip(self, title="HipJoint", xlabel="t", ylabel="theta", save_name="HipJoint", save_path = "/data/ResultPictures"):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if len(self._data) == 18:
            for i in range(6):
                plt.plot(self._data[3*i], label="hip"+str(i+1))
        else:
            for i in range(len(self._data)):
                plt.plot(self._data[i], label="hip"+str(i+1))
        plt.legend()

        self._save_figure(save_name, save_path)

        

    def plotKnee(self, title="KneeJoint", xlabel="t", ylabel="theta", save_name="KneeJoint", save_path = "/data/ResultPictures"):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if len(self._data) == 18:
            for i in range(6):
                plt.plot(self._data[3*i + 1], label="knee"+str(i+1))
        else:
            for i in range(len(self._data)):
                plt.plot(self._data[i], label="knee"+str(i+1))
        plt.legend()

        self._save_figure(save_name, save_path)

    def plotAnkle(self, title="AnkleJoint", xlabel="t", ylabel="theta", save_name="AnkleJoint", save_path = "/data/ResultPictures"):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if len(self._data) == 18:
            for i in range(6):
                plt.plot(self._data[3*i + 1], label="ankle"+str(i+1))
        else:
            for i in range(len(self._data)):
                plt.plot(self._data[i], label="ankle"+str(i+1))
        plt.legend()

        self._save_figure(save_name, save_path)

    def plotSingleLeg(self, legID=1, title="SingleLeg", xlabel="t", ylabel="theta", save_name="SingleLeg", save_path = "/data/ResultPictures"):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if len(self._data) == 18:
            for i in range(3):
                plt.plot(self._data[3*(legID-1) + i], label="leg" + str(legID) + str(i+1))
        else:
            for i in range(len(self._data)):
                plt.plot(self._data[i], label="leg" + str(legID) + str(i+1))
        plt.legend()

        self._save_figure(save_name, save_path)

    def plotReward(self, title="Reward", xlabel="t", ylabel="reward", save_name="Reward", save_path = "/data/ResultPictures"):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.plot(self._data, label="Reward")
        plt.legend()

        self._save_figure(save_name, save_path)

    # def plotMultipleLeg(self, legID=[1, 2, 3], title="MultipleLeg", xlabel="t", ylabel="theta", save_name="MultipleLeg", save_path = "/data/ResultPictures"):
    #     plt.title(title)
    #     plt.xlabel(xlabel)
    #     plt.ylabel(ylabel)

    #     if len(self._data) == 18:
    #         for i in range(len(legID)):
    #             for j in range(3):
    #                 plt.plot(self._data[3*(legID[i]-1) + j], label="ankle"+str(j+1))
    #     else:
    #         for i in range(len(self._data)):
    #             plt.plot(self._data[i], label="ankle"+str(i+1))
    #     plt.legend()

    #     self._save_figure(save_name, save_path)

    def _save_figure(self, save_name, save_path):
        save_path = save_path + "/" + save_name + ".png"
        plt.savefig(save_path)
        # plt.show()
        plt.close('all')
        
    