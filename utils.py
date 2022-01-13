import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os

class FIFO_plot_buffer:
    def __init__(self, options, num_steps):
        exp_num = options.experiment + options.experiment_repeat
        self.save_path = os.path.join(options.experiment_path, exp_num)
        
        self.max_size = options.memory_size
        self.num_steps = num_steps
        
        self.buffer = np.zeros((0,3))
        self.disappear = np.zeros((0,3))
        
        self.zlim = options.target_acc * options.target_pruned

    def add(self, x, y, reward):
        obs = np.array([[x, y, reward]])

        for i in range(self.num_steps):
            self.buffer = np.append(self.buffer, obs, axis=0)
        
        if self.buffer.shape[0] > self.max_size:
            tmp = self.buffer[:(self.buffer.shape[0]-self.max_size)]
            self.disappear = np.append(self.disappear, tmp, axis=0)
            self.disappear = np.unique(self.disappear, axis=0)
            self.buffer = self.buffer[(self.buffer.shape[0]-self.max_size):,:]
        
    def get_buffer_past(self):
        return self.buffer[:-self.num_steps,0].tolist(), self.buffer[:-self.num_steps,1].tolist(), self.buffer[:-self.num_steps,2].tolist()
    
    def get_buffer_current(self):
        return self.buffer[-self.num_steps:,0].tolist(), self.buffer[-self.num_steps:,1].tolist(), self.buffer[-self.num_steps:,2].tolist()
    
    def get_buffer_all(self):
        return self.buffer[:,0].tolist(), self.buffer[:,1].tolist(), self.buffer[:,2].tolist()
    
    def get_disappear(self):
        return self.disappear[:,0].tolist(), self.disappear[:,1].tolist(), self.disappear[:,2].tolist()
    
    def plot_buffer(self, episode, do_print=False, save=False):
        before_x, before_y, before_z = self.get_buffer_past()
        current_x, current_y, current_z = self.get_buffer_current()
                
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(before_x, before_y, before_z, color="green", s=1, marker="o")
        ax.scatter(current_x, current_y, current_z, color="red", s=50, marker="^")
        
        if do_print:
            plt.show()
            plt.close()
            
        if save:
            file_name = str(episode+1) + '_buffer_plot.png'
            file_name = os.path.join(self.save_path, file_name)
            plt.savefig(file_name)
            plt.close()
            print(file_name, "is saved")
        
    def plot_all(self, episode, do_print=False, save=False):
        disappear_x, disappear_y, disappear_z = self.get_disappear()
        before_x, before_y, before_z = self.get_buffer_past()
        current_x, current_y, current_z = self.get_buffer_current()
                
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(before_x, before_y, before_z, color="green", s=1, marker="o")
        ax.scatter(current_x, current_y, current_z, color="red", s=50, marker="^")
        ax.scatter(disappear_x, disappear_y, disappear_z, color="blue", s=1, marker="o")
        
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_zlim(0,self.zlim)
        
        if do_print:
            plt.show()
            plt.close()
            
        if save:
            file_name = str(episode+1) + '_all_plot.png'
            file_name = os.path.join(self.save_path, file_name)
            plt.savefig(file_name)
            plt.close()
            print(file_name, "is saved")
        

              