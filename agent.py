import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from scipy import stats
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

class Actor(nn.Module):
    def __init__(self, len_states, num_actions, hidden=1000, middle=1000):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(len_states, hidden)
        self.fc2 = nn.Linear(hidden, middle)
        #self.fc3 = nn.Linear(middle, middle)
        self.fc4 = nn.Linear(middle, hidden)
        self.fc5 = nn.Linear(hidden, (num_actions + 1))
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        #out = self.fc3(out)
        #out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        out = self.sigmoid(out)
        return out

class Critic(nn.Module):
    def __init__(self, len_states, num_actions, hidden=1000, middle=1000):
        super(Critic, self).__init__()
        self.fc1_1 = nn.Linear(len_states, hidden)
        self.fc1_2 = nn.Linear((num_actions + 1), hidden)
        self.fc2 = nn.Linear(hidden, middle)
        #self.fc3 = nn.Linear(middle, middle)
        self.fc4 = nn.Linear(middle, hidden)
        self.fc5 = nn.Linear(hidden, 1)
        self.relu = nn.LeakyReLU()
        
    def forward(self, state_action_list):
        state, action = state_action_list
        out = self.fc1_1(state) + self.fc1_2(action)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        #out = self.fc3(out)
        #out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        return out

class DDPG_Agent:
    def __init__(self, len_states, options):
        ## ---- save path ---- ##
        exp_num = options.experiment + options.experiment_repeat
        self.save_path = os.path.join(options.experiment_path, exp_num)
        
        ## ---- set basic information ---- ##
        self.len_states = len_states
        if options.method == 'both':
            self.num_actions = 3
        elif options.method == 'rectangular':
            self.num_actions = 2
        else: # channel
            self.num_actions = 1
        
        ## ---- build network ---- ##
        self.device = options.device
        
        self.actor = Actor(self.len_states, self.num_actions).to(self.device)
        self.actor_target = Actor(self.len_states, self.num_actions).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=options.lr)
        
        self.critic = Critic(self.len_states, self.num_actions).to(self.device)
        self.critic_target = Critic(self.len_states, self.num_actions).to(self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=options.lr)
        
        ## ---- instantiate memory ---- ##
        self.batch_size = options.batch_size
        self.memory = Replay_Memory(options.memory_size)
        
        self.warmup = options.warmup
        assert self.warmup <= options.memory_size, "Memory size should be bigger than warmup size"
        print("Warm up buffer size is", self.warmup)
        
        ## ---- initialize with same weight ---- ##
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        ## ---- Hyper parameter for DDPG Agent ---- ##
        self.init_delta = options.init_delta
        self.delta = self.init_delta
        self.delta_decay = options.delta_decay

        self.discount = options.discount
        self.tau = options.tau
        
        self.exploration = options.exploration
        self.exploitation = options.exploitation
        
        ## ---- Moving average ---- ##
        self.moving_average = None
        self.moving_alpha = 0.5
        
        ## ---- loss plot ---- ##
        self.td_error_plot = np.zeros((0))
        self.q_value_plot = np.zeros((0))
        
    def reset_delta(self):
        self.delta = self.init_delta
        
    def searching_action(self, episode, current_state, data_processor):
        if self.memory.n_entries < self.warmup:
            action = self.random_action() # return one state index and number of actions
        else:
            embedded_state = data_processor.embed(current_state)
            tensor_state = torch.from_numpy(embedded_state)
            assert (tensor_state > 1).sum() == 0, "Embedding is wrong"
            action = self.actor(tensor_state.float().to(self.device))
            action = action.clone().detach().cpu().numpy()
            
            if episode <= self.exploitation: # exploration에는 initial delta가 유지된다.
                self.delta = self.init_delta * (self.delta_decay ** (self.exploitation - episode))
            action = self.truncated_random_action(action, self.delta)
            
        return action
    
    def deterministic_action(self, current_state, data_processor, do_print=True):
        self.actor.eval()
        
        embedded_state = data_processor.embed(current_state)
        tensor_state = torch.from_numpy(embedded_state)
        assert (tensor_state > 1).sum() == 0, "Embedding is wrong"
        action = self.actor(tensor_state.float().to(self.device))
        action = action.clone().detach().cpu().numpy()
        
        if do_print:
            print(action)
        
        return action
    
    def truncated_random_action(self, mean, var):
        action_0 = self.sample_from_truncated_normal_distribution(lower=0, upper=1, mu=mean[0], sigma=var)
        if self.num_actions == 3:
            action_1 = self.sample_from_truncated_normal_distribution(lower=0, upper=1, mu=mean[1], sigma=var)
            action_2 = self.sample_from_truncated_normal_distribution(lower=0, upper=1, mu=mean[2], sigma=var)
            action_3 = self.sample_from_truncated_normal_distribution(lower=0, upper=1, mu=mean[3], sigma=var)
                
            action = np.array([action_0[0], action_1[0], action_2[0], action_3[0]])
                
        elif self.num_actions == 2:
            action_1 = self.sample_from_truncated_normal_distribution(lower=0, upper=1, mu=mean[1], sigma=var)
            action_2 = self.sample_from_truncated_normal_distribution(lower=0, upper=1, mu=mean[2], sigma=var)
                
            action = np.array([action_0[0], action_1[0], action_2[0]])
                
        else:
            action_1 = self.sample_from_truncated_normal_distribution(lower=0, upper=1, mu=mean[1], sigma=var)
                
            action = np.array([action_0[0], action_1[0]])
    
        return action
    
    def random_action(self):
        action = np.random.uniform(0,1,(self.num_actions + 1))
        return action
        
    def update_memory(self, current_states, actions, next_states, rewards, terminals, length):
        for i in range(length):
            self.memory.add(current_states[i], actions[i], next_states[i], rewards[i], terminals[i])
        
        if self.memory.n_entries < self.warmup:
            print("Buffer is currently being warmed up. [%d/%d]" % (self.memory.n_entries, self.warmup))
        
    def update_network(self):
        if self.memory.n_entries < self.warmup:
            pass
        else:
            ## ---- get sample ---- ##
            batch = self.memory.sample(self.batch_size)
            
            batch = np.array(batch).transpose()
            
            current_state = np.vstack(batch[0])
            action = np.vstack(batch[1])
            next_state = np.vstack(batch[2])
            reward = np.vstack(batch[3])
            terminal = np.vstack(batch[4])
            
            current_state = torch.from_numpy(current_state).float().to(self.device)
            action = torch.from_numpy(action).float().to(self.device)
            next_state = torch.from_numpy(next_state).float().to(self.device)
            reward = torch.from_numpy(reward).float().to(self.device)
            terminal = torch.from_numpy(terminal).float().to(self.device)            
            
            ## ---- moving average ---- ##
            batch_mean_reward = reward.mean().item()
            if self.moving_average is None:
                self.moving_average = batch_mean_reward
            else:
                self.moving_average += self.moving_alpha * (batch_mean_reward - self.moving_average)
            reward = reward - self.moving_average
            
            ######## -------- Critic -------- ########            
            self.critic_optim.zero_grad()
            
            ## ---- get priority: td error ---- ##
            with torch.no_grad():
                critic_action_next = self.actor_target(next_state)
                critic_Q_target_next = self.critic_target([next_state, critic_action_next])
            
            critic_Q_target = reward + self.discount * critic_Q_target_next * terminal
            critic_Q_expected = self.critic([current_state, action])

            td_error = critic_Q_target - critic_Q_expected
            self.td_error_plot = np.append(self.td_error_plot, np.array([td_error.abs().mean().item()]), axis=0)
            #print("[TD-error: %f]" % td_error.abs().mean().item(), end=" ")
                
            ## ---- update network ---- ##
            critic_loss = F.mse_loss(critic_Q_expected, critic_Q_target).mean()
            critic_loss.backward()
            self.critic_optim.step()
            
            ######## -------- Actor  -------- ########
            self.actor_optim.zero_grad()
            
            ## ---- get Q-value ---- ##
            actor_action_expected = self.actor(current_state)
            Q_value = self.critic([current_state, actor_action_expected])
            
            self.q_value_plot = np.append(self.q_value_plot, np.array([Q_value.mean().item()]), axis=0)
            #print("[Q-value: %f]" % Q_value.mean().item())
            
            ## ---- update network ---- ##
            actor_loss = -Q_value.mean()
            actor_loss.backward()
            self.actor_optim.step()           
            
            ################ ---------------- update target network ---------------- ################
            self.update_target_network()
            
    def update_target_network(self):
            self.soft_update(self.actor_target, self.actor)
            self.soft_update(self.critic_target, self.critic)
        
    def soft_update(self, target, source):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
                
    def sample_from_truncated_normal_distribution(self, lower, upper, mu, sigma, size=1):
        return stats.truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=size)
    
    def plot_td_error(self, do_print=False, save=False):
        y_critic = self.td_error_plot.tolist()
        x_epi = list(range(len(y_critic)))
        
        plt.plot(x_epi[5:], y_critic[5:], color='green', label = "TD-error")
        plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        plt.legend()
        
        if do_print:
            plt.show()
            plt.close()
            
        if save:
            file_name = os.path.join(self.save_path, "TD_error.png")
            plt.savefig(file_name)
            plt.close()
            print(file_name, "is saved")
        
    def plot_q_value(self, do_print=False, save=False):
        y_actor = self.q_value_plot.tolist()
        x_epi = list(range(len(y_actor)))
        
        plt.plot(x_epi[5:], y_actor[5:], color='red', label = "Q-value")
        plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        plt.legend()
        
        if do_print:
            plt.show()
            plt.close()
            
        if save:
            file_name = os.path.join(self.save_path, "Q_value.png")
            plt.savefig(file_name)
            plt.close()
            print(file_name, "is saved")

class Replay_Memory:
    def __init__(self, capacity=1000000):
        self.capacity = capacity
        self.buffer = np.zeros(self.capacity, dtype=object)
        
        self.buffer_idx = 0
        self.n_entries = 0
        
    def add(self, current_state, action, next_state, reward, terminal):
        transition = [current_state, action, next_state, reward, terminal]
        self.buffer[self.buffer_idx] = transition
        
        self.buffer_idx += 1
        
        if self.buffer_idx >= self.capacity: # First in, First out
            self.buffer_idx = 0            
        
        if self.n_entries < self.capacity: # How many transitions are stored in buffer
            self.n_entries += 1
            
    def sample(self, batch_size):
        batch = []
        assert self.n_entries >= batch_size, "Buffer is not enough"
        for i in np.random.choice(self.n_entries, batch_size, replace=False):
            batch.append(self.buffer[i])
        return batch
