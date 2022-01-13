import numpy as np


class processor:
    def __init__(self,options):
        if options.method == 'both':
            self.num_actions = 3
        elif options.method == 'rectangular':
            self.num_actions = 2
        else:
            self.num_actions = 1
        
        self.alpha = options.alpha
        
        ## ---- embedding information ---- ##
        self.minmax_layer = [0,1,2,3,4,5,6,7,8,9,10,11,12]
        self.smin = np.zeros((len(self.minmax_layer), 1)) # static information = 13 and dynamic information 'previous'
        self.smax = np.zeros((len(self.minmax_layer), 1)) # static information = 13 and dynamic information 'previous'
        
    def reset(self):
        ## ---- empty array ---- ##
        self.states = np.zeros((0, self.len_state))
        self.states_indices = np.zeros((0))
        self.actions = np.zeros((0, (self.num_actions + 1)))
        self.step_index = 0
        self.remain_state_idx = set([i for i in range(1,(self.num_layer + 1))])
        
    def state_reciever(self, state): # state는 dictionary지만 numpy array로 변경시켜서 집어넣을 것이다.
        self.states = np.append(self.states, state[:21].reshape(1,21), axis=0) # previous is not used
        self.latest_state = state
    
    def action_reciever(self, action, flop):
        
        revised_action, destination, change = self.change_destination(action)
        
        ## ---- prune 비율 설정 ---- ##
        rest = self.latest_state[19]
        destination_flop = int(round(self.flops[destination]))
        duty = self.latest_state[20]
        will_reduce = self.cal_flop(self.num_actions, revised_action, destination, destination_flop)
        
        remain = rest - destination_flop
        debt = duty - will_reduce
        
        if self.type[destination] == 0:
            left = 0.2
        else:
            left = 0.02
        
        if change:
            for i in range(1, (self.num_actions + 1)):
                if remain == 0:
                    assert duty/destination_flop >=0 and duty/destination_flop <=1, "duty/destination_flop is not in (0,1)"
                    revised_action[i] = 1 - duty/destination_flop
                else:
                    if will_reduce > duty:
                        assert duty/destination_flop >=0 and duty/destination_flop <=1, "duty/destination_flop is not in (0,1)"
                        revised_action[i] = max(left, (1 - duty/destination_flop))
                    else:
                        if debt > (remain * 0.5):
                            revised_action[i] = left
                        else:
                            revised_action[i] = np.random.uniform(left, max(left, (1 - debt/remain)), 1)[0]
                        
        else:
            for i in range(1, (self.num_actions + 1)):
                revised_action[i] = max(left, revised_action[i])
            
        self.actions = np.append(self.actions, revised_action.reshape(1,(self.num_actions + 1)), axis=0)
        self.step_index += 1
        
        return revised_action
    
    def change_destination(self, action):
        revised_action = action.copy()
        
        ## ---- state_index 방문 여부 확인 ---- ##
        destination = int(round(self.num_layer * revised_action[0]))
        if destination in self.remain_state_idx:
            change = False
            
        else:
            destination = np.random.choice(list(self.remain_state_idx),1)[0]
            revised_action[0] = destination/self.num_layer
            change = True
            #print("[Destination Changed]")
        
        self.remain_state_idx = self.remain_state_idx - set([destination])
        
        return revised_action, destination, change
        
    def cal_flop(self, num_actions, revised_action, destination, destination_flop):
        if num_actions == 3:
            num_prune_c = int(round((1 - revised_action[1]) * self.channels[destination]))
            num_prune_h = int(round((1 - revised_action[2]) * self.heights[destination]))
            num_prune_w = int(round((1 - revised_action[3]) * self.widths[destination]))
            
            if self.type[destination]  == 0:
                if self.heights[destination] == 1:
                    revised_action[2] = 1
                    num_prune_h = 0
                if self.widths[destination] == 1:
                    revised_action[3] = 1
                    num_prune_w = 0
                    
                if num_prune_c == 0:
                    C_ratio = self.channels[destination]
                else:
                    C_ratio = num_prune_c
                if num_prune_h == 0:
                    H_ratio = self.heights[destination]
                else:
                    H_ratio = num_prune_h
                if num_prune_w == 0:
                    W_ratio = self.widths[destination]
                else:
                    W_ratio = num_prune_w
                    
                if num_prune_c == 0 and num_prune_h == 0 and num_prune_w == 0:
                    FLOPs = 0                 
                else:
                    FLOPs = destination_flop
                
                will_reduce = (C_ratio/self.channels[destination])*(H_ratio/self.heights[destination])*(W_ratio/self.widths[destination])*FLOPs
                
                return int(round(will_reduce))
                
            else:
                if num_prune_c == 0:
                    C_ratio = self.channels[destination]
                    FLOPs = 0
                    bias_FLOPs = 0
                else:
                    C_ratio = num_prune_c
                    FLOPs = destination_flop
                    bias_FLOPs = self.out_ch[destination]
                
                will_reduce = (C_ratio/self.channels[destination]) * (FLOPs - bias_FLOPs)
                
                return int(round(will_reduce))
            
        elif num_actions == 2:
            num_prune_h = int(round((1 - revised_action[1]) * self.heights[destination]))
            num_prune_w = int(round((1 - revised_action[2]) * self.widths[destination]))
            
            if self.type[destination]  == 0:
                if self.heights[destination] == 1:
                    revised_action[1] = 1
                if self.widths[destination] == 1:
                    revised_action[2] = 1
            else:
                
                if num_prune_h == 0:
                    H_ratio = self.heights[destination]
                else:
                    H_ratio = num_prune_h
                if num_prune_w == 0:
                    W_ratio = self.widths[destination]
                else:
                    W_ratio = num_prune_w
                
                if num_prune_h == 0 and num_prune_w == 0:
                    FLOPs = 0
                else:
                    FLOPs = destination_flop
                
                will_reduce = (H_ratio/self.heights[destination])*(W_ratio/self.widths[destination])*FLOPs
                
                return int(round(will_reduce))
                
        else:
            num_prune_c = int(round((1 - revised_action[1]) * self.channels[destination]))
            
            if self.type[destination]  == 0:
                if num_prune_c == 0:
                    C_ratio = self.channels[destination]
                    FLOPs = 0
                else:
                    C_ratio = num_prune_c
                    FLOPs = destination_flop
                    
                will_reduce = (C_ratio/self.channels[destination]) * FLOPs
                
                return int(round(will_reduce))
            
            else:
                if num_prune_c == 0:
                    C_ratio = self.channels[destination]
                    FLOPs = 0
                    bias_FLOPs = 0
                else:
                    C_ratio = num_prune_c
                    FLOPs = destination_flop
                    bias_FLOPs = self.out_ch[destination]
                    
                will_reduce = (C_ratio/self.channels[destination]) * (FLOPs - bias_FLOPs)
        
                return int(round(will_reduce))

    def reward_reciever(self, reward):
        self.terminals = np.ones((self.step_index, 1))
        self.terminals[-1] = 0
        self.rewards = np.zeros((self.step_index, 1))
        self.rewards[:] = reward

    def state_index_reciever(self, state_index):
        self.states_indices = np.append(self.states_indices, state_index, axis=0)
    
    def convert_data(self):
        ## ---- layer-wise relative embedding ---- ##
        for j in range(self.step_index + 1):
            self.states[j,13] = self.states[j,13]/self.states[j,9] if self.states[j,9] != 0 else self.states[j,13]    # x
            self.states[j,14] = self.states[j,14]/self.states[j,10] if self.states[j,10] != 0 else self.states[j,14]  # y
            self.states[j,15] = self.states[j,15]/self.states[j,9] if self.states[j,9] != 0 else self.states[j,15]    # receptive height
            self.states[j,16] = self.states[j,16]/self.states[j,10] if self.states[j,10] != 0 else self.states[j,16]  # receptive width
        
        ## ---- total flop-wise absolute embedding ---- ##
        self.states[:,17] /= self.total_FLOPs
        self.states[:,18] /= self.total_FLOPs
        
        self.states[:,19] /= self.goal
        self.states[:,20] /= self.goal
        
        ## ---- previous layer ---- ##
        #self.states[:,21] = (self.states[:,21] - self.smin[0])/(self.smax[0] - self.smin[0])
        
        ## ---- min-max embedding ---- ##
        j = 0
        for i in self.minmax_layer:
            smin = self.smin[j]
            smax = self.smax[j]
            
            if smin == smax:
                self.states[:,i] = max(min(1, smin), 0)
            else:
                self.states[:,i] = (self.states[:,i] - smin)/(smax - smin)
                
            j += 1
        
        self.current_states = self.states[:self.step_index,:21]
        self.next_states = self.states[1:, :21]
        
        print("~"*30)
        print("%d steps are searched" % self.step_index)
        print("~"*30)
        return self.current_states.copy(), self.actions.copy(), self.next_states.copy(), self.rewards.copy(), self.terminals.copy(), self.step_index
    
    def embed(self, state): # state's type is numpy array
        embedded_state = state.copy()
        
        embedded_state[13] = embedded_state[13]/embedded_state[9] if embedded_state[9] != 0 else embedded_state[13]
        embedded_state[14] = embedded_state[14]/embedded_state[10] if embedded_state[10] != 0 else embedded_state[14]
        embedded_state[15] = embedded_state[15]/embedded_state[9] if embedded_state[9] != 0 else embedded_state[15]
        embedded_state[16] = embedded_state[16]/embedded_state[10] if embedded_state[10] != 0 else embedded_state[16]
        
        embedded_state[17] /= self.total_FLOPs
        embedded_state[18] /= self.total_FLOPs
        
        embedded_state[19] /= self.goal
        embedded_state[20] /= self.goal
        
        #embedded_state[21] = (embedded_state[21] - self.smin[0])/(self.smax[0] - self.smin[0])
        
        j = 0
        for i in self.minmax_layer:
            smin = self.smin[j]
            smax = self.smax[j]
            
            if smin == smax:
                embedded_state[i] = max(min(1, smin), 0)
            else:
                embedded_state[i] = (embedded_state[i] - smin)/(smax - smin)
                
            j += 1
        
        if (embedded_state[:21] > 1).sum() != 0:
            print(state)
            print(embedded_state[:21])
        
        return embedded_state[:21]
    
    def get_information(self, environment):
        state, _, done = environment.reset()
        self.total_FLOPs = environment.total_FLOPs
        self.num_layer = environment.num_layer
        self.len_state = environment.len_state
        self.goal = environment.goal
        
        self.reset()
        self.state_reciever(np.array(list(state.values())))
        
        action = np.ones((self.num_actions + 1))
        
        while done != True:
            state, _, done = environment.step(action, 1, True)
            self.state_reciever(np.array(list(state.values())))
        
        ## ---- save flop information ---- ##
        self.heights = self.states[:,3].copy()
        self.widths = self.states[:,4].copy()
        self.channels = self.states[:,11].copy()
        self.flops = self.states[:,12].copy()
        self.type = self.states[:,1].copy()
        self.out_ch = self.states[:,2].copy()
        
        ## ---- for minmax embedding ---- ##
        j = 0
        for i in self.minmax_layer:
            self.smin[j] = self.states[:,i].min()
            self.smax[j] = self.states[:,i].max()
            
            j += 1
        
        ## ---- reset ---- ##
        state, state_index, done = environment.reset()

        return state, state_index, done














