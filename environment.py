import torch
import copy
import numpy as np
import torch.nn as nn

class Environment:
    def __init__(self, model, test_loader, options):
        
        ## ---- save method ---- ##
        self.method = options.method
        print(self.method)
        ## ---- prepare dataset ---- ##
        self.validation_set = []
        self.test_set = []
        self.half_length = int(len(test_loader.dataset)/2)
        for batch_idx, data in enumerate(test_loader):
            if batch_idx < 50:
                self.validation_set.append(data)
            else:
                self.test_set.append(data)
        print("Validation/Test set length is both", self.half_length)
        
        ## ---- build model ---- ##
        self.device = options.device
        self.model = model.to(self.device)
        
        if self.device == "cuda:0":
            self.checkpoint = torch.load(options.pretrained_path)
        else:
            self.checkpoint = torch.load(options.pretrained_path, map_location=lambda storage, location: 'cpu')
        
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        
        ## ---- prepare layers to deal with ---- ##
        self.module_list = list(self.model.modules())
        self.prunable_layer_types = [torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Linear]
        
        self.prunable_layer = []
        for i, m in enumerate(self.module_list):
            if type(m) in self.prunable_layer_types:
                self.prunable_layer.append(i)
                
        print('Prunable layer is', self.prunable_layer)
        self.num_layer = len(self.prunable_layer)
        print('Length of Prunable layer is', self.num_layer)
        
        ## ---- define states ---- ##
        def extract_information_from_layer(layer, x):
            def get_layer_type(layer):
                layer_str = str(layer)
                return layer_str[:layer_str.find('(')].strip()
            
            type_name = get_layer_type(layer)
            
            if type_name in ['Conv2d']:
                layer.input_height = x.size()[2]
                layer.input_width = x.size()[3]
                layer.output_height = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) / layer.stride[0] + 1)
                layer.output_width = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) / layer.stride[1] + 1)
                layer.flops = int(round(layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1] * layer.output_height * layer.output_width / layer.groups))
                layer.type = 0 # Convolutional
                
            elif type_name in ['Linear']:
                layer.input_height = 1
                layer.input_width = 1
                layer.output_height = 1 
                layer.output_width = 1
                
                weight_ops = layer.weight.numel()
                bias_ops = layer.bias.numel()
                layer.flops = int(round(weight_ops + bias_ops))
                
                layer.stride = 0, 0
                layer.kernel_size = 1, 1
                layer.in_channels = layer.in_features
                layer.out_channels = layer.out_features
                layer.type = 1 # Linear
                layer.padding = 0, 0
                
            return
        
        def new_forward(m): # 이름 바꾸기
            def lambda_forward(x): # 
                extract_information_from_layer(m, x)
                y = m.old_forward(x)
                return y
            
            return lambda_forward
        
        for idx in self.prunable_layer:  # get all
            m = self.module_list[idx]
            m.old_forward = m.forward
            m.forward = new_forward(m)
        
        with torch.no_grad():
            self.model.eval()
            correct = 0
            for x, y in self.validation_set:
                output = self.model(x.float().to(self.device))
                pred = output.argmax(1, keepdim=True)
                correct += pred.eq(y.long().to(self.device).view_as(pred)).sum().item()
        
        self.accuracy_original = correct/self.half_length
        
        def embed_state_from_information(self, layer_idx, initial=False):
            
            m = self.module_list[layer_idx]
                
            if initial:
                ## ---- There isn't kernel ---- ##
                kernel_index = 0
                kernel_out_channels = 0
                kernel_height = 0
                kernel_width = 0
                stride_height = 0
                stride_width = 0
                padding_height = 0
                padding_width = 0
                FLOPs = 0
                kernel_type = 0
                
                ## ---- output feature information ---- ##    
                feature_height = 0
                feature_width = 0
                feature_channel = 0
                
            else:
                ## ---- kernel information ---- ##
                kernel_index = self.current_state_idx
                kernel_out_channels = m.out_channels
                kernel_height = m.kernel_size[0]
                kernel_width = m.kernel_size[1]
                stride_height = m.stride[0]
                stride_width = m.stride[1]
                padding_height = m.padding[0]
                padding_width = m.padding[1]
                FLOPs = m.flops
                kernel_type = m.type
                
                ## ---- input feature information ---- ##
                feature_height = m.input_height
                feature_width = m.input_width
                feature_channel = m.in_channels
                
            ## ---- pseudo response information ---- ##
            x = 0
            y = 0
            receptive_height = 0
            receptive_width = 0
            searched = 0
            rest = 0
            reduced = 0
            duty = 0
            previous = 0
            
            ## ---- make dictionary for state ---- ##
            layer_info = dict()
            
            ## -- static information -- ##
            layer_info['kernel_index'] = kernel_index                 #0
            layer_info['kernel_type'] = kernel_type                   #1
            layer_info['kernel_out_channels'] = kernel_out_channels   #2
            layer_info['kernel_height'] = kernel_height               #3
            layer_info['kernel_width'] = kernel_width                 #4
            layer_info['stride_height'] = stride_height               #5
            layer_info['stride_width'] = stride_width                 #6
            layer_info['padding_height'] = padding_height             #7
            layer_info['padding_width'] = padding_width               #8
            
            layer_info['feature_height'] = feature_height             #9
            layer_info['feature_width'] =  feature_width              #10
            layer_info['feature_channel'] = feature_channel           #11
            
            layer_info['FLOPs'] = int(FLOPs)                          #12
            
            ## -- dynamic information(depends on action) -- ##
            layer_info['x'] = x                                       #13
            layer_info['y'] = y                                       #14
            
            layer_info['receptive_height'] = receptive_height         #15
            layer_info['receptive_width'] = receptive_width           #16
            
            layer_info['searched'] = searched                         #17
            layer_info['rest'] = rest                                 #18
            
            layer_info['reduced'] = reduced                           #19
            layer_info['duty'] = duty                                 #20
            
            layer_info['previous'] = previous                         #21
            
            return layer_info
        
        ## ---- get layer information ---- ##
        self.basic_states = []
        self.current_state_idx = 0
        self.basic_states.append(embed_state_from_information(self, layer_idx=i, initial=True))
        for i in self.prunable_layer:
            self.current_state_idx += 1
            self.basic_states.append(embed_state_from_information(self, layer_idx=i, initial=False))

        ## ---- Calculate total FLOP in neural network ---- ##
        self.total_FLOPs = 0
        for j in range(len(self.basic_states)):
            self.total_FLOPs += self.basic_states[j]['FLOPs']
        self.basic_states[0]['rest'] = self.total_FLOPs
        print('Total FLOP is %d' % self.total_FLOPs)
        
        ## ---- length of state ---- ##
        self.len_state = len(self.basic_states[0]) - 1 # 22 - 1(previous)
        
    def goal(self, flop):
        self.goal = int(round(self.total_FLOPs * flop))
        self.basic_states[0]['duty'] = self.goal
        print('Goal to prune is %d' % self.total_FLOPs)
        
    def reset(self):
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        
        self.episodic_states = copy.deepcopy(self.basic_states)
        initial_state = self.episodic_states[0]
        
        self.previous_state_idx = 0
        self.step_index = 0
        
        return initial_state, np.array([self.previous_state_idx]), False
        
    def step(self, action, flop, sequentially=False):
        ## ---- decide which layer to go ---- ##
        if sequentially:
            current_state_idx = self.previous_state_idx + 1            
        else:
            current_state_idx = int(round(self.num_layer * action[0]))
        
        ## ---- get current state and kernel ---- ##
        previous_state = self.episodic_states[self.previous_state_idx]
        current_state = self.episodic_states[current_state_idx]
        
        current_kernel = self.module_list[self.prunable_layer[(current_state_idx-1)]]
        weight = current_kernel.weight.data.clone().cpu().numpy()
        
        ## ---- prune ---- ##
        if self.method == 'channel':
            assert len(action[1:]) == 1, "Channel pruning action should be single float number"
            assert action[1] >=0 and action[1] <=1, "action 1 is not in (0,1)"
            
            ## ---- decide how many channel to save ---- ##
            num_prune_c = int(round((1 - action[1]) * current_state['feature_channel']))
            ## ---- execute action ---- ##
            if current_state['kernel_type'] == 0: # Convolutional
                ## ---- prune channel ---- ##
                importance_c = np.abs(weight).sum((0,2,3))
                sorted_idx_c = np.argsort(importance_c)
                preserve_idx_c = sorted_idx_c[num_prune_c:] if num_prune_c != 0 else range(0, weight.shape[1])
                mask_c = np.ones(weight.shape[1], bool)
                mask_c[preserve_idx_c] = False
                weight[:,mask_c,:,:] = 0
                
                ## ---- update information ---- ##
                if num_prune_c == 0:
                    C_ratio = current_state['feature_channel']
                    FLOPs = 0
                else:
                    C_ratio = num_prune_c
                    FLOPs = current_state['FLOPs']
                x = 0
                y = 0
                receptive_height = current_state['feature_height']
                receptive_width = current_state['feature_width']
                searched = previous_state['searched'] + current_state['FLOPs']
                rest = self.total_FLOPs - searched
                reduced = previous_state['reduced'] + (C_ratio/current_state['feature_channel']) * FLOPs
                duty = self.goal - reduced
                previous = self.previous_state_idx
                
            else: # Linear
                ## ---- prune node ---- ##
                importance_c = np.abs(weight).sum((0))
                sorted_idx_c = np.argsort(importance_c)
                preserve_idx_c = sorted_idx_c[num_prune_c:] if num_prune_c != 0 else range(0, weight.shape[1])
                mask_c = np.ones(weight.shape[1], bool)
                mask_c[preserve_idx_c] = False
                weight[:,mask_c] = 0
                
                ## ---- update information ---- ##
                if num_prune_c == 0:
                    C_ratio = current_state['feature_channel']
                    FLOPs = 0
                    bias_FLOPs = 0
                else:
                    C_ratio = num_prune_c
                    FLOPs = current_state['FLOPs']
                    bias_FLOPs = current_state['kernel_out_channels']
                
                x = 0
                y = 0
                receptive_height = current_state['feature_height']
                receptive_width = current_state['feature_width']
                searched = previous_state['searched'] + current_state['FLOPs']
                rest = self.total_FLOPs - searched
                reduced = previous_state['reduced'] + (C_ratio/current_state['feature_channel']) * (FLOPs - bias_FLOPs)
                duty = self.goal - reduced
                previous = self.previous_state_idx
        
        elif self.method == 'rectangular':
            assert len(action[1:]) == 2, "Rectangular pruning action should be single float number"
            assert action[1] >=0 and action[1] <=1, "action 1 is not in (0,1)"
            assert action[2] >=0 and action[2] <=1, "action 2 is not in (0,1)"
            
            ## ---- decide how many rectangular to save ---- ##
            num_prune_h = int(round((1 - action[1]) * current_state['kernel_height']))
            num_prune_w = int(round((1 - action[2]) * current_state['kernel_width']))
            ## ---- execute action ---- ##
            if current_state['kernel_type'] == 0: # Convolutional
                ## ---- exception for 1x1 convolutional filter ---- ##
                if current_state['kernel_height'] == 1:
                    num_prune_h = 0
                if current_state['kernel_width'] == 1:
                    num_prune_w = 0
                
                ## ---- prune kernel ---- ##
                importance_h = np.abs(weight).sum((0,1,2))
                importance_w = np.abs(weight).sum((0,1,3))
                sorted_idx_h = np.argsort(importance_h)
                sorted_idx_w = np.argsort(importance_w)
                preserve_idx_h = sorted_idx_h[num_prune_h:] if num_prune_h != 0 else range(0, weight.shape[3])
                preserve_idx_w = sorted_idx_w[num_prune_w:] if num_prune_w != 0 else range(0, weight.shape[2])
                mask_h = np.ones(weight.shape[2], bool)
                mask_w = np.ones(weight.shape[3], bool)
                mask_h[preserve_idx_h] = False
                mask_w[preserve_idx_w] = False
                weight[:,:,mask_h,:] = 0
                weight[:,:,:,mask_w] = 0
                
                ## ---- update information ---- ##
                if num_prune_h == 0:
                    H_ratio = current_state['kernel_height']
                else:
                    H_ratio = num_prune_h
                if num_prune_w == 0:
                    W_ratio = current_state['kernel_width']
                else:
                    W_ratio = num_prune_w
                
                if num_prune_h == 0 and num_prune_w == 0:
                    FLOPs = 0
                else:
                    FLOPs = current_state['FLOPs']
                
                x = max(0, (np.where(mask_h==False)[0][0] - current_state['padding_height'])) if num_prune_h != current_state['kernel_height'] else 0
                y = max(0, (np.where(mask_w==False)[0][0] - current_state['padding_width'])) if num_prune_w != current_state['kernel_width'] else 0
                receptive_height = min((current_state['feature_height']-x),(2*current_state['padding_height'] + current_state['feature_height'] - np.where(mask_h==False)[0][0] - len(mask_h) + np.where(mask_h==False)[0][-1] + 1)) if num_prune_h != current_state['kernel_height'] else current_state['feature_height']
                receptive_width = min((current_state['feature_width']-y),(2*current_state['padding_width'] + current_state['padding_width'] + current_state['feature_width'] - np.where(mask_w==False)[0][0] - len(mask_w) + np.where(mask_w==False)[0][-1] + 1)) if num_prune_w != current_state['kernel_width'] else current_state['feature_width']
                searched = previous_state['searched'] + current_state['FLOPs']
                rest = self.total_FLOPs - searched
                reduced = previous_state['reduced'] + (H_ratio/current_state['kernel_height'])*(W_ratio/current_state['kernel_width'])*FLOPs
                duty = self.goal - reduced
                previous = self.previous_state_idx
                
            else:# Rectangular doesn't have way to prune linear kernel
                x = 0
                y = 0
                receptive_height = current_state['feature_height']
                receptive_width = current_state['feature_width']
                searched = previous_state['searched'] + current_state['FLOPs']
                rest = self.total_FLOPs - searched
                reduced = previous_state['reduced']
                duty = self.goal - reduced
                previous = self.previous_state_idx
            
        elif self.method == 'both':
            assert len(action[1:]) == 3, "Channel and rectangular pruning action should be pair of float numbers"
            assert action[1] >=0 and action[1] <=1, "action 1 is not in (0,1)"
            assert action[2] >=0 and action[2] <=1, "action 2 is not in (0,1)"
            assert action[3] >=0 and action[3] <=1, "action 3 is not in (0,1)"
            
            ## ---- decide how many channel and rectangular to save ---- ##
            num_prune_c = int(round((1 - action[1]) * current_state['feature_channel']))
            num_prune_h = int(round((1 - action[2]) * current_state['kernel_height']))
            num_prune_w = int(round((1 - action[3]) * current_state['kernel_width']))
            ## ---- execute action ---- ##
            if current_state['kernel_type'] == 0: # Convolutional
                if current_state['kernel_height'] == 1:
                    num_prune_h = 0
                if current_state['kernel_width'] == 1:
                    num_prune_w = 0
                ## ---- prune channel and kernel ---- ##
                importance_c = np.abs(weight).sum((0,2,3))
                importance_h = np.abs(weight).sum((0,1,2))
                importance_w = np.abs(weight).sum((0,1,3))
                sorted_idx_c = np.argsort(importance_c)
                sorted_idx_h = np.argsort(importance_h)
                sorted_idx_w = np.argsort(importance_w)
                preserve_idx_c = sorted_idx_c[num_prune_c:] if num_prune_c != 0 else range(0, weight.shape[1])
                preserve_idx_h = sorted_idx_h[num_prune_h:] if num_prune_h != 0 else range(0, weight.shape[3])
                preserve_idx_w = sorted_idx_w[num_prune_w:] if num_prune_w != 0 else range(0, weight.shape[2])
                mask_c = np.ones(weight.shape[1], bool)
                mask_h = np.ones(weight.shape[2], bool)
                mask_w = np.ones(weight.shape[3], bool)
                mask_c[preserve_idx_c] = False
                mask_h[preserve_idx_h] = False
                mask_w[preserve_idx_w] = False
                weight[:,mask_c,:,:] = 0
                weight[:,:,mask_h,:] = 0
                weight[:,:,:,mask_w] = 0
                
                ## ---- update information ---- ##
                if num_prune_c == 0:
                    C_ratio = current_state['feature_channel']
                else:
                    C_ratio = num_prune_c
                if num_prune_h == 0:
                    H_ratio = current_state['kernel_height']
                else:
                    H_ratio = num_prune_h
                if num_prune_w == 0:
                    W_ratio = current_state['kernel_width']
                else:
                    W_ratio = num_prune_w
                    
                if num_prune_c == 0 and num_prune_h == 0 and num_prune_w == 0:
                    FLOPs = 0                 
                else:
                    FLOPs = current_state['FLOPs']
                
                x = max(0, (np.where(mask_h==False)[0][0] - current_state['padding_height'])) if num_prune_h != current_state['kernel_height'] else 0
                y = max(0, (np.where(mask_w==False)[0][0] - current_state['padding_width'])) if num_prune_w != current_state['kernel_width'] else 0
                receptive_height = min((current_state['feature_height']-x),(2*current_state['padding_height'] + current_state['feature_height'] - np.where(mask_h==False)[0][0] - len(mask_h) + np.where(mask_h==False)[0][-1] + 1)) if num_prune_h != current_state['kernel_height'] else current_state['feature_height']
                receptive_width = min((current_state['feature_width']-y), (2*current_state['padding_width'] + current_state['feature_width'] - np.where(mask_w==False)[0][0] - len(mask_w) + np.where(mask_w==False)[0][-1] + 1)) if num_prune_w != current_state['kernel_width'] else current_state['feature_width']
                searched = previous_state['searched'] + current_state['FLOPs']
                rest = self.total_FLOPs - searched
                reduced = previous_state['reduced'] + (C_ratio/current_state['feature_channel'])*(H_ratio/current_state['kernel_height'])*(W_ratio/current_state['kernel_width'])*FLOPs
                duty = self.goal - reduced
                previous = self.previous_state_idx
                
            else: # Linear
                ## ---- prune node ---- ##
                importance_c = np.abs(weight).sum((0))
                sorted_idx_c = np.argsort(importance_c)
                preserve_idx_c = sorted_idx_c[num_prune_c:] if num_prune_c != 0 else range(0, weight.shape[1])
                mask_c = np.ones(weight.shape[1], bool)
                mask_c[preserve_idx_c] = False
                weight[:,mask_c] = 0
                
                ## ---- update information ---- ##
                if num_prune_c == 0:
                    C_ratio = current_state['feature_channel']
                    FLOPs = 0
                    bias_FLOPs = 0
                else:
                    C_ratio = num_prune_c
                    FLOPs = current_state['FLOPs']
                    bias_FLOPs = current_state['kernel_out_channels']
                x = 0
                y = 0
                receptive_height = current_state['feature_height']
                receptive_width = current_state['feature_width']
                searched = previous_state['searched'] + current_state['FLOPs']
                rest = self.total_FLOPs - searched
                reduced = previous_state['reduced'] + (C_ratio/current_state['feature_channel']) * (FLOPs - bias_FLOPs)
                duty = self.goal - reduced
                previous = self.previous_state_idx
                
        else:
            raise NameError("You should select proper method.")
        
        ## ---- assign pruned weight array to parameter tensor ---- ##
        current_kernel.weight.data = torch.from_numpy(weight).to(self.device)
        
        ## ---- revise current state information ---- ##
        current_state['x'] = x
        current_state['y'] = y
        current_state['receptive_height'] = receptive_height
        current_state['receptive_width'] = receptive_width
        current_state['searched'] = searched
        current_state['rest'] = rest
        current_state['reduced'] = reduced
        current_state['duty'] = duty
        current_state['previous'] = previous

        self.previous_state_idx = current_state_idx
        self.step_index += 1
        print('[Index %2d] [FLOP reduced: %f]' % (current_state_idx, reduced/self.total_FLOPs))
        
        ## ---- decide final state ---- ##
        if duty <= 0 or self.step_index == self.num_layer: # Final
            return current_state, np.array([current_state_idx]), True
        else:
            return current_state, np.array([current_state_idx]), False
        
    def val_loss(self):
        criterion = nn.CrossEntropyLoss()
        loss = 0
        with torch.no_grad():
            self.model.eval()
            for x, y in self.validation_set:
                output = self.model(x.float().to(self.device))
                loss_tmp = criterion(output, y.long().to(self.device))
                loss += loss_tmp.item()
        loss /= len(self.validation_set)
        return loss           
    
    def val_acc(self):
        accuracy = 0
        with torch.no_grad():
            self.model.eval()
            correct = 0
            for x, y in self.validation_set:
                output = self.model(x.float().to(self.device))
                pred = output.argmax(1, keepdim=True)
                correct += pred.eq(y.long().to(self.device).view_as(pred)).sum().item()
        accuracy = correct / self.half_length
        return accuracy
        
    def test_loss(self):
        criterion = nn.CrossEntropyLoss()
        loss = 0
        with torch.no_grad():
            self.model.eval()
            for x, y in self.test_set:
                output = self.model(x.float().to(self.device))
                loss_tmp = criterion(output, y.long().to(self.device))
                loss += loss_tmp.item()
        loss /= len(self.validation_set)
        return loss
        
    def test_acc(self):
        accuracy = 0
        with torch.no_grad():
            self.model.eval()
            correct = 0
            for x, y in self.test_set:
                output = self.model(x.float().to(self.device))
                pred = output.argmax(1, keepdim=True)
                correct += pred.eq(y.long().to(self.device).view_as(pred)).sum().item()
        accuracy = correct / self.half_length
        return accuracy
    
    def original_result(self):
        _, _, _ = self.reset()
        val_reward = self.val_loss()
        val_acc = self.val_acc()
        test_reward = self.test_loss()
        test_acc = self.test_acc()
        
        print("="*100)
        print("Original Model's")
        print("[Validation] [Loss: %f] [Accuracy: %f]" % (val_reward, val_acc))
        print("[Test]       [Loss: %f] [Accuracy: %f]" % (test_reward, test_acc))
        
    def fine_tune(self, re_initialize=False):
        
        return


        
    














