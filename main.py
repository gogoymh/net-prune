from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import argparse
import numpy as np
import timeit
import os
import math

import network
from environment import Environment
from agent import DDPG_Agent
from data_processor import processor
##############################################################################################################################
#CUDA_VISIBLE_DEVICES=0 python main.py --experiment "both" --experiment_repeat "_1" --pretrained_path "/DATA/ymh/modelcomp/experiment/resnet56_finetune_both1.pth" --beta 0 --beta_increment_per_sampling 0 --alpha 0 --search_repeat 100 --restrict_var 200 --var_plus 0.0001

##############################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--experiment_path", type=str, default="C://유민형//개인 연구//net prune//experiment", help="File directory")
#parser.add_argument("--experiment_path", type=str, default="/DATA/ymh/modelcomp/experiment", help="File directory")
#parser.add_argument("--experiment_path", type=str, default="/data1/ymh/modelcomp/experiment", help="File directory")
#parser.add_argument("--experiment_path", type=str, default="/home/cscoi/MH/experiment", help="File directory")
parser.add_argument("--experiment", type=str, default="1.0.0", help="Experiment Index")
parser.add_argument("--experiment_repeat", type=str, default=".0", help="Experiment Index")
parser.add_argument("--model", type=str, default="resnet56", help="Model to prune")
parser.add_argument("--pretrained_path", type=str, default="C://results//resnet56_real5.pth", help="Pretrained Model Path")
#parser.add_argument("--pretrained_path", type=str, default="/DATA/ymh/modelcomp/experiment/resnet56_real5.pth", help="Pretrained Model Path")
#parser.add_argument("--pretrained_path", type=str, default="/data1/ymh/modelcomp/experiment/resnet56_real5.pth", help="Pretrained Model Path")
#parser.add_argument("--pretrained_path", type=str, default="/home/cscoi/MH/resnet56_real5.pth", help="Pretrained Model Path")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
parser.add_argument("--method", type=str, default="channel", help="Method")

parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
parser.add_argument("--memory_size", type=int, default=2000, help="Memory size")
parser.add_argument("--warmup", type=int, default=1000, help="Warm up episodes")
parser.add_argument("--exploration", type=int, default=100, help="Total episodes")
parser.add_argument("--exploitation", type=int, default=300, help="Total episodes")

parser.add_argument("--init_delta", type=float, default=0.5, help="Initial delta")
parser.add_argument("--delta_decay", type=float, default=0.95, help="Delta decay")
parser.add_argument("--discount", type=float, default=1, help="Discount factor for Q-value function")
parser.add_argument("--tau", type=float, default=0.01, help="Tau for soft update")

parser.add_argument("--alpha", type=float, default=0.05, help="Search parameter")

opt = parser.parse_args()
print("="*100)
print(opt)
print("="*100)
##############################################################################################################################
exp_num = opt.experiment + opt.experiment_repeat
save_path = os.path.join(opt.experiment_path, exp_num)
if os.path.isdir(save_path):
    print("Save path exists: ",save_path)
else:
    os.mkdir(save_path)
    print("Save path is created: ",save_path)

##############################################################################################################################
'''
train_loader = DataLoader(
                datasets.CIFAR10(
                        "../data/CIFAR10",
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                ),
                        ),
                batch_size=128, shuffle=True, pin_memory=True)
'''

test_loader = DataLoader(
                datasets.CIFAR10(
                        '../data/CIFAR10',
                        train=False,
                        download=True,
                        transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                ),
                        ),
                batch_size=100, shuffle=False, pin_memory=True)

################################################### Instanciate ###############################################################
if opt.model == "resnet56":
    model = network.resnet56(num_classes=10)

env = Environment(model, test_loader, opt)
agent = DDPG_Agent(env.len_state, opt)
data_processor = processor(opt)

#################################################### Search ###################################################################
total_start = timeit.default_timer()

for flop_per in range(10, 11, 1):
#for flop_per in range(1,51,1):
    flop = flop_per/100
    env.goal(flop)
    state, state_index, done = data_processor.get_information(env) # reset is included
    data_processor.reset()
    agent.reset_delta()
    data_processor.state_reciever(np.array(list(state.values())))
    data_processor.state_index_reciever(state_index)
    
    episode = opt.exploration + opt.exploitation
    total_episode = episode
    
    ## ---- search ---- ##
    flop_start = timeit.default_timer()
    while episode:
        episode_start = timeit.default_timer()
        print("="*100)
        print("[FLOP: %d] [Episode: %d]" % ((100 - flop_per), (total_episode - episode + 1)))
        while done != True:
            ## ---- get action ---- ##
            action = agent.searching_action(episode, np.array(list(state.values())), data_processor)
            revised_action = data_processor.action_reciever(action, flop) # 같은 layer로 이동하는 것 방지 및 초과 prune 방지
            
            ## ---- next state ---- ##
            state, state_index, done = env.step(revised_action, flop)
            data_processor.state_reciever(np.array(list(state.values())))
            data_processor.state_index_reciever(state_index)

            if done:
                ## ---- get reward ---- ##
                reward = - env.val_loss() #* env.step_index
                data_processor.reward_reciever(reward)
                
                ## ---- update buffer ---- ##
                current_states, actions, next_states, rewards, terminals, length = data_processor.convert_data() # returns embedded numpy array
                agent.update_memory(current_states, actions, next_states, rewards, terminals, length)
                
                ## ---- update network ---- ##
                agent.update_network() # Synchronous Update    
                
                print("[Reward: %f] [delta: %f]" % (reward, agent.delta), end=" ")
                
                if agent.memory.n_entries < agent.warmup:
                    episode += 1 # warm up: episode is not consumed
                    
                episode_finish = timeit.default_timer()
                print("[Single Episode Time: %f]" % (episode_finish-episode_start))
        
        ## ---- start new episode ---- ##
        state, state_index, done = env.reset()
        data_processor.reset()
        data_processor.state_reciever(np.array(list(state.values())))
        data_processor.state_index_reciever(state_index)
        episode -= 1
    
    ## ---- Test ---- ##
    print("="*100)
    print("Pruned model's when [FLOP: %f]" % flop)
    while done != True: # Test ans Save
        ## ---- get action ---- ##
        action = agent.deterministic_action(np.array(list(state.values())), data_processor)
        revised_action = data_processor.action_reciever(action, flop)
        
        ## ---- next state ---- ##
        state, state_index, done = env.step(revised_action, flop)
        data_processor.state_reciever(np.array(list(state.values())))
        data_processor.state_index_reciever(state_index)
        
        if done:
            ## ---- get reward ---- ##
            val_reward = - env.val_loss() #* env.step_index # validation set reward
            val_acc = env.val_acc() # validation set accuracy
            print("="*100)
            print("[Validation] [Loss: %f] [Accuracy: %f]" % (env.val_loss(), val_acc))
            
            test_reward = - env.test_loss() #* env.step_index # test set
            test_acc = env.test_acc()
            print("[Test]       [Loss: %f] [Accuracy: %f]" % (env.test_loss(), test_acc))

            env.original_result() # compare with original one
            
            #agent.save_actor()
            
            flop_finish = timeit.default_timer()
            print("="*100)
            print("[Single FLOP Searching Time: %f]" % (flop_finish-flop_start))
            
total_finish = timeit.default_timer()
agent.plot_td_error(do_print=True, save=False)
agent.plot_q_value(do_print=True, save=False)
#################################################################################################################################
print("="*100)
print(opt)
print("="*100)
print("How long did it take: %f seconds" % (total_finish - total_start))
