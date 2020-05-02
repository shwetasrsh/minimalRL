import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# An effective way to monitor and analyze an agent’s success during training is to chart its cumulative reward at the end of 
# each episode.

#Hyperparameters
#learning_rate = 0.0005
#gamma         = 0.98
#buffer_limit  = 50000
#batch_size    = 32


#Hyperparameters from open ai gym
learning_rate = 0.001
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 128


# this buffer is a dataset of our agent's past experiences
# this ensures that the agent is learning from its entire history
class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        #Here 4 because each state representation is an input and that takes 4 preprocessed image frames
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else : 
            return out.argmax().item()
            
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)
        
        q_out = q(s)
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    #env = gym.make('CartPole-v1')
    #env = gym.make('MsPacman-v0')
    env = gym.make('MsPacman-No-Frameskip-v0')
    #use env = gym.make('MsPacman-No-Frameskip-v0')
    #and then on this apply the opitcal flow
    q = Qnet()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = Qnet().to(device)
    q_target = Qnet()
    # here q_target and q are new instance objects of the class Qnet
    q_target.load_state_dict(q.state_dict())
    # state_dict() => In pytorch, the learnable parameters(i.e. weights and biases) of a model are contained in the model.param-
    # eters(). A state_dict is simply a python dictionary object that maps each layer to its parameter tensor. Only layers with
    # learnable parameters(convolutional layers, linear layers etc) and registered buffers(batchnorm's running mean) have entries
    # in the state_dict.
    # When saving a model for inference, it is only necessary to save the trained model's learned parameters.
    # A commmon pytorch convention is to save models using either .pt or .pth file extension.
    # load_state_dict => loads the model's parameters
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        s = env.reset()
        # reset() function resets and returns the starting frame
        # reset function returns an initial observation
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)          
            s_prime, r, done, info = env.step(a)
            # this line performs a random action, returns the new frame,reward and whether the game is over.
            # this step => gives new state and reward from the environment by applying the action
            # s_prime => an environment specific object representing your observation of the environment
            # r => amount of reward achieved by the previous action. The scale varies between environments, but the goal is
            # always to increase your total reward
            # done => whether its time to reset the environment again. Most tasks are divided up into well defined episodes 
            # and done being true indicates the episode has terminated.
            # info => diagnostic information useful for debugging. 
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r/100.0,s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break
            
        if memory.size()>2000:
            train(q, q_target, memory, optimizer)
        
        #from here we will begin the LTH
        
        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0
    env.close()

if __name__ == '__main__':
    main()
