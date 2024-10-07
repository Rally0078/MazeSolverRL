import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import collections
#ExperienceReplay will store Transition objects

Transition = collections.namedtuple('Experience',
                                    field_names=['current_state', 'action',
                                                 'next_state', 'reward',
                                                 'is_game_on'])

class Agent:
    #Init method
    def __init__(self, environment, memory_buffer, n_actions):
        self.env = environment  #self.env will hold the instance of environment, which contains state
        self.experiences = memory_buffer
        self.n_actions = n_actions
        self.total_reward = 0
        self.min_reward = -1.0
        self.isgameon = True
    
    #Given policy net, make a move with a given exploration factor epsilon
    def make_a_move(self, policy_net, epsilon, device="cpu"):
        action = self.select_action(policy_net, epsilon, device)
        current_state = self.env.state()
        next_state, reward, self.isgameon = self.env.state_update(action)
        self.total_reward += reward

        if self.total_reward < self.min_reward:
            self.isgameon = False
        if not self.isgameon:
            self.total_reward = 0
        
        transition = Transition(current_state=current_state,
                                action=action,
                                next_state=next_state,
                                reward=reward,
                                is_game_on=self.isgameon)
        self.experiences.push(transition)

    #Select action by randomly picking one(exploration) or by using known knowledge of the policy(exploitation)
    def select_action(self, policy_net, epsilon, device="cpu"):
        state = torch.Tensor(self.env.state()).to(device).view(1,-1)
        qvalues = policy_net(state).cpu().detach().numpy().squeeze()
        if np.random.random() < epsilon:
            action = np.random.randint(self.n_actions, size=1)[0]
        else:
            action = int(np.argmax(qvalues))
        return action


class RatAgent(Agent):
    def __init__(self, environment, memory_buffer, n_actions):
        super().__init__(environment, memory_buffer, n_actions)
        self.min_reward = -self.env.maze.size * 0.05 * 2 - self.env.maze.size * 0.2 

    def plot_policy_map(self, policy_net, filename, offset):
        policy_net.eval()
        with torch.no_grad():
            fig, ax = plt.subplots()
            ax.imshow(self.env.maze, 'Greys')

            for free_cell in self.env.allowed_states:
                self.env.current_position = np.asarray(free_cell)
                qvalues = policy_net(torch.Tensor(self.env.state()).view(1,-1).to('cuda'))
                action = int(torch.argmax(qvalues).detach().cpu().numpy())
                policy = self.env.directions[action]

                ax.text(free_cell[1]-offset[0], free_cell[0]-offset[1], policy)
            ax = plt.gca();

            plt.xticks([], [])
            plt.yticks([], [])

            ax.plot(self.env.goal_position[1], self.env.goal_position[0],
                    'bs', markersize = 4)
            plt.savefig(filename, dpi = 300, bbox_inches = 'tight')
            #plt.show()
            plt.close()