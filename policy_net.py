import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNet(nn.Module):
    def __init__(self, n_input_size, n_actions, device):
        super().__init__()
        self.size = n_input_size
        self.device = device
        self.layerfc = nn.Sequential(
            nn.Linear(in_features=self.size, out_features=self.size),
            nn.ReLU()
        )
        self.layerfc2 = nn.Sequential(
            nn.Linear(in_features=self.size, out_features=self.size),
            nn.ReLU()
        )
        self.layer_no_activation = nn.Linear(in_features=self.size, out_features=n_actions)
        
    def forward(self, X):
        X = X.to(self.device)
        output = self.layerfc(X)
        output = self.layerfc2(output)
        output = self.layer_no_activation(output)
        return output

def Qloss(batch, policy_net, gamma=0.99, device="cpu"):
    current_states, actions, next_states, rewards, isgameons = batch
    lbatch = len(current_states)
    state_action_values = policy_net(current_states.view(lbatch,-1))
    state_action_values = state_action_values.gather(1, actions.unsqueeze(-1))
    state_action_values = state_action_values.squeeze(-1)
    #print(state_action_values.shape)
    next_state_values = policy_net(next_states.view(lbatch, -1))
    next_state_values = next_state_values.max(1)[0]
    
    next_state_values = next_state_values.detach()
    expected_state_action_values = next_state_values * gamma + rewards
    #print(next_state_values.shape)
    return nn.MSELoss()(state_action_values, expected_state_action_values)

