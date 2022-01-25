"""
Demonstrations Dataset.
"""

from torch.utils.data import Dataset

class DemonstrationsDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self, index):
        state = self.data['state'][index]
        new_state = self.data['new_state'][index]
        action = self.data['action'][index]
        reward = self.data['reward'][index]
        
        return {'state': state, 'new_state': new_state, 'action': action, 'reward': reward}
    
    def __len__(self):
        return len(self.data['reward'])