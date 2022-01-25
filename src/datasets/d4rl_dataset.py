"""
Pytorch dataset for datasets in the format provided by d4rl.
"""

from torch.utils.data import Dataset

class D4RLDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self, index):
        state = self.data['observations'][index]
        new_state = None
        action = self.data['actions'][index]
        reward = self.data['rewards'][index]
        
        return {'state': state, 'new_state': new_state, 'action': action, 'reward': reward}
    
    def __len__(self):
        return len(self.data['rewards'])