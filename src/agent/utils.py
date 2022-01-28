import torch
from torch.nn.functional import one_hot

def select_greedy_action(state, model):
    state = torch.from_numpy(state).float()
    out = model(state)
    if len(out) > 1:
        out = out[0]
    return out.argmax().item()


def indicator_fn(action_e, n_actions, loss_different=0.8):
    l_val = torch.full((action_e.shape[0], n_actions), loss_different)
    action_e_mask = one_hot(action_e.squeeze(-1), num_classes=n_actions).bool()
    l_val[action_e_mask] = 0
    return l_val