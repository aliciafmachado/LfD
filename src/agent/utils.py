import torch

def select_greedy_action(state, model):
    state = torch.from_numpy(state).float()
    out = model(state)
    if len(out) > 1:
        out = out[0]
    return out.argmax().item()