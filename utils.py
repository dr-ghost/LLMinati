from tensordict import TensorDict

def batchestize(state : TensorDict):
    for i in state.keys():
        state[i] = state[i].unsqueeze(0)
        
def unbatchestize(state : TensorDict):
    for i in state.keys():
        state[i] = state[i].squeeze(0)

def modulus(x : int, m : int):
    res = x % m
    return res if res >= 0 else -res

def modulus_arr(arr : list, x : int, m : int):
    res = x % m
    return arr[res]

def max_arr(lst1 : list, lst2 : list):
    return [max(i, j) for i, j in zip(lst1, lst2)]

alpha = 0.96