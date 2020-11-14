from typing import (
    Tuple,
)
import numpy
import torch

from utils_types import (
    BatchAction,
    BatchDone,
    BatchNext,
    BatchReward,
    BatchState,
    Indices,
    TensorStack5,
    TorchDevice,
)

class SumTree(object):
    write = 0
    full = 0
    p_max = 1

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros( 2*capacity - 1 )
        self.data = numpy.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):  # 前向传播

        parent = (idx - 1) // 2
        #print(idx,self.tree[idx],change)

        self.tree[parent] += change   # p值更新后，其上面的父节点也都要更新

        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self.p_max=max(p,self.tree[self.capacity-1:].max())
        self._propagate(idx, change)

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:  # 叶结点存满了就从头开始覆写
            self.full = 1
            self.write = 0

    def _retrieve(self, idx, s):  # 检索s值
        left = 2 * idx + 1
        right = left + 1
        #print(idx)
        if 2 * left + 1 >= len(self.tree):  # 说明已经到叶结点了
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)  # 递归调用
        else:
            return self._retrieve(right, s-self.tree[left])

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

    def total(self):
        return self.tree[0]
    
class ReplayMemory(object):
    e = 0.01
    #a = 0.6
    
    def __init__(
            self,
            channels: int,
            capacity: int,
            device: TorchDevice,
    ) -> None:
        self.__device = device
        self.__capacity = capacity
        self.__size = 0
        self.__pos = 0
        
        self.Tree = SumTree(capacity)
        
        self.__m_states = torch.zeros(
            (capacity, channels, 84, 84), dtype=torch.uint8)
        self.__m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.__m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.__m_dones = torch.zeros((capacity, 1), dtype=torch.bool)
    
    def _getPriority(self, error):
        return error + self.e

    def add(self, error, sample):
        p = self._getPriority(error)
        self.Tree.add(error, sample) 
        
    def push(
            self,
            folded_state: TensorStack5,
            action: int,
            reward: int,
            done: bool,
    ) -> None:
        self.__m_states[self.__pos] = folded_state
        self.__m_actions[self.__pos, 0] = action
        self.__m_rewards[self.__pos, 0] = reward
        self.__m_dones[self.__pos, 0] = done

        self.__pos = (self.__pos + 1) % self.__capacity
        self.__size = max(self.__size, self.__pos)
        
        #print(self.__pos,self.Tree.full)
        
        #print(folded_state,action,reward,done)
        
        data = self.__pos
        #print(data)
        
        #if(self.Tree.p_max!=1):
        #    print(self.Tree.p_max)
        self.Tree.add(self.Tree.p_max,data)

    def sample(self, batch_size: int) -> Tuple[
            BatchState,
            BatchAction,
            BatchReward,
            BatchNext,
            BatchDone,
            Indices
    ]:
        

        num = self.Tree.total() / batch_size
        #print('val:',self.Tree.total())

        indices = []
            #print(self.Tree.total(),len(self.Tree.tree))
        for i in range(batch_size):
            a = num * i
            b = num * (i + 1)
            s = numpy.random.uniform(a,b)
                #print(a,b,abs(s))
            (idx, p, data) = self.Tree.get(abs(s))
                #print('get it',idx,p,data)
            indices.append(data)
            #print(type(b_state),type(b_next),type(b_action),type(b_reward),type(b_done))
            #print(len(b_state),len(b_next),len(b_action),len(b_reward),len(b_done))
        indices=torch.tensor(indices)
            
        b_state = self.__m_states[indices, :4].to(self.__device).float()
        b_next = self.__m_states[indices, 1:].to(self.__device).float()
        b_action = self.__m_actions[indices].to(self.__device)
        b_reward = self.__m_rewards[indices].to(self.__device).float()
        b_done = self.__m_dones[indices].to(self.__device).float()
        #print(b_state,b_next,b_action,b_reward,b_done)
        #print(b_state.size(),b_next.size(),b_action.size(),b_reward,b_done.size())
        #print(type(b_state),type(b_next),type(b_action),type(b_reward),type(b_done))
        return b_state, b_action, b_reward, b_next, b_done, indices

    def __len__(self) -> int:
        return self.__size
