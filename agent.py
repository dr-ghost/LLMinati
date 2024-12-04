import torch
import torch.nn as nn
import torch.optim as optim 
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from tensordict import TensorDict
import datetime
import time

from env import PuzzleEnv
from model import OPT1, PPOLoss, OPT1Noisy
from utils import batchestize, unbatchestize

from transformers import AutoModel, AutoTokenizer


POLICY_UPDATE = 2
DISPLAY_UPDATE = 200
SAVE_MODEL = 1000

class Agent: # To Be changed
    def __init__(self, deployed : bool = False) -> None:
        
        self.loss = PPOLoss()
        
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        dec = "policy_d_82000"
        
        self.decodr = OPT1Noisy(temperature=0.0001)


        self.decodr.load_state_dict(torch.load(f"./model_saves/{dec}.pt"))
        
        self.decodr.to(self.device)
        
        self.old_decodr = OPT1Noisy(temperature=0.0001).to(self.device)
        self.old_decodr.load_state_dict(self.decodr.state_dict())
        
        self.model = AutoModel.from_pretrained("./model_saves/smol_train_30000").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("./model_saves/smol_train_30000")
        
        self.deployed = deployed
        
        self.optimizer = optim.Adam(self.decodr.parameters(), lr=0.0007)
        
    def smol_embeddings(self, state, objects):
        lengths = state["state_platform"][:, 3]
        lengths_str = ", ".join(map(str, lengths.tolist()))
        
        objects_str = ", ".join([f"{obj[0]} (size: {obj[1]})" for obj in objects])
        
        prompt = (
            f"Given length of platforms with capacities {{{lengths_str}}} and objects "
            f"{{{objects_str}}}, assign each object to exactly one space such that the total "
            f"length of objects in each space does not exceed its capacity, and each object "
            f"is placed into one space."
        )
        
        print(prompt)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.device)
        
        outputs = self.model(**inputs)
        
        embeddings = outputs.last_hidden_state

        return embeddings
        
        
    def __call__(self, state : TensorDict) -> TensorDict:
        """returns action for a given context/state"""
        
        batchestize(state)
        
        #state = state.to(self.device)
        smol_state = self.smol_embeddings(state)
            
        action = self.decodr(state, smol_state)
        
        unbatchestize(state)
        
        return action
    
    def log_dist_probs(self, action_logits):
        dist = torch.distributions.Independent(torch.distributions.Normal(action_logits, 1.0), 1)
        
        return dist.log_prob(action_logits)
    
    def entropy(self, action_logits):
        dist = torch.distributions.Independent(torch.distributions.Normal(action_logits, 1.0), 1)
        
        return dist.entropy()
    
    def inference(self, env : PuzzleEnv, objects):
        start = env._reset(inference=True, objects=objects)
        
        # print(start["state_platform"].shape, start["state_object"].shape)
        
        smol_state = self.smol_embeddings(start, objects)
        
        batchestize(start)
        
        action_probs = self.decodr(start, smol_state)
            
        unbatchestize(start)
       
        action = torch.argmax(action_probs, dim=1)
        env.render(action, filepath="./episode_replay/episode")
        
    def inference_(self, env : PuzzleEnv):
        start = env._reset()
        
        # print(start["state_platform"].shape, start["state_object"].shape)
        
        objects = [(np.random.choice(["Puzzle_1", "Puzzle_2", "Puzzle_3", "Puzzle_4", "Puzzle_5", "Puzzle_6", "Puzzle_7"]), 200 * np.random.random() + 500) for i in range(start["state_object"].shape[0])]
        
        smol_state = self.smol_embeddings(start, objects)
        
        batchestize(start)
        
        action_probs = self.decodr(start, smol_state)
            
        unbatchestize(start)
       
        action = torch.argmax(action_probs, dim=1)
        env.render(action, filepath="./episode_replay/episode")

        
    
    def train(self, episodes: int, env : PuzzleEnv, log_dir : str = "./logs/train/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")):
        """
        one Puzzle config per episode to train on
        
        1) Ideas to have multiple epochs per episode
        2) Recycle the same configurations (considering real-like configs to train on) 
        """
        reward_plot = []
        episode_plot = []
        loss_plot = []
        reward_avg = []
        
        fig, ax = plt.subplots()
        
        plt.ion()
        
        batch_size = 128
        
        for episode in range(episodes):
            # opt.zero_grad()
        
            # loss = 0
        
            # for j in range(batch_size):       

            #     action_logits = policy(obs)
                
            #     loss += nn.MSELoss()(action_logits, target)
        
            # loss.backward()
        
            # opt.step()
        
            # loss_plot.append(loss.cpu().item())
            # episode_plot.append(i)
        
            # if i % 50:
            #     ax.clear()
            #     ax.plot(episode_plot, loss_plot, label="Reward")
                
            #     plt.savefig("/content/NVAI_Object_Placement/episode_replay/plot.png")
        
            # if i % 100:
            #     ob_ = env._reset()
                
            #     batchestize(ob_)
                  
            #     print(policy(ob_))
            
            self.optimizer.zero_grad()
            
            #loss = 0
                        
            reward_d = 0
            
            self.decodr.reset_noise()
            
            for j in range(batch_size):
                state = env._reset()
                smol_state = self.smol_embeddings(start)
                batchestize(start)
                    
                action_probs = self.decodr(start, smol_state)
            
                unbatchestize(start)
       
                action_probs = torch.argmax(action_probs, dim=1)
            
                # log_probs_old_d = log_probs_old_d + self.log_dist_probs(self.old_decodr(state))
            
                # log_probs_x_d = log_probs_x_d + self.log_dist_probs(action_probs)
            
                # entropy_d = entropy_d + self.entropy(action_probs)
            
                #action = torch.argmax(action_probs, dim=1)
                        
                unbatchestize(state)

                if episode % 5 == 0:
                    with torch.no_grad():
                        reward_d = reward_d + env._step(action_probs)["reward"]
                else:
                    reward_d = reward_d + env._step(action_probs)["reward"]
                
                #loss = loss + self.loss(reward_d, log_probs_old, log_probs_x, entropy)
            
            state_dct_save = self.decodr.state_dict().copy()
            
            #loss = loss / batch_size
            
            reward_d = reward_d / batch_size
            
            start = env._reset()
            smol_state = self.smol_embeddings(start)
            batchestize(start)
            
            #loss = 0
            
            action_probs = self.decodr(start, smol_state)
            
            log_probs_old_d = self.log_dist_probs(self.old_decodr(start, smol_state))
            log_probs_x_d = self.log_dist_probs(action_probs)
            
            entropy_d = self.entropy(action_probs)
            
            unbatchestize(start)
            
            
            loss = self.loss(reward_d, log_probs_old_d, log_probs_x_d, entropy_d)            
            
            loss.backward()
            
            self.optimizer.step()
            
            if episode % POLICY_UPDATE == 0:
                with torch.no_grad():
                    self.old_decodr.load_state_dict(state_dct_save)
            

            # reward_plot.append(reward_d.cpu().numpy())
            
            # if len(reward_plot) > 100:
            #     del reward_plot[0]
            
            # reward_avg = np.mean(reward_plot)
            
            reward_avg.append(reward_d.detach().cpu().numpy())
            
            if len(reward_avg) > 10:
                del reward_avg[0]
            
            
            
            reward_plot.append(np.mean(reward_avg))
            
            if len(reward_plot) > 100:
                del reward_plot[0]
                
            loss_plot.append(loss.cpu().item())
            
            if len(loss_plot) > 100:
                del loss_plot[0]
            
            episode_plot.append(episode)
            
            if len(episode_plot) > 100:
                del episode_plot[0]
            
            if episode % DISPLAY_UPDATE == 0:
                ax.clear()
                ax.plot(episode_plot, reward_plot, label="Reward")
                
                plt.savefig("./episode_replay/plot.png")
                
                ax.clear()
                
                ax.plot(episode_plot, loss_plot, label="Loss")
                
                plt.savefig("./episode_replay/loss.png")
                        
            
            with torch.no_grad():
                if episode % DISPLAY_UPDATE == 0:
                    action = torch.argmax(action_probs, dim=1)
                    env.render(action, filepath="./episode_replay/episode")
                    
                    batchestize(state)
                    print(self.decodr(state))
                    unbatchestize(state)
                
                if episode % SAVE_MODEL == 0:
                    torch.save(self.decodr.state_dict(), f"./model_saves/policy_s_{episode}.pt")
                    self.model.save_pretrained(f"./model_saves/smol_train_{episode}")
                    
        return reward_plot