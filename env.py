import torch
from torch import tensor
from torch import functional as F
import random

import torchrl
from torchrl.envs.common import EnvBase
from torchrl.data import CompositeSpec, BoundedTensorSpec, DiscreteTensorSpec, UnboundedContinuousTensorSpec, UnboundedDiscreteTensorSpec
from torchrl.envs.transforms import TransformedEnv
from matplotlib import pyplot as plt
from tensordict import TensorDict
import matplotlib.patches as patches
from copy import deepcopy

from Embeddings.feature_extraction import GPlatformEmbeddings, GObjectEmbeddings
from utils import modulus, modulus_arr, max_arr

import math
import numpy as np

import warnings
warnings.filterwarnings('ignore')


PLATFORM_WIDTH = 620

class PuzzleEnv(EnvBase):
    _batch_locked = True

    def __init__(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), batch_size=64, seed=None, deployed=False):
        
        super(PuzzleEnv, self).__init__(batch_size=[batch_size])
        self.deployed = deployed
        self.state = None
        self.device = device
        
        self.to(self.device)
        if seed is not None:
            self._set_seed(seed)
        
        self.object_specs = {
            'Puzzle_1': {'width': 575, 'min_length': 150, 'max_length': 300, 'max_count': 2, 'priority': 1},
            'Puzzle_2': {'width': 575, 'min_length': 400, 'max_length': 1200, 'max_count': 10, 'priority': 2},
            'Puzzle_3': {'width': 575, 'min_length': 400, 'max_length': 1200, 'max_count': 10, 'priority': 1},
            'Puzzle_4': {'width': 575, 'min_length': 350, 'max_length': 1050, 'max_count': 10, 'priority': 3},
            'Puzzle_5': {'width': 575, 'min_length': 300, 'max_length': 350, 'max_count': 3, 'priority': 2},
            'Puzzle_6': {'width': 575, 'min_length': 470, 'max_length': 600, 'max_count': 3, 'priority': 1},
            'Puzzle_7': {'width': 575, 'min_length': 450, 'max_length': 650, 'max_count': 1, 'priority': 1},
            'Puzzle_8': {'width': 575, 'min_length': 700, 'max_length': 1050, 'max_count': 1, 'priority': 1},
            'Puzzle_9': {'width': 575, 'min_length': 550, 'max_length': 650, 'max_count': 1, 'priority': 3},
            'Puzzle_10': {'width': 575, 'min_length': 600, 'max_length': 600, 'max_count': 1, 'priority': 2},
            'Puzzle_11': {'width': 575, 'min_length': 600, 'max_length': 600, 'max_count': 1, 'priority': 1},
            'Puzzle_12': {'width': 575, 'min_length': 450, 'max_length': 650, 'max_count': 1, 'priority': 3},
            'Puzzle_13': {'width': 575, 'min_length': 450, 'max_length': 600, 'max_count': 1, 'priority': 2},
            'Puzzle_14': {'width': 575, 'min_length': 450, 'max_length': 750, 'max_count': 1, 'priority': 1},
            'Puzzle_15': {'width': 575, 'min_length': 700, 'max_length': 1200, 'max_count': 1, 'priority': 1},
            'Puzzle_16': {'width': 575, 'min_length': 450, 'max_length': 600, 'max_count': 1, 'priority': 1},
            'Puzzle_17': {'width': 575, 'min_length': 450, 'max_length': 600, 'max_count': 1, 'priority': 2}
        }
        
        self.platform_emb = GPlatformEmbeddings()
        self.object_emb = GObjectEmbeddings()

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        random.seed(seed)
    
    def _step(self, action):
        state_platform = self.state["state_platform"]
        state_object = self.state["state_object"]
                
        r = self.calculate_reward(action)
        
        if self.deployed:
            return TensorDict({"reward" : r, "possible" : self.is_possible_placement(state_platform, state_object, action), "state" : self.generate_state(state_platform, state_object, action)})
            
        return TensorDict({"reward" : r}).to(self.device)
    
    def _reset(self, state=None, inference=False, objects=None):
        if state == None:
            if inference:
                self.state = self.generate_state(objects, hooks={"min_room_width" : 3000, "max_room_width" : 3500, "min_room_length" : 4000, "max_room_length" : 4500, "no_corner_probability" : 0.5, "max_free_platform_start" : 620, "min_free_platform_start" : 540, "zero_start_probability" : 0.6, "min_no_corner_start" : PLATFORM_WIDTH + PLATFORM_WIDTH//2.5, "max_no_corner_start" : PLATFORM_WIDTH + PLATFORM_WIDTH, "straight_corner_free" : PLATFORM_WIDTH})
            else:
                self.state = self.generate_random_state(hooks={"min_room_width" : 3000, "max_room_width" : 3500, "min_room_length" : 4000, "max_room_length" : 4500, "no_corner_probability" : 0.5, "max_free_platform_start" : 620, "min_free_platform_start" : 540, "zero_start_probability" : 0.6, "min_no_corner_start" : PLATFORM_WIDTH + PLATFORM_WIDTH//2.5, "max_no_corner_start" : PLATFORM_WIDTH + PLATFORM_WIDTH, "straight_corner_free" : PLATFORM_WIDTH})
        else:
            self.state = state
        
        #Generate Specifications according to state
        #self._update_spec(self.state)
            
        self.state = self.state.to(self.device)
        return self.state
        
    # def _make_spec(self):
    #     #self.reward_spec = UnboundedContinuousTensorSpec(shape=(1, ), dtype=torch.float32)
    #     pass
    # def _update_spec(self, state):
    #     # n_objects = state["state_object"].shape[-1]
        
    #     # self.observation_spec = CompositeSpec(
    #     #     state_platform=BoundedTensorSpec(low=0, high=8, shape=(self.batch_size, 6, n_objects), dtype=torch.float32),
    #     #     state_object=BoundedTensorSpec(low=0, high=8, shape=(self.batch_size, 7, n_objects), dtype=torch.float32)
    #     # )
        
    #     # self.action_spec = BoundedTensorSpec(low=0, high=8, shape=(self.batch_size, n_objects), dtype=torch.float32)
    #     pass
        
    # def set_seed(self, seed : int):
    #     rng = torch.manual_seed(seed)
    #     self.rng = rng
    # def place_objects_on_platforms(self, state_platforms, objs):
    #     placements = []  # Store the position of each object on its platform
    #     platform_free_space = [{  # Initialize free space records for each platform
    #         'remaining_length': platform[3].item(),  # Length of the platform
    #         'x': platform[1].item(),  # X-coordinate of the platform
    #         'y': platform[2].item(),  # Y-coordinate of the platform
    #         'objects': []  # Objects placed on this platform
    #     } for platform in state_platforms]

    #     for obj in objs:
    #         obj_width = random.randint(obj["min_size"].item(), obj["max_size"].item())  # Determine size within specified range
    #         placed = False
            
    #         # Sort platforms based on remaining length and priority (not implemented, could use priority if provided)
    #         sorted_platforms = sorted(platform_free_space, key=lambda x: x['remaining_length'], reverse=True)

    #         for platform in sorted_platforms:
    #             # Check if the object fits on the current platform
    #             if platform['remaining_length'] >= obj_width:
    #                 # Place the object on the platform
    #                 obj_placement = {
    #                     'object_index': obj['object_index'].item(),
    #                     'x': platform['x'],  # Start position on the platform
    #                     'y': platform['y'],
    #                     'width': obj_width
    #                 }
    #                 placements.append(obj_placement)
    #                 platform['objects'].append(obj_placement)
    #                 platform['x'] += obj_width  # Update start x for next object
    #                 platform['remaining_length'] -= obj_width  # Update remaining length
    #                 placed = True
    #                 break
            
    #         if not placed:
    #             # If object couldn't be placed on any platform, log or handle the case (could be skipped or errored)
    #             print(f"Could not place object {obj['object_index'].item()} of width {obj_width}")

    #     return placements

    def render(self, action = None, filepath : str ="./episode_replay/episode"):
        platform_width = PLATFORM_WIDTH
        
        fig, ax = plt.subplots()
                
        
        dim0 = deepcopy(self.state["room_dimensions"][0]).cpu()
        dim1 = deepcopy(self.state["room_dimensions"][1]).cpu()
        
        state_platform = deepcopy(self.state["state_platform"]).cpu()
        state_object = deepcopy(self.state["state_object"]).cpu()
        
        ax.set_xlim(0, dim0)
        ax.set_ylim(0, dim1)
        
        plat_off_lst = [0 for i in state_platform]
        
        assert not ("state_platform" not in self.state or len(state_platform)) == 0,\
            "No platforms to render. Call reset first"

        # Draw platforms
        for platform in state_platform:
            x, y, length, vertical = platform[4].item(), platform[5].item(), platform[3].item(), bool(platform[6].item())
            if vertical:
                rect = patches.Rectangle((x, y), platform_width, length, linewidth=1, edgecolor='r', facecolor='none')
            else:
                rect = patches.Rectangle((x, y), length, platform_width, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        # Draw objects
        if action is not None:
            for obj_id in range(state_object.shape[0]):
                platform_id = action[obj_id]
                min_l = state_object[obj_id, 1]
                
                x, y = state_platform[platform_id, 4], state_platform[platform_id, 5]
                
                if bool(state_platform[platform_id, 6].item()):
                    rect = patches.Rectangle((x, y + plat_off_lst[platform_id]), platform_width, min_l, linewidth=1, edgecolor='k', facecolor='none')
                    plat_off_lst[platform_id] += min_l
                    ax.add_patch(rect)
                    
                    tag_x = x + platform_width / 2
                    tag_y = y + plat_off_lst[platform_id] - min_l / 2
                    ax.text(tag_x, tag_y, list(self.object_specs.keys())[int(state_object[obj_id, 0].item())], ha='center', va='center')
                else:
                    rect = patches.Rectangle((x + plat_off_lst[platform_id], y), min_l, platform_width, linewidth=1, edgecolor='k', facecolor='none')
                    plat_off_lst[platform_id] += min_l
                    ax.add_patch(rect)
                    
                    tag_x = x + plat_off_lst[platform_id] - min_l / 2
                    tag_y = y + platform_width / 2
                    ax.text(tag_x, tag_y, list(self.object_specs.keys())[int(state_object[obj_id, 0].item())], ha='center', va='center')
                
        plt.savefig(f"{filepath}.png")
        plt.close()


    def close(self):
        print("Closing environment")
            
    def calculate_reward(self, action_probs):
        
        # for i in range(self.state["state_object"].shape[0]):
        #     for j in range(self.state["state_object"].shape[1]):
        #         tot_len[j] = tot_len[j] + action_probs[i, j] * self.state["state_object"][i, 1]
        
        tot_len = torch.matmul(action_probs.t(), self.state["state_object"][:, 1])
        
        tmp = self.state["state_platform"][:, 3] - tot_len
        
        R = torch.sum(tmp * torch.where(tmp > 0, 0 * torch.ones_like(tmp), 1 * torch.ones_like(tmp)))
        
        # for i in range(len(tot_len)):
        #     R = R + self.state["state_platform"][i, 3] - tot_len[i]
            
        #R = R + torch.sum(action_probs * torch.log(action_probs)) * 500
        
        return R

        # device = action_probs.device
        # weights = torch.tensor([10, 10, -10, -10], device=device)
        # differences = action_probs[:, 0] - action_probs[:, 1]
        # weighted_differences = differences * weights
        # rw = weighted_differences.sum()
        # return rw
            
            
        
        
        
        
    def is_possible_placement(self, action):
        # Create a list of objects for each platform
        
        glist = [[] for i in range(self.state["state_platform"].shape[0])]
        
        for i in range(self.state["state_object"].shape[0]):
            glist[action[i]].append(i)
            
        correct_placement, incorrect_placement = 0, 0
        for i in range(self.state["state_platform"].shape[0]):
            tot = 0
            for j in glist[i]:
                tot += self.state["state_object"][j, 1]
            
            if tot <= self.state["state_platform"][i, 3]:
                if tot != 0:
                    correct_placement += 1
            else:
                incorrect_placement += 1
        
        
        return incorrect_placement == 0, correct_placement, incorrect_placement
        
    
    def check_overlap(self, new_rect, rects, min_distance=10):
        """Checks if the new rectangle overlaps with existing rectangles considering minimum distance."""
        
        new_x1, new_y1, new_x2, new_y2 = new_rect
        for rect in rects:
            x1, y1, x2, y2 = rect
            if not (new_x2 + min_distance < x1 or new_x1 > x2 + min_distance or
                    new_y2 + min_distance < y1 or new_y1 > y2 + min_distance):
                return True
        return False

    def generate_random_state(self, max_attempts=100, hooks : dict = {}):
        """
        Generates random intial Puzzle configurations
        """
        # platform_width = 620  # Fixed platform width in mm
        # min_platform_length = 810  # Minimum platform length in mm
        # max_platform_length = 1410 # Maximum platform lenght in mm
        # side_gap = 620  # Minimum distance from the side walls in mm
        
        # room_length = 4000#random.randint(4000, 4500)  # Random room length between 2000mm and 8000mm
        # room_width = 3000#random.randint(3000, 4000)  # Random room width between 800mm and 4000mm

        # total_platforms = random.randint(2, 4)  # Ensure 2 to 5 platforms are planned
        # walls = [0, 1, 2, 3]  # Possible walls
        # random.shuffle(walls)
        # used_walls = walls  # Use only three walls
        
        # wall_index = {0:0, 1:0, 2:0, 3:0}

        # state_platforms = []
        # platform_rects = []
        
        # tot_plat_len = 0
        # for plt_idx in range(total_platforms):
        #     attempts = 0
        #     while attempts < max_attempts:
        #         edge_choice = random.choice(used_walls)
        #         available_length = room_width if edge_choice in [2, 3] else room_length
        #         if available_length <= 2 * side_gap + min_platform_length:
        #             continue  # Try next iteration if not enough space

        #         length = random.randint(min_platform_length, available_length - 2 * side_gap)
        #         position = random.randint(side_gap, available_length - length - side_gap)

        #         if edge_choice in [0, 1]:  # Top or Bottom
        #             x_coord = position
        #             y_coord = 0 if edge_choice == 0 else room_width - platform_width
        #         else:  # Left or Right
        #             x_coord = 0 if edge_choice == 2 else room_length - platform_width
        #             y_coord = position

        #         rect = (x_coord, y_coord, x_coord + length, y_coord + platform_width)

        #         if not self.check_overlap(rect, platform_rects, side_gap):
        #             platform_rects.append(rect)
        #             index = len(state_platforms)
        #             wall_number = edge_choice
        #             wall_idx = wall_index[edge_choice]
        #             vertical = edge_choice >= 2
        #             platform_properties = torch.tensor([index, wall_number, wall_idx, length, x_coord, y_coord, vertical], dtype=torch.float32)
        #             state_platforms.append(platform_properties)

        #             tot_plat_len += length
        #             break
        #         attempts += 1

        # attempts = 0

        # while attempts < max_attempts:
        #     obj_no = random.randint(2, 7)
        #     objs = []

        #     tot_obj_len = 0

        #     for i in range(obj_no):
        #         idx = random.randint(0, len(list(self.object_specs.keys())) - 1)
        #         key = list(self.object_specs.keys())[idx]
                
        #         tot_obj_len += self.object_specs[key]["min_length"]

        #         objs.append(self.object_emb({
        #             "object_index": torch.tensor([idx]).unsqueeze(0),
        #             "min_size": torch.tensor([self.object_specs[key]["min_length"]]).unsqueeze(0),
        #             "max_size": torch.tensor([self.object_specs[key]["max_length"]]).unsqueeze(0),
        #             "priority": torch.tensor([self.object_specs[key]["priority"]]).unsqueeze(0),
        #             "repeat": torch.tensor([self.object_specs[key]["max_count"] > 1]).unsqueeze(0),
        #             "min_count": torch.tensor([1]).unsqueeze(0),
        #             "max_count": torch.tensor([self.object_specs[key]["max_count"]]).unsqueeze(0)
        #         }))
                
        #     if tot_obj_len <= tot_plat_len:
        #         break
            
        #     attempts += 1

        # return TensorDict({
        #     "state_platform": torch.stack(state_platforms) if state_platforms else torch.tensor([]),
        #     "state_object": torch.stack(objs).squeeze(-1),
        #     "room_dimensions": torch.tensor([room_length, room_width])
        # })
        
        #room dimensions
        room_width = random.randint(hooks["min_room_width"], hooks["max_room_width"])
        room_length = random.randint(hooks["min_room_length"], hooks["max_room_length"])

        #platform
        platform_width = PLATFORM_WIDTH

        #random walls
        walls = [0, 1, 2, 3]
        random.shuffle(walls)
        no_of_walls = random.randint(1, 4)
        walls = walls[:no_of_walls]
        
        if no_of_walls == 3:
            if 3 in walls:
                if 1 in walls:
                    if 0 in walls:
                        walls = [3, 0, 1]
                    else:
                        walls = [1, 2, 3]
                else:
                    walls = [2, 3, 0]
            else:
                walls.sort()
        elif no_of_walls == 2:
            if 3 in walls:
                if 1 in walls:
                    walls = [1, 3]
                elif 2 in walls:
                    walls = [2, 3]
                else:
                    walls = [3, 0]
            else:
                walls.sort()
        else:
            walls.sort()

        wall_platform_space = [[-1, -1] for i in range(4)]

        #distribution
        np.random.uniform(540, 550)
        np.random.uniform(0, math.pi)

        #corners
        if no_of_walls == 4:
            n_corners = 4

            # hypotenuse_corner = np.random.uniform(540, 550, n_corners)
            # angle_corner = np.random.uniform(0, math.pi/2 + hooks["no_corner_probability"], n_corners)

            # for corner_no in range(n_corners):

            #     wall_ind0 = modulus(corner_no - 1, no_of_walls)
            #     wall_ind1 = modulus(corner_no + 1, no_of_walls)

            #     if angle_corner[corner_no] > math.pi/2:
            #         #if no corner then the maker leaves platform_width on both of the adjoining walls

            #         wall_platform_space[wall_ind0][1] = max(wall_platform_space[wall_ind0][1], platform_width)
            #         wall_platform_space[wall_ind1][0] = max(wall_platform_space[wall_ind0][0], platform_width)

            #         print()

            #     elif math.pi/2>angle_corner[corner_no]>0:
            #         # if a normal corner, we have to calculate markers using trigonometry.

            #         wall_platform_space[wall_ind0][1] = max(wall_platform_space[wall_ind0][1], hypotenuse_corner[corner_no] * math.cos(angle_corner[corner_no]) + platform_width)
            #         wall_platform_space[wall_ind1][0] = max(wall_platform_space[wall_ind0][0], hypotenuse_corner[corner_no] * math.cos(angle_corner[corner_no]) + platform_width)
                
            #     elif angle_corner[corner_no] == math.pi/2:
            #         wall_platform_space[wall_ind0][1] = 0
            #         wall_platform_space[wall_ind1][0] = hooks["straight_corner_free"]

            #     elif angle_corner[corner_no] == 0:
            #         wall_platform_space[wall_ind0][1] = hooks["straight_corner_free"]
            #         wall_platform_space[wall_ind1][0] = 0
            
            # for wall_no in walls:
            #     if wall_platform_space[wall_no][0] == -1:
            #         wall_platform_space[wall_no][0] = random.randint(0, hooks["max_free_platform_start"])
            #     if wall_platform_space[wall_no][1] == -1:
            #         wall_platform_space[wall_no][1] = random.randint(0, hooks["max_free_platform_start"])

            for corner_no in range(n_corners):
                
                wall_ind0 = modulus_arr(walls, corner_no, no_of_walls)
                wall_ind1 = modulus_arr(walls, corner_no + 1, no_of_walls)


                corner_type = random.choice([0, 1, 2, 3])
                
                if corner_type == 0:
                    wall_platform_space[wall_ind0][1] = random.randint(hooks["min_no_corner_start"], hooks["max_no_corner_start"])
                    wall_platform_space[wall_ind1][0] = random.randint(hooks["min_no_corner_start"], hooks["max_no_corner_start"])
                    
                elif corner_type == 1:
                    wall_platform_space[wall_ind0][1] = 0
                    wall_platform_space[wall_ind1][0] = hooks["straight_corner_free"] + platform_width
                    
                elif corner_type == 2:
                    wall_platform_space[wall_ind0][1] = hooks["straight_corner_free"] + platform_width
                    wall_platform_space[wall_ind1][0] = 0
                    
                else: #corner_type == 3
                    hypotenuse_corner = np.random.uniform(540, 550, 1)
                    angle_corner = np.random.uniform(math.pi/12, math.pi/2 - math.pi/12, 1)

                    wall_platform_space[wall_ind0][1] = hypotenuse_corner[0] * math.cos(angle_corner[0]) + platform_width
                    wall_platform_space[wall_ind1][0] = hypotenuse_corner[0] * math.sin(angle_corner[0]) + platform_width
            
            for wall_no in walls:
                if wall_platform_space[wall_no][0] == -1:
                    wall_platform_space[wall_no][0] = random.randint(hooks["min_free_platform_start"], hooks["max_free_platform_start"]) if random.random() > hooks["zero_start_probability"] else 0
                if wall_platform_space[wall_no][1] == -1:
                    wall_platform_space[wall_no][1] = random.randint(hooks["min_free_platform_start"], hooks["max_free_platform_start"]) if random.random() > hooks["zero_start_probability"] else 0


        elif no_of_walls == 3:
            n_corners = 2

            # hypotenuse_corner = np.random.uniform(540, 550, n_corners)
            # angle_corner = np.random.uniform(0, math.pi/2 + hooks["no_corner_probability"], n_corners)

            for corner_no in range(n_corners):
                
                wall_ind0 = modulus_arr(walls, corner_no, no_of_walls)
                wall_ind1 = modulus_arr(walls, corner_no + 1, no_of_walls)


                corner_type = random.choice([0, 1, 2, 3])
                
                if corner_type == 0:
                    wall_platform_space[wall_ind0][1] = random.randint(hooks["min_no_corner_start"], hooks["max_no_corner_start"])
                    wall_platform_space[wall_ind1][0] = random.randint(hooks["min_no_corner_start"], hooks["max_no_corner_start"])
                    
                elif corner_type == 1:
                    wall_platform_space[wall_ind0][1] = 0
                    wall_platform_space[wall_ind1][0] = hooks["straight_corner_free"] + platform_width
                    
                elif corner_type == 2:
                    wall_platform_space[wall_ind0][1] = hooks["straight_corner_free"] + platform_width
                    wall_platform_space[wall_ind1][0] = 0
                    
                else: #corner_type == 3
                    hypotenuse_corner = np.random.uniform(540, 550, 1)
                    angle_corner = np.random.uniform(math.pi/12, math.pi/2 - math.pi/12, 1)

                    wall_platform_space[wall_ind0][1] = hypotenuse_corner[0] * math.cos(angle_corner[0]) + platform_width
                    wall_platform_space[wall_ind1][0] = hypotenuse_corner[0] * math.sin(angle_corner[0]) + platform_width
            
            for wall_no in walls:
                if wall_platform_space[wall_no][0] == -1:
                    wall_platform_space[wall_no][0] = random.randint(hooks["min_free_platform_start"], hooks["max_free_platform_start"]) if random.random() > hooks["zero_start_probability"] else 0
                if wall_platform_space[wall_no][1] == -1:
                    wall_platform_space[wall_no][1] = random.randint(hooks["min_free_platform_start"], hooks["max_free_platform_start"]) if random.random() > hooks["zero_start_probability"] else 0

            
        elif no_of_walls == 2:
            wall_ind0 = walls[0]
            wall_ind1 = walls[1]

            if modulus(wall_ind1 - wall_ind0, 4) == 1:
                corner_type = random.choice([0, 1, 2, 3])
                
                if corner_type == 0:
                    wall_platform_space[wall_ind0][1] = random.randint(hooks["min_no_corner_start"], hooks["max_no_corner_start"])
                    wall_platform_space[wall_ind1][0] = random.randint(hooks["min_no_corner_start"], hooks["max_no_corner_start"])
                    
                elif corner_type == 1:
                    wall_platform_space[wall_ind0][1] = 0
                    wall_platform_space[wall_ind1][0] = hooks["straight_corner_free"] + platform_width
                    
                elif corner_type == 2:
                    wall_platform_space[wall_ind0][1] = hooks["straight_corner_free"] + platform_width
                    wall_platform_space[wall_ind1][0] = 0
                    
                else: #corner_type == 3
                    hypotenuse_corner = np.random.uniform(540, 550, 1)
                    angle_corner = np.random.uniform(math.pi/12, math.pi/2 - math.pi/12, 1)

                    wall_platform_space[wall_ind0][1] = hypotenuse_corner[0] * math.cos(angle_corner[0]) + platform_width
                    wall_platform_space[wall_ind1][0] = hypotenuse_corner[0] * math.sin(angle_corner[0]) + platform_width
            else:
                wall_platform_space[wall_ind0] = [random.randint(hooks["min_free_platform_start"], hooks["max_free_platform_start"]) if random.random() > hooks["zero_start_probability"] else 0, random.randint(hooks["min_free_platform_start"], hooks["max_free_platform_start"]) if random.random() > hooks["zero_start_probability"] else 0]
                wall_platform_space[wall_ind1] = [random.randint(hooks["min_free_platform_start"], hooks["max_free_platform_start"]) if random.random() > hooks["zero_start_probability"] else 0, random.randint(hooks["min_free_platform_start"], hooks["max_free_platform_start"]) if random.random() > hooks["zero_start_probability"] else 0]
            
            for wall_no in walls:
                if wall_platform_space[wall_no][0] == -1:
                    wall_platform_space[wall_no][0] = random.randint(hooks["min_free_platform_start"], hooks["max_free_platform_start"]) if random.random() > hooks["zero_start_probability"] else 0
                if wall_platform_space[wall_no][1] == -1:
                    wall_platform_space[wall_no][1] = random.randint(hooks["min_free_platform_start"], hooks["max_free_platform_start"]) if random.random() > hooks["zero_start_probability"] else 0

        else: #no_of_walls == 1
            wall_platform_space[walls[0]] = [random.randint(hooks["min_free_platform_start"], hooks["max_free_platform_start"]) if random.random() > hooks["zero_start_probability"] else 0, random.randint(hooks["min_free_platform_start"], hooks["max_free_platform_start"]) if random.random() > hooks["zero_start_probability"] else 0]

        state_platforms = []
        idx = 0
        for wall_no in walls:
            is_two = False #True if random.random() > hooks["platform_seperation_probability"] else False

            platform_space = wall_platform_space[wall_no]

            vertical = wall_no % 2 == 0

            if is_two:
                # available_length = room_width if wall_no in [0, 2] else room_length - platform_space[0] - platform_space[1]
                # length_1 = 
                # length_2 = 
                # position = platform_space[0] if vertical else platform_space[1]

                # if vertical:  # Left or Right
                #     x_coord = 0 if wall_no == 0 else room_length - platform_width
                #     y_coord = position
                # else: # Top or Bottom
                #     x_coord = position
                #     y_coord = 0 if wall_no == 3 else room_width - platform_width
                
                # state_platforms.append(torch.tensor([idx, wall_no, 0, length, x_coord, y_coord, vertical], dtype=torch.float32))
                # idx += 2
                pass
            else:
                #print(platform_space)
                length = room_width - platform_space[0] - platform_space[1] if wall_no in [0, 2] else room_length - platform_space[0] - platform_space[1]
                position = platform_space

                if vertical:  # Left or Right
                    x_coord = 0 if wall_no == 0 else room_length - platform_width
                    y_coord = position[0] if wall_no == 0 else position[1]
                else: # Top or Bottom
                    x_coord = position[0] if wall_no == 1 else position[1]
                    y_coord = 0 if wall_no == 3 else room_width - platform_width
                
                state_platforms.append(torch.tensor([idx, wall_no, 0, length, x_coord, y_coord, vertical], dtype=torch.float32))
                idx += 1
        
        #Object Generation
        state_objects = []
        
        lst_objs = list(self.object_specs.keys())
        
        for i in range(len(state_platforms)):
            num_fails = 2
            
            length_plat = state_platforms[i][3].item()
            
            while length_plat > 0:
                selected_obj = random.choice(lst_objs)
                
                if self.object_specs[selected_obj]["min_length"] < length_plat:
                    state_objects.append(self.object_emb({
                        "object_index": torch.tensor([lst_objs.index(selected_obj)]).unsqueeze(0),
                        "min_size": torch.tensor([self.object_specs[selected_obj]["min_length"]]).unsqueeze(0),
                        "max_size": torch.tensor([self.object_specs[selected_obj]["max_length"]]).unsqueeze(0),
                        "priority": torch.tensor([self.object_specs[selected_obj]["priority"]]).unsqueeze(0),
                        "repeat": torch.tensor([self.object_specs[selected_obj]["max_count"] > 1]).unsqueeze(0),
                        "min_count": torch.tensor([1]).unsqueeze(0),
                        "max_count": torch.tensor([self.object_specs[selected_obj]["max_count"]]).unsqueeze(0)
                    }))
                    
                    length_plat -= self.object_specs[selected_obj]["min_length"]
                else:
                    num_fails -= 1
                
                if num_fails <= 0:
                    break
                
        return TensorDict({
            "state_platform": torch.stack(state_platforms),
            "state_object": torch.stack(state_objects).squeeze(-1),
            "room_dimensions": torch.tensor([room_length, room_width])
        })
        
    def generate_state(self, objects, max_attempts=100, hooks : dict = {}):
        """
        Generates random intial Puzzle configurations
        """
        # platform_width = 620  # Fixed platform width in mm
        # min_platform_length = 810  # Minimum platform length in mm
        # max_platform_length = 1410 # Maximum platform lenght in mm
        # side_gap = 620  # Minimum distance from the side walls in mm
        
        # room_length = 4000#random.randint(4000, 4500)  # Random room length between 2000mm and 8000mm
        # room_width = 3000#random.randint(3000, 4000)  # Random room width between 800mm and 4000mm

        # total_platforms = random.randint(2, 4)  # Ensure 2 to 5 platforms are planned
        # walls = [0, 1, 2, 3]  # Possible walls
        # random.shuffle(walls)
        # used_walls = walls  # Use only three walls
        
        # wall_index = {0:0, 1:0, 2:0, 3:0}

        # state_platforms = []
        # platform_rects = []
        
        # tot_plat_len = 0
        # for plt_idx in range(total_platforms):
        #     attempts = 0
        #     while attempts < max_attempts:
        #         edge_choice = random.choice(used_walls)
        #         available_length = room_width if edge_choice in [2, 3] else room_length
        #         if available_length <= 2 * side_gap + min_platform_length:
        #             continue  # Try next iteration if not enough space

        #         length = random.randint(min_platform_length, available_length - 2 * side_gap)
        #         position = random.randint(side_gap, available_length - length - side_gap)

        #         if edge_choice in [0, 1]:  # Top or Bottom
        #             x_coord = position
        #             y_coord = 0 if edge_choice == 0 else room_width - platform_width
        #         else:  # Left or Right
        #             x_coord = 0 if edge_choice == 2 else room_length - platform_width
        #             y_coord = position

        #         rect = (x_coord, y_coord, x_coord + length, y_coord + platform_width)

        #         if not self.check_overlap(rect, platform_rects, side_gap):
        #             platform_rects.append(rect)
        #             index = len(state_platforms)
        #             wall_number = edge_choice
        #             wall_idx = wall_index[edge_choice]
        #             vertical = edge_choice >= 2
        #             platform_properties = torch.tensor([index, wall_number, wall_idx, length, x_coord, y_coord, vertical], dtype=torch.float32)
        #             state_platforms.append(platform_properties)

        #             tot_plat_len += length
        #             break
        #         attempts += 1

        # attempts = 0

        # while attempts < max_attempts:
        #     obj_no = random.randint(2, 7)
        #     objs = []

        #     tot_obj_len = 0

        #     for i in range(obj_no):
        #         idx = random.randint(0, len(list(self.object_specs.keys())) - 1)
        #         key = list(self.object_specs.keys())[idx]
                
        #         tot_obj_len += self.object_specs[key]["min_length"]

        #         objs.append(self.object_emb({
        #             "object_index": torch.tensor([idx]).unsqueeze(0),
        #             "min_size": torch.tensor([self.object_specs[key]["min_length"]]).unsqueeze(0),
        #             "max_size": torch.tensor([self.object_specs[key]["max_length"]]).unsqueeze(0),
        #             "priority": torch.tensor([self.object_specs[key]["priority"]]).unsqueeze(0),
        #             "repeat": torch.tensor([self.object_specs[key]["max_count"] > 1]).unsqueeze(0),
        #             "min_count": torch.tensor([1]).unsqueeze(0),
        #             "max_count": torch.tensor([self.object_specs[key]["max_count"]]).unsqueeze(0)
        #         }))
                
        #     if tot_obj_len <= tot_plat_len:
        #         break
            
        #     attempts += 1

        # return TensorDict({
        #     "state_platform": torch.stack(state_platforms) if state_platforms else torch.tensor([]),
        #     "state_object": torch.stack(objs).squeeze(-1),
        #     "room_dimensions": torch.tensor([room_length, room_width])
        # })
        
        #room dimensions
        room_width = random.randint(hooks["min_room_width"], hooks["max_room_width"])
        room_length = random.randint(hooks["min_room_length"], hooks["max_room_length"])

        #platform
        platform_width = PLATFORM_WIDTH

        #random walls
        walls = [0, 1, 2, 3]
        random.shuffle(walls)
        no_of_walls = random.randint(1, 4)
        walls = walls[:no_of_walls]
        
        if no_of_walls == 3:
            if 3 in walls:
                if 1 in walls:
                    if 0 in walls:
                        walls = [3, 0, 1]
                    else:
                        walls = [1, 2, 3]
                else:
                    walls = [2, 3, 0]
            else:
                walls.sort()
        elif no_of_walls == 2:
            if 3 in walls:
                if 1 in walls:
                    walls = [1, 3]
                elif 2 in walls:
                    walls = [2, 3]
                else:
                    walls = [3, 0]
            else:
                walls.sort()
        else:
            walls.sort()

        wall_platform_space = [[-1, -1] for i in range(4)]

        #distribution
        np.random.uniform(540, 550)
        np.random.uniform(0, math.pi)

        #corners
        if no_of_walls == 4:
            n_corners = 4

            # hypotenuse_corner = np.random.uniform(540, 550, n_corners)
            # angle_corner = np.random.uniform(0, math.pi/2 + hooks["no_corner_probability"], n_corners)

            # for corner_no in range(n_corners):

            #     wall_ind0 = modulus(corner_no - 1, no_of_walls)
            #     wall_ind1 = modulus(corner_no + 1, no_of_walls)

            #     if angle_corner[corner_no] > math.pi/2:
            #         #if no corner then the maker leaves platform_width on both of the adjoining walls

            #         wall_platform_space[wall_ind0][1] = max(wall_platform_space[wall_ind0][1], platform_width)
            #         wall_platform_space[wall_ind1][0] = max(wall_platform_space[wall_ind0][0], platform_width)

            #         print()

            #     elif math.pi/2>angle_corner[corner_no]>0:
            #         # if a normal corner, we have to calculate markers using trigonometry.

            #         wall_platform_space[wall_ind0][1] = max(wall_platform_space[wall_ind0][1], hypotenuse_corner[corner_no] * math.cos(angle_corner[corner_no]) + platform_width)
            #         wall_platform_space[wall_ind1][0] = max(wall_platform_space[wall_ind0][0], hypotenuse_corner[corner_no] * math.cos(angle_corner[corner_no]) + platform_width)
                
            #     elif angle_corner[corner_no] == math.pi/2:
            #         wall_platform_space[wall_ind0][1] = 0
            #         wall_platform_space[wall_ind1][0] = hooks["straight_corner_free"]

            #     elif angle_corner[corner_no] == 0:
            #         wall_platform_space[wall_ind0][1] = hooks["straight_corner_free"]
            #         wall_platform_space[wall_ind1][0] = 0
            
            # for wall_no in walls:
            #     if wall_platform_space[wall_no][0] == -1:
            #         wall_platform_space[wall_no][0] = random.randint(0, hooks["max_free_platform_start"])
            #     if wall_platform_space[wall_no][1] == -1:
            #         wall_platform_space[wall_no][1] = random.randint(0, hooks["max_free_platform_start"])

            for corner_no in range(n_corners):
                
                wall_ind0 = modulus_arr(walls, corner_no, no_of_walls)
                wall_ind1 = modulus_arr(walls, corner_no + 1, no_of_walls)


                corner_type = random.choice([0, 1, 2, 3])
                
                if corner_type == 0:
                    wall_platform_space[wall_ind0][1] = random.randint(hooks["min_no_corner_start"], hooks["max_no_corner_start"])
                    wall_platform_space[wall_ind1][0] = random.randint(hooks["min_no_corner_start"], hooks["max_no_corner_start"])
                    
                elif corner_type == 1:
                    wall_platform_space[wall_ind0][1] = 0
                    wall_platform_space[wall_ind1][0] = hooks["straight_corner_free"] + platform_width
                    
                elif corner_type == 2:
                    wall_platform_space[wall_ind0][1] = hooks["straight_corner_free"] + platform_width
                    wall_platform_space[wall_ind1][0] = 0
                    
                else: #corner_type == 3
                    hypotenuse_corner = np.random.uniform(540, 550, 1)
                    angle_corner = np.random.uniform(math.pi/12, math.pi/2 - math.pi/12, 1)

                    wall_platform_space[wall_ind0][1] = hypotenuse_corner[0] * math.cos(angle_corner[0]) + platform_width
                    wall_platform_space[wall_ind1][0] = hypotenuse_corner[0] * math.sin(angle_corner[0]) + platform_width
            
            for wall_no in walls:
                if wall_platform_space[wall_no][0] == -1:
                    wall_platform_space[wall_no][0] = random.randint(hooks["min_free_platform_start"], hooks["max_free_platform_start"]) if random.random() > hooks["zero_start_probability"] else 0
                if wall_platform_space[wall_no][1] == -1:
                    wall_platform_space[wall_no][1] = random.randint(hooks["min_free_platform_start"], hooks["max_free_platform_start"]) if random.random() > hooks["zero_start_probability"] else 0


        elif no_of_walls == 3:
            n_corners = 2

            # hypotenuse_corner = np.random.uniform(540, 550, n_corners)
            # angle_corner = np.random.uniform(0, math.pi/2 + hooks["no_corner_probability"], n_corners)

            for corner_no in range(n_corners):
                
                wall_ind0 = modulus_arr(walls, corner_no, no_of_walls)
                wall_ind1 = modulus_arr(walls, corner_no + 1, no_of_walls)


                corner_type = random.choice([0, 1, 2, 3])
                
                if corner_type == 0:
                    wall_platform_space[wall_ind0][1] = random.randint(hooks["min_no_corner_start"], hooks["max_no_corner_start"])
                    wall_platform_space[wall_ind1][0] = random.randint(hooks["min_no_corner_start"], hooks["max_no_corner_start"])
                    
                elif corner_type == 1:
                    wall_platform_space[wall_ind0][1] = 0
                    wall_platform_space[wall_ind1][0] = hooks["straight_corner_free"] + platform_width
                    
                elif corner_type == 2:
                    wall_platform_space[wall_ind0][1] = hooks["straight_corner_free"] + platform_width
                    wall_platform_space[wall_ind1][0] = 0
                    
                else: #corner_type == 3
                    hypotenuse_corner = np.random.uniform(540, 550, 1)
                    angle_corner = np.random.uniform(math.pi/12, math.pi/2 - math.pi/12, 1)

                    wall_platform_space[wall_ind0][1] = hypotenuse_corner[0] * math.cos(angle_corner[0]) + platform_width
                    wall_platform_space[wall_ind1][0] = hypotenuse_corner[0] * math.sin(angle_corner[0]) + platform_width
            
            for wall_no in walls:
                if wall_platform_space[wall_no][0] == -1:
                    wall_platform_space[wall_no][0] = random.randint(hooks["min_free_platform_start"], hooks["max_free_platform_start"]) if random.random() > hooks["zero_start_probability"] else 0
                if wall_platform_space[wall_no][1] == -1:
                    wall_platform_space[wall_no][1] = random.randint(hooks["min_free_platform_start"], hooks["max_free_platform_start"]) if random.random() > hooks["zero_start_probability"] else 0

            
        elif no_of_walls == 2:
            wall_ind0 = walls[0]
            wall_ind1 = walls[1]

            if modulus(wall_ind1 - wall_ind0, 4) == 1:
                corner_type = random.choice([0, 1, 2, 3])
                
                if corner_type == 0:
                    wall_platform_space[wall_ind0][1] = random.randint(hooks["min_no_corner_start"], hooks["max_no_corner_start"])
                    wall_platform_space[wall_ind1][0] = random.randint(hooks["min_no_corner_start"], hooks["max_no_corner_start"])
                    
                elif corner_type == 1:
                    wall_platform_space[wall_ind0][1] = 0
                    wall_platform_space[wall_ind1][0] = hooks["straight_corner_free"] + platform_width
                    
                elif corner_type == 2:
                    wall_platform_space[wall_ind0][1] = hooks["straight_corner_free"] + platform_width
                    wall_platform_space[wall_ind1][0] = 0
                    
                else: #corner_type == 3
                    hypotenuse_corner = np.random.uniform(540, 550, 1)
                    angle_corner = np.random.uniform(math.pi/12, math.pi/2 - math.pi/12, 1)

                    wall_platform_space[wall_ind0][1] = hypotenuse_corner[0] * math.cos(angle_corner[0]) + platform_width
                    wall_platform_space[wall_ind1][0] = hypotenuse_corner[0] * math.sin(angle_corner[0]) + platform_width
            else:
                wall_platform_space[wall_ind0] = [random.randint(hooks["min_free_platform_start"], hooks["max_free_platform_start"]) if random.random() > hooks["zero_start_probability"] else 0, random.randint(hooks["min_free_platform_start"], hooks["max_free_platform_start"]) if random.random() > hooks["zero_start_probability"] else 0]
                wall_platform_space[wall_ind1] = [random.randint(hooks["min_free_platform_start"], hooks["max_free_platform_start"]) if random.random() > hooks["zero_start_probability"] else 0, random.randint(hooks["min_free_platform_start"], hooks["max_free_platform_start"]) if random.random() > hooks["zero_start_probability"] else 0]
            
            for wall_no in walls:
                if wall_platform_space[wall_no][0] == -1:
                    wall_platform_space[wall_no][0] = random.randint(hooks["min_free_platform_start"], hooks["max_free_platform_start"]) if random.random() > hooks["zero_start_probability"] else 0
                if wall_platform_space[wall_no][1] == -1:
                    wall_platform_space[wall_no][1] = random.randint(hooks["min_free_platform_start"], hooks["max_free_platform_start"]) if random.random() > hooks["zero_start_probability"] else 0

        else: #no_of_walls == 1
            wall_platform_space[walls[0]] = [random.randint(hooks["min_free_platform_start"], hooks["max_free_platform_start"]) if random.random() > hooks["zero_start_probability"] else 0, random.randint(hooks["min_free_platform_start"], hooks["max_free_platform_start"]) if random.random() > hooks["zero_start_probability"] else 0]

        state_platforms = []
        idx = 0
        for wall_no in walls:
            is_two = False #True if random.random() > hooks["platform_seperation_probability"] else False

            platform_space = wall_platform_space[wall_no]

            vertical = wall_no % 2 == 0

            if is_two:
                # available_length = room_width if wall_no in [0, 2] else room_length - platform_space[0] - platform_space[1]
                # length_1 = 
                # length_2 = 
                # position = platform_space[0] if vertical else platform_space[1]

                # if vertical:  # Left or Right
                #     x_coord = 0 if wall_no == 0 else room_length - platform_width
                #     y_coord = position
                # else: # Top or Bottom
                #     x_coord = position
                #     y_coord = 0 if wall_no == 3 else room_width - platform_width
                
                # state_platforms.append(torch.tensor([idx, wall_no, 0, length, x_coord, y_coord, vertical], dtype=torch.float32))
                # idx += 2
                pass
            else:
                #print(platform_space)
                length = room_width - platform_space[0] - platform_space[1] if wall_no in [0, 2] else room_length - platform_space[0] - platform_space[1]
                position = platform_space

                if vertical:  # Left or Right
                    x_coord = 0 if wall_no == 0 else room_length - platform_width
                    y_coord = position[0] if wall_no == 0 else position[1]
                else: # Top or Bottom
                    x_coord = position[0] if wall_no == 1 else position[1]
                    y_coord = 0 if wall_no == 3 else room_width - platform_width
                
                state_platforms.append(torch.tensor([idx, wall_no, 0, length, x_coord, y_coord, vertical], dtype=torch.float32))
                idx += 1
        
        #Object Generation
        lst_objs = list(self.object_specs.keys())
        
        state_objects = []
        
        for i, j in objects:
            state_objects.append(self.object_emb({
                "object_index": torch.tensor([lst_objs.index(i)]).unsqueeze(0),
                "min_size": torch.tensor([j]).unsqueeze(0),
                "max_size": torch.tensor([self.object_specs[i]["max_length"]]).unsqueeze(0),
                "priority": torch.tensor([self.object_specs[i]["priority"]]).unsqueeze(0),
                "repeat": torch.tensor([self.object_specs[i]["max_count"] > 1]).unsqueeze(0),
                "min_count": torch.tensor([1]).unsqueeze(0),
                "max_count": torch.tensor([self.object_specs[i]["max_count"]]).unsqueeze(0)
            }))
        
        #
        
        # for i in range(len(state_platforms)):
        #     num_fails = 2
            
        #     length_plat = state_platforms[i][3].item()
            
        #     while length_plat > 0:
        #         selected_obj = random.choice(lst_objs)
                
        #         if self.object_specs[selected_obj]["min_length"] < length_plat:
        #             state_objects.append(self.object_emb({
        #                 "object_index": torch.tensor([lst_objs.index(selected_obj)]).unsqueeze(0),
        #                 "min_size": torch.tensor([self.object_specs[selected_obj]["min_length"]]).unsqueeze(0),
        #                 "max_size": torch.tensor([self.object_specs[selected_obj]["max_length"]]).unsqueeze(0),
        #                 "priority": torch.tensor([self.object_specs[selected_obj]["priority"]]).unsqueeze(0),
        #                 "repeat": torch.tensor([self.object_specs[selected_obj]["max_count"] > 1]).unsqueeze(0),
        #                 "min_count": torch.tensor([1]).unsqueeze(0),
        #                 "max_count": torch.tensor([self.object_specs[selected_obj]["max_count"]]).unsqueeze(0)
        #             }))
                    
        #             length_plat -= self.object_specs[selected_obj]["min_length"]
        #         else:
        #             num_fails -= 1
                
        #         if num_fails <= 0:
        #             break
                
        return TensorDict({
            "state_platform": torch.stack(state_platforms),
            "state_object": torch.stack(state_objects).squeeze(-1),
            "room_dimensions": torch.tensor([room_length, room_width])
        })

