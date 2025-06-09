import time
import gym
import numpy as np
from torch import load
from tools import getRotationMatrix
import transforms3d
from .Interface import Interface
from .IRcreator import RandomItemCreator, LoadItemCreator, RandomInstanceCreator, RandomCateCreator
from .space import Space
from .cvTools import getConvexHullActions
import random
import threading
import math 



def getConvexHullActions(*args, **kwargs):
    # rot, x, y, h, v
    return np.array([[0, 5, 5, 0.1, 1.0]]) # Return one dummy action candidate

def getRotationMatrix(*args, **kwargs):
    return [np.eye(3)], [np.eye(3)]
     
import transforms3d.quaternions


def non_blocking_simulation(interface, finished, id, non_blocking_result):
    # Simulate work
    # time.sleep(0.01)
    succeeded, valid = interface.simulateToQuasistatic(givenId=id,
                                    linearTol=0.01,
                                    angularTol=0.01)

    finished[0] = True
    non_blocking_result[0] = [succeeded, valid]
    
class PackingGame(gym.Env):
    def __init__(self,
                 args
    ):
        args = vars(args) # Assumes args is a namespace, if dict, skip this
        self.resolutionAct = args.get('resolutionA', 0.01)
        self.resolutionH   = args.get('resolutionH', 0.01)
        self.bin_dimension = args.get('bin_dimension', np.array([1.0, 1.0, 1.0]))
        self.scale         = args.get('scale', 1.0)
        self.objPath       = args.get('objPath', '.')
        self.meshScale     = args.get('meshScale', 1.0)
        self.shapeDict     = args.get('shapeDict') 
        self.infoDict      = args.get('infoDict', {0: [{'volume': 0.001}], 1: [{'volume': 0.001}]})
        #self.dicPath       = load(args['dicPath'])
        self.dicPath       = load(args['dicPath'])
        self.ZRotNum       = args.get('ZRotNum', 6)
        self.heightMapPre  = args.get('heightMap', True)
        self.globalView    = not args.get('only_simulate_current', False)
        self.selectedAction= args.get('selectedAction', 10)
        self.bufferSize    = args.get('bufferSize', 1)
        self.chooseItem    = self.bufferSize > 1
        self.simulation    = args.get('simulation', True)
        self.evaluate      = args.get('evaluate', False)
        self.maxBatch      = args.get('maxBatch', 1)
        self.heightResolution   = args.get('resolutionZ', 0.01)
        self.dataSample    = args.get('dataSample', 'pose')
        self.dataname      = args.get('test_name', None)
        self.visual        = args.get('visual', False)
        self.non_blocking  = args.get('non_blocking', False)
        self.time_limit    = args.get('time_limit', 1.0)


        self.interface = None
        self.item_vec = np.zeros((1000, 9))
        self.rangeX_A, self.rangeY_A = np.ceil(self.bin_dimension[0:2] / self.resolutionAct).astype(np.int32)
        self.space = Space(self.bin_dimension, self.resolutionAct, self.resolutionH, False,   self.ZRotNum,
                           args['shotInfo'], self.scale)

        if self.evaluate and self.dataname is not None:
            self.item_creator = LoadItemCreator(data_name=self.dataname)
        else:
            if self.dataSample == 'category':
                self.item_creator = RandomCateCreator(np.arange(0, len(self.shapeDict.keys())), self.dicPath)
            elif self.dataSample == 'instance':
                self.item_creator = RandomInstanceCreator(np.arange(0, len(self.shapeDict.keys())), self.dicPath)
            else:
                assert self.dataSample == 'pose'
                self.item_creator = RandomItemCreator(np.arange(0, len(self.shapeDict.keys())))

        self.next_item_vec = np.zeros((9))

        self.item_idx = 0
        self.id = -1 # Initialize
        self.next_item_ID = 0 # Initialize
        self.candidates = getConvexHullActions() # Initialize

        self.transformation = []
        DownFaceList, ZRotList = getRotationMatrix(1, self.ZRotNum)
        for d in DownFaceList:
            for z in ZRotList:
                quat = transforms3d.quaternions.mat2quat(np.dot(z, d)[0:3, 0:3])
                self.transformation.append([quat[1],quat[2],quat[3],quat[0]]) # Saved in xyzw
        self.transformation = np.array(self.transformation)

        self.rotNum = self.ZRotNum
        self.act_len = self.selectedAction if self.selectedAction else 1 # avoid 0

        if self.chooseItem:
            self.act_len = self.bufferSize

        if not self.chooseItem:
            self.obs_len = len(self.next_item_vec.reshape(-1))

            if self.selectedAction:
                 # Ensure selectedAction is not None or 0 for multiplication
                self.obs_len += (self.selectedAction or 1) * 5
            else:
                self.obs_len += self.act_len
        else:
            self.obs_len = self.bufferSize

        if self.heightMapPre:
            self.obs_len += self.space.heightmapC.size
        
        # Handle obs_len potentially being zero if components are missing
        self.obs_len = max(1, self.obs_len) 

        self.observation_space = gym.spaces.Box(low=0.0, high=self.bin_dimension[2],
                                                shape=(self.obs_len,))
        self.action_space = gym.spaces.Discrete(self.act_len)

        self.tolerance = 0 # default 0.002

        self.episodeCounter = 0
        self.updatePeriod = 500
        self.trajs = []
        self.orderAction = 0
        self.hierachical = False

        if self.non_blocking:
            self.nullObs = np.zeros((self.obs_len))
            self.finished = [True]
            self.non_blocking_result = [None]
            self.nowTask = False
        
        self.reset() # call reset to initialize members

    def seed(self, seed=None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        return [seed]

    def close(self):
       if self.interface:
          self.interface.close()

    def reset(self, index = None):
        self.space.reset()

        self.episodeCounter = (self.episodeCounter + 1) % self.updatePeriod
        if self.episodeCounter == 0 or self.interface is None:
            if self.interface is not None:
                self.interface.close()
                del self.interface
            self.interface = Interface(bin=self.bin_dimension, foldername=self.objPath, visual=self.visual,
                                       scale=self.scale, simulationScale=self.meshScale, maxBatch=self.maxBatch,)
        else:
             self.interface.reset()
        # if self.episodeCounter == 0 or self.interface is None:
        #     if self.interface is not None:
        #         self.interface.close()
        #         del self.interface
        #     self.interface = Interface(bin=self.bin_dimension, foldername=self.objPath, visual=self.visual,
        #                                scale=self.scale, simulationScale=self.meshScale, maxBatch=self.maxBatch,)
        # else:
        #     self.interface.reset()
            
        self.item_creator.reset(index)
        self.packed = []
        self.packedId = []
        self.next_item_vec[:] = 0
        self.item_idx = 0
        self.item_vec[:] = 0
        self.id = None
        self.rotIdx_last_action = 0 # Keep track of rotation
        return self.cur_observation()

    def get_ratio(self):
        totalVolume = 0
        # Check if infoDict has the key before access
        for idx in range(self.item_idx):
            item_id_key = int(self.item_vec[idx][0])
            if item_id_key in self.infoDict and self.infoDict[item_id_key]:
                 totalVolume += self.infoDict[item_id_key][0].get('volume', 0)
        bin_volume = np.prod(self.bin_dimension)
        return totalVolume / bin_volume if bin_volume > 0 else 0

    def get_item_ratio(self, next_item_ID):
        bin_volume = np.prod(self.bin_dimension)
        item_volume = 0
        if next_item_ID in self.infoDict and self.infoDict[next_item_ID]:
             item_volume = self.infoDict[next_item_ID][0].get('volume',0)
        return  item_volume / bin_volume if bin_volume > 0 else 0

    def gen_next_item_ID(self):
        preview = self.item_creator.preview(1)
        return preview[0] if preview else 0

    # get_action_candidates, get_all_possible_observation - NO CHANGE

    def cur_observation(self, genItem = True, draw = False):
         # Define default result shape to avoid errors if conditions are not met
        result = np.zeros(self.obs_len)
        
        if self.item_idx != 0:
            positions, orientations = self.interface.getAllPositionAndOrientation(inner=False)
            if positions and orientations: # check if not empty
               self.item_vec[0:self.item_idx, 1:4] = np.array([positions[0:self.item_idx]])
               self.item_vec[0:self.item_idx, 4:8] = np.array([orientations[0:self.item_idx]])
               
        if not self.chooseItem:
            if genItem:
                self.next_item_ID = self.gen_next_item_ID()
            self.next_item_vec[0] = self.next_item_ID

            naiveMask = self.space.get_possible_position(self.next_item_ID, self.shapeDict[self.next_item_ID], self.selectedAction)

            current_result_list = [self.next_item_vec.reshape(-1)]

            if not self.selectedAction:
                 current_result_list.append(naiveMask.reshape(-1))
                # result = np.concatenate((self.next_item_vec.reshape(-1),
                #                      naiveMask.reshape(-1)))

            # Candidate generation
            if self.selectedAction:
                self.candidates = None
                self.candidates = getConvexHullActions(self.space.posZValid, self.space.naiveMask,
                                                             self.heightResolution)
                num_candidates = 0
                if self.candidates is not None:
                    num_candidates = len(self.candidates)
                    if num_candidates > self.selectedAction:
                        # sort with height
                        selectedIndex = np.argsort(self.candidates[:,3])[0: self.selectedAction]
                        self.candidates = self.candidates[selectedIndex]
                    elif num_candidates < self.selectedAction:
                        dif = self.selectedAction - len(self.candidates)
                        self.candidates = np.concatenate((self.candidates, np.zeros((dif, 5))), axis=0)
                
                # Fallback if no candidates or candidates is None
                if self.candidates is None or num_candidates == 0:
                     # Create default candidates if none found
                    self.candidates = np.zeros((self.selectedAction, 5))
                    if self.space.posZValid.size > 0: # Check if not empty
                        poszFlatten = self.space.posZValid.reshape(-1)
                        # Ensure selectedIndex length does not exceed array size
                        num_select = min(self.selectedAction, len(poszFlatten))
                        selectedIndex = np.argsort(poszFlatten)[0: num_select]
                        if selectedIndex.size > 0 and self.rotNum > 0 and self.rangeX_A > 0 and self.rangeY_A > 0:
                           ROT,X,Y = np.unravel_index(selectedIndex, (self.rotNum, self.rangeX_A, self.rangeY_A))
                           H = poszFlatten[selectedIndex]
                           V_flat = self.space.naiveMask.reshape(-1)
                           V = V_flat[selectedIndex] if len(V_flat) > max(selectedIndex) else np.zeros_like(H)
                           H[:] = self.bin_dimension[-1] # Default height
                           # Assign to the allocated zero array
                           self.candidates[0:num_select, :] = np.concatenate([ROT.reshape(-1, 1), X.reshape(-1, 1),
                                                          Y.reshape(-1, 1), H.reshape(-1, 1), V.reshape(-1, 1)], axis=1)

                current_result_list.insert(0, self.candidates.reshape(-1)) # Prepend candidates
                #result = np.concatenate((self.candidates.reshape(-1), result))
            
            if self.heightMapPre:
                current_result_list.append(self.space.heightmapC.reshape(-1))
             
            # Concatenate all parts safely
            try:
                result = np.concatenate(current_result_list)
                 # Ensure final result matches obs_len if padding/truncation is needed
                if len(result) != self.obs_len:
                   # print(f"Warning: Observation length mismatch. Expected {self.obs_len}, Got {len(result)}. Padding/Truncating.")
                   temp_res = np.zeros(self.obs_len)
                   copy_len = min(len(result), self.obs_len)
                   temp_res[:copy_len] = result[:copy_len]
                   result = temp_res
            except ValueError as e:
                 print(f"Error concatenating observation parts: {e}")
                 result = np.zeros(self.obs_len) # Fallback to zeros

        else: # self.chooseItem is True
             self.next_k_item_ID = self.item_creator.preview(self.bufferSize)
             # Ensure elements are numeric arrays before concatenation
             items_array = np.array(self.next_k_item_ID).reshape(-1)
             heightmap_array = self.space.heightmapC.reshape(-1)
             try:
                 result_list = [items_array]
                 if self.heightMapPre: # Add heightmap only if required
                    result_list.append(heightmap_array)
                 result = np.concatenate(result_list)
                 # Match obs_len
                 if len(result) != self.obs_len:
                    temp_res = np.zeros(self.obs_len)
                    copy_len = min(len(result), self.obs_len)
                    temp_res[:copy_len] = result[:copy_len]
                    result = temp_res

             except ValueError as e:
                 print(f"Error concatenating observation (chooseItem): {e}")
                 result = np.zeros(self.obs_len)
            #result = np.concatenate((np.array(self.next_k_item_ID), self.space.heightmapC.reshape(-1)))

        return result


    def action_to_position(self, action):
        # Add boundary check for action index
        if self.candidates is None or action >= len(self.candidates) or action < 0:
             # Return default/error values
            return 0, np.round((0, 0, self.bin_dimension[2]), decimals=6), (0,0)
            
        rotIdx, lx, ly = self.candidates[action][0:3].astype(int)
         # Ensure rotIdx is within valid bounds if needed elsewhere
        rotIdx = max(0, min(rotIdx, self.rotNum -1 ))
        lx = max(0, lx)
        ly = max(0, ly)
        self.rotIdx_last_action = rotIdx # STORE THE ROTATION INDEX
        return rotIdx, np.round((lx * self.resolutionAct, ly * self.resolutionAct, self.bin_dimension[2]), decimals=6), (lx,ly)

    def prejudge(self, rotIdx, translation, naiveMask):
         # Ensure shapeDict has the key and item at rotIdx before access
        if self.next_item_ID not in self.shapeDict or rotIdx >= len(self.shapeDict[self.next_item_ID]):
             # print(f"Warning: shapeDict missing ID {self.next_item_ID} or index {rotIdx}")
             return False # Cannot place if shape info is missing
             
        extents = self.shapeDict[self.next_item_ID][rotIdx].extents
        if np.round(translation[0] + extents[0] - self.bin_dimension[0], decimals=6)  > 0 \
            or np.round(translation[1] + extents[1] - self.bin_dimension[1], decimals=6) > 0:
            return False
        if naiveMask is None or np.sum(naiveMask) == 0: # Check if naiveMask is valid
            return False
        return True

    # Note the transform between Ra coord and Rh coord
    def step(self, action):
        if self.non_blocking and not self.finished[0]:
             # Use self.nullObs which is initialized in __init__
            return self.nullObs, 0.0, False, {'Valid': False} 

        success, sim_suc = False, False # Initialize
        if self.non_blocking and self.finished[0] and self.nowTask:
            success, sim_suc = self.non_blocking_result[0]
            self.nowTask = False
            self.non_blocking_result[0] = None
             # Need to define rotIdx, targetFLB, coordinate, rotation, height even in this path if used later
             # This part of the logic seems incomplete in original code for variable scope.
             # Assuming action leads to stored state or re-evaluation needed. 
             # For now, just use the results. Need self.id to be correct.
            rotIdx, targetFLB, coordinate = 0, [0,0,0], (0,0) # Placeholders
            rotation = [0,0,0,1]
            height = 0

        else:
            rotIdx, targetFLB, coordinate = self.action_to_position(action)
            # Ensure rotIdx is valid for transformation access
            rotIdx_safe = max(0, min(int(rotIdx), len(self.transformation)-1))
            rotation = self.transformation[rotIdx_safe]

            sim_suc = False
            success = self.prejudge(rotIdx, targetFLB, self.space.naiveMask)
            self.id = self.interface.addObject(self.dicPath[self.next_item_ID][0:-4], targetFLB = targetFLB, rotation = rotation,
                                          linearDamping = 0.5, angularDamping = 0.5)
            
            # Ensure index is valid for posZmap
            height = 0
            pz_shape = self.space.posZmap.shape
            if 0 <= rotIdx < pz_shape[0] and 0 <= coordinate[0] < pz_shape[1] and 0 <= coordinate[1] < pz_shape[2]:
               height = self.space.posZmap[rotIdx, coordinate[0], coordinate[1]]
               
            self.interface.adjustHeight(self.id , height + self.tolerance)

            if success:
                if self.simulation:
                    if self.non_blocking:
                          self.finished[0] = False
                          subProcess = threading.Thread(target=non_blocking_simulation, args=(self.interface, self.finished, self.id, self.non_blocking_result))
                          subProcess.start()
                          self.nowTask = True
                          start_time = time.time()
                          end_time   = start_time
                          while end_time - start_time < self.time_limit:
                                end_time = time.time()
                                if self.finished[0]:
                                     success, sim_suc = self.non_blocking_result[0] # update success/sim_suc
                                     break
                          if not self.finished[0]:
                               return self.nullObs, 0.0, False, {'Valid': False} # Timeout
                    else:
                        success, sim_suc = self.interface.simulateToQuasistatic(givenId=self.id,
                                                                            linearTol = 0.01,
                                                                            angularTol = 0.01)
                else:
                    success, sim_suc = self.interface.simulateHeight(self.id)

        if not self.globalView and self.id is not None:
            self.interface.disableObject(self.id)

        bounds = None
        positionT, orientationT = [0,0,0], [0,0,0,1] # Defaults
        if self.id is not None:
           bounds = self.interface.get_wraped_AABB(self.id, inner=False)
           positionT, orientationT = self.interface.get_Wraped_Position_And_Orientation(self.id, inner=False)
           # Check if item path exists
           self.packed.append([self.next_item_ID, self.dicPath[self.next_item_ID], positionT, orientationT])
           self.packedId.append(self.id)

        if not success:
            if self.globalView and self.evaluate:
                 for replayIdx, idNow in enumerate(self.packedId):
                    if idNow is not None:
                       pT, oT = self.interface.get_Wraped_Position_And_Orientation(idNow, inner=False)
                       if replayIdx < len(self.packed):
                           self.packed[replayIdx][2] = pT
                           self.packed[replayIdx][3] = oT
            reward = 0.0
            info = {'counter': self.item_idx,
                    'ratio': self.get_ratio(),
                    'Valid': True, # Original code says True here, maybe should be False on !success?
                    }
            # Ensure packed lists are consistent on failure
            if self.packed: self.packed.pop()
            if self.packedId: self.packedId.pop()
            if self.id is not None and self.id in self.interface.objs:
               self.interface.objs.remove(self.id)
               self.interface.removeBody(self.id)

            observation = self.cur_observation()
            return observation, reward, True, info # Note: Done=True on failure

        if sim_suc:
            shape_info = self.shapeDict.get(self.next_item_ID) # Get shape info
            if self.globalView:
                self.space.shot_whole()
            elif shape_info: # Check shape_info is not None
                self.space.place_item_trimesh(shape_info[0], (positionT, orientationT), (bounds, self.next_item_ID))

            self.item_vec[self.item_idx, 0] = self.next_item_ID
            self.item_vec[self.item_idx, -1] = 1
            
            # R_place = Volume Ratio (β not defined in paper, using 10 from code)
            item_ratio = self.get_item_ratio(self.next_item_ID)
            beta = 10.0 
            volume_reward = beta * item_ratio # β * R_place

            # ΔCt = Compactness Score (k/n), with α = 0.3 
            # "直径为待放置物体的投影面积的对角线长的1.2倍" -> Radius = 0.5 * 1.2 * diagonal
            # Use the extent of the rotation that was actually chosen
            compactness_score = 0.0
            # Ensure shape_info and index exists
            if shape_info and 0 <= self.rotIdx_last_action < len(shape_info):
                 extent = shape_info[self.rotIdx_last_action].extents[:2]
                 diagonal = np.linalg.norm(extent) 
                 radius = 0.6 * diagonal # (1.2 * diagonal) / 2
                 # Call corrected compute_compactness, passing self.id and bin_dims
                 compactness_score = self.compute_compactness(positionT, radius, self.id, self.bin_dimension)
            
            alpha = 0.3 # 权重系数 α : 通过实验设为 0.3
            compactness_reward = alpha * compactness_score # α * ΔCt (where ΔCt = k/n)

            # Total Reward: Rt = β·Rplace + α· ΔCt
            reward = volume_reward + compactness_reward
            
            self.item_idx += 1
            self.item_creator.update_item_queue(self.orderAction)
            self.item_creator.generate_item()  # add a new box to the list
            observation = self.cur_observation()
            return observation, reward, False, {'Valid': True}
        
        else: # sim_suc is False (Invalid call)
            if self.packed: self.packed.pop()
            if self.packedId: self.packedId.pop()
            # Ensure id exists before trying to remove
            if self.id is not None:
                # Check if self.id is actually in the list before removing
                if self.id in self.interface.objs:
                   self.interface.objs.remove(self.id) # remove only if present
                self.interface.removeBody(self.id) # interface should handle invalid ID
            self.item_creator.update_item_queue(self.orderAction)
            self.item_creator.generate_item()  # Add a new box to the list
            observation = self.cur_observation()
            return observation, 0.0, False, {'Valid': False}

    def compute_compactness(self, position, radius, self_id, bin_dims, n=24, z_offset=0.01):
        """
         Calculates compactness score k/n based on PDF.
         k = number of occluded rays
         n = total number of rays (e.g., 24 for 15 degree interval)
         Occlusion: hit another object OR the bin wall within radius R.
         Radius R: 0.5 * 1.2 * diagonal length of object base.
         Center O: object base projection center.
         Self-occlusion must be ignored.
        """
        if n <= 0 or radius <= 1e-6:
            return 0.0
            
        # from math import cos, sin, pi # Use numpy
        occlusion_count = 0
        # 从该点发出 n 条等角射线 (如每 15°划分一次) -> n=360/15=24
        angles = np.linspace(0, 2 * math.pi, n, endpoint=False) 
        
        # 设物体底面投影中心为O
        # Use a small z-offset to cast rays slightly above the ground/base
        center = np.array([position[0], position[1], z_offset]) 
        
        boundary_tolerance = 1e-4 # Avoid float issues exactly on boundary

        for theta in angles:
            is_occluded = False
            dir_vec = np.array([math.cos(theta), math.sin(theta), 0])
            target = center + dir_vec * radius # Endpoint for ray test at distance R
            
            # Check occlusion by other objects using rayTest
            # interface.rayTest assumed to return tuple (hitObjectID, ...) or None
            # Pybullet returns objectUniqueID = -1 if no hit.
            ray_result = self.interface.rayTest(center.tolist(), target.tolist())
           
            # Check if ray hit an object AND that object is NOT the item itself
            if ray_result is not None:
                 hit_id = ray_result[0] # Assume index 0 is the object ID
                 if hit_id != -1 and hit_id != self_id:
                      # 若某射线方向在R内与其他物体...发生碰撞，视为“被遮挡”
                       is_occluded = True
            
            # Check occlusion by bin wall if not already occluded by an object
            # If the target point at distance R is outside the bin, the ray crossed the boundary.
            if not is_occluded:
                 # ...或边界发生碰撞，视为“被遮挡”
                 if (target[0] < -boundary_tolerance or 
                     target[0] > bin_dims[0] + boundary_tolerance or
                     target[1] < -boundary_tolerance or
                     target[1] > bin_dims[1] + boundary_tolerance):
                     is_occluded = True

            if is_occluded:
                 occlusion_count += 1 # This is 'k'
        
        # ΔCt = (360-theta_free)/360 = k/n; score Ct = 360 * k/n
        # The reward function uses k/n directly.
        compactness_score_kn = occlusion_count / n 
        return compactness_score_kn
