import sys
import os
import torch
import numpy as np
from tqdm import tqdm
import random
import math
import h5py
import MinkowskiEngine as ME
from abc import abstractmethod
from torch.utils.data import Dataset
import open3d as o3d
import pickle
np.set_printoptions(threshold=sys.maxsize)
class GeneralMultiImputedDataset(Dataset):
    def __init__(self, data_cfg=None, split="train"):
        self.data_cfg = data_cfg
        self.split = split
        human_path = data_cfg.human_dir #TODO

        self.human_data = {}

        reflect_trans = np.eye(4)
        reflect_trans[0,0] = -1
        reflect_trans3 = np.eye(3)
        reflect_trans3[0,0] = -1

        base_num = 11470
        for human_type in tqdm(os.listdir(human_path)):
            mesh_files_dir = f"{human_path}/{human_type}/meshes"
            joints_files_dir = f"{human_path}/{human_type}/joints"

            human_type_dict = {}
            for filename in os.listdir(mesh_files_dir):
                human_pose_name = filename.split(".")[0]

                human_mesh_left = o3d.io.read_triangle_mesh(f"{mesh_files_dir}/{filename}")
                human_mesh_right = o3d.geometry.TriangleMesh(human_mesh_left)
                human_mesh_right.transform(reflect_trans)

                with open(os.path.join(f"{joints_files_dir}/{human_pose_name}.pkl"), 'rb') as f:
                    human_joints_left = pickle.load(f)

                human_joints_right = {}

                for key in human_joints_left:
                    if "right" in key:
                        human_joints_right[key.replace("right", "left")] = reflect_trans3@human_joints_left[key]
                    elif "left" in key:
                        human_joints_right[key.replace("left","right")] = reflect_trans3@human_joints_left[key]
                    else:
                        human_joints_right[key] = reflect_trans3@human_joints_left[key]
                human_type_dict[human_pose_name] = {
                                                        "left_mesh": human_mesh_left,
                                                        "joints_left": human_joints_left,
                                                        "right_mesh": human_mesh_right,
                                                        "joints_right": human_joints_right,
                                                    }
                #if base_num!=len(np.asarray(human_mesh_left.vertices)):
                #    print(len(np.asarray(human_mesh_left.vertices)))
            self.human_data[human_type] = human_type_dict


        # load language data from disk to memory
        self._load_language_data()

        # load scene data from disk to memory
        self._load_scene_data()

        # pack scene and language data
        self._pack_data_to_chunks()



    def _open_hdf5(self):
        self.multiview_data = h5py.File(self.data_cfg.scene_metadata.scene_multiview_file, "r", libver="latest")

    def _load_scene_data(self):
        scene_data_path = self.data_cfg.scene_dataset_path
        self.all_scene_data = {}
        for scene_id in tqdm(self.scene_ids, desc=f"Loading {self.split} data from disk"):

            scene_path = os.path.join(scene_data_path, self.split+"_imputed", f"{scene_id}.pth")
            try:
                scene_data = torch.load(scene_path)
            except Exception as e:
                print(f"Errored in scene: {scene_id} \n{e}")
            scene_data["rgb"] = scene_data["rgb"].astype(np.float32) / 127.5 - 1  # scale rgb to [-1, 1]
            self.all_scene_data[scene_id] = scene_data

    @abstractmethod
    def _load_language_data(self):
        # this function is overridden by child class
        pass

    def _pack_data_to_chunks(self):
        # this array maintains lists of pointers pointing to language and scene data
        self.chunk_lang_indices = np.empty(shape=0, dtype=np.uint16)
        self.chunk_scene_indices = np.empty(shape=0, dtype=np.uint16)
        self.chunk_transform_indices = np.empty(shape=0, dtype=np.uint8)

        for i,scene_id in enumerate(self.scene_ids):
            big_chunk_lang_indices=np.arange(len(self.language_data[scene_id]), dtype=np.uint16)
            big_chunk_scene_indices = np.full(shape=len(self.language_data[scene_id]), fill_value=i, dtype=np.uint16)
            np.random.shuffle(big_chunk_lang_indices)

            self.chunk_lang_indices = np.concatenate([self.chunk_lang_indices,big_chunk_lang_indices], axis = 0)
            self.chunk_scene_indices = np.concatenate([self.chunk_scene_indices, big_chunk_scene_indices], axis = 0)



    def _get_xyz_augment_matrix(self):
        aug_settings = self.data_cfg.scene_augmentation
        m = np.eye(3, dtype=np.float32)
        #return m
        if self.split == "train" and aug_settings.jitter_xyz:
            m += np.random.randn(3, 3) * 0.1
        if self.split == "train" and aug_settings.flip_x and random.random() > 0.5:
            m[0][0] *= -1
        if self.split == "train" and aug_settings.rotate_z:
            rad = random.choice((0, 0.5, 1, 1.5)) * np.pi  # randomly rotate around z-axis by 0, 90, 180, 270 degrees
            c = np.cos(rad)
            s = np.sin(rad)
            m = m @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return m.astype(np.float32)

    @abstractmethod
    def _augment_language(self):
        # this function is overridden by child class
        pass

    def shuffle_chunks(self):
        # called after each epoch
        self._pack_data_to_chunks()

    def __len__(self):
        return len(self.chunk_lang_indices)

    def __getitem__(self, index):
        data_dict = {}

        try:
            scene_id = self.scene_ids[self.chunk_scene_indices[index]]
        except:
            print("Error: ",self.chunk_scene_indices[index],",", self.scene_ids[self.chunk_scene_indices[index]])
        #print(scene_id)
        object_id = self.language_data[scene_id][self.chunk_lang_indices[index]]["object_id"]

        scene_data = self.all_scene_data[scene_id].copy()

        human_data = scene_data['human_info']
        lr = ['left', 'right'][random.randint(0,1)]

        all_humans_data = human_data[object_id]

        if len(all_humans_data['left'])==0 and len(all_humans_data['right'])==0:
            raise RuntimeError(f"ERROR Human: {scene_id}: {object_id}")
            lr = None
        elif len(all_humans_data['left'])>0 and len(all_humans_data['right'])==0:
            #print("left_compulsory")
            lr = 'left'
        elif len(all_humans_data['right'])>0 and len(all_humans_data['left'])==0:
            #print("right_compulsory")
            lr = 'right'

        if not lr is None:
            multi_human_lr = all_humans_data[lr]
            human_id = random.randint(0,len(multi_human_lr)-1)
            human_dict = multi_human_lr[human_id]
            human_transforms = human_dict['axis_aligned_transforms']
            arm_angles = human_dict['angle_bin']
            transform_id = random.randint(0,len(human_transforms)-1)
            human_transform = human_transforms[transform_id]
            arm_angle = arm_angles[transform_id]
            human_name = human_dict['human_name']
            instance_id = human_dict['instance_id']
            sem_label = human_dict['sem_label']

            human_joints = self.human_data[human_name][f"l_h_{int(arm_angle.item())}"][f"joints_{lr}"]

            human_center_translate = torch.eye(4, dtype=torch.float64)
            human_center_translate[:3,3] = -torch.tensor(human_joints['left_shoulder'])
            #print('left_shoulder: ', human_joints['left_shoulder'])
            human_perturbation_range = 0.5*self.data_cfg.human_perturbation_percentage*2*torch.pi/100
            human_perturbation = (torch.rand(1)-0.5)*human_perturbation_range
            perturb_transform = torch.eye(4, dtype=torch.float64)
            perturb_transform[0,0] = torch.cos(human_perturbation)
            perturb_transform[1,1] = torch.cos(human_perturbation)
            perturb_transform[0,1] = torch.sin(human_perturbation)
            perturb_transform[1,0] = -torch.sin(human_perturbation)
            perturb_transform = (human_center_translate.inverse()@perturb_transform@human_center_translate).numpy()
            human_transform = human_transform@perturb_transform

            human_mesh = o3d.geometry.TriangleMesh(self.human_data[human_name][f"l_h_{int(arm_angle.item())}"][f"{lr}_mesh"])
            human_mesh.transform(human_transform)
            human_joints_final = {}
            for name, coord in human_joints.items():
                joint_coord_homo = np.array([0.,0,0,1], np.float64)
                joint_coord_homo[:3] = coord
                joint_coord_transformed_homo = human_transform@joint_coord_homo
                human_joints_final[name] = joint_coord_transformed_homo[:3]
            human_points = np.array(human_mesh.vertices)


            human_choices = np.random.choice(len(human_points), len(human_points), replace=False)

            human_points = human_points[human_choices]
            human_rgb = 2*np.array(human_mesh.vertex_colors)-1
            human_rgb = human_rgb[human_choices]
            human_normals = np.array(human_mesh.vertex_normals)[human_choices]

            human_instance_ids = np.array([instance_id]*len(human_points))[human_choices]
            human_sem_labels = np.array([sem_label]*len(human_points))[human_choices]
            human_aabb_bbox = human_dict['aabb_corner_xyz'][transform_id]
            scene_data["xyz"] = np.concatenate([scene_data["xyz"], human_points], axis=0).astype(np.float32)
            scene_data["rgb"] = np.concatenate([scene_data["rgb"], human_rgb], axis=0).astype(np.float32)
            scene_data["normal"] = np.concatenate([scene_data["normal"], human_normals], axis=0).astype(np.float32)
            scene_data["instance_ids"] = np.concatenate([scene_data["instance_ids"], human_instance_ids],axis=0)
            scene_data["sem_labels"] = np.concatenate([scene_data["sem_labels"],human_sem_labels],axis=0)

            scene_data["aabb_corner_xyz"] = np.concatenate([scene_data["aabb_corner_xyz"], human_aabb_bbox[None]])
            scene_data["aabb_obj_ids"] = np.concatenate([scene_data["aabb_obj_ids"],np.array([instance_id])])
            data_dict["human_lr_label"] = torch.tensor([0]) if lr=='left' else torch.tensor([1])
            data_dict["has_human"] = torch.tensor([True])
       
        scene_center_xyz = scene_data["xyz"].mean(axis=0)

        original_num_points = scene_data["xyz"].shape[0]
        choices = np.ones(shape=original_num_points, dtype=bool)

        # sample points
        if self.split == "train" and original_num_points > self.data_cfg.max_num_point:
            choices = np.random.choice(original_num_points, self.data_cfg.max_num_point, replace=False)

        # augment the whole scene (only applicable for the train set)
        xyz_augment_matrix = self._get_xyz_augment_matrix()
        data_dict["point_xyz"] = (scene_data["xyz"] - scene_center_xyz)[choices] @ xyz_augment_matrix

        try:
            human_joints_final_augmented = {}
            for name, joint in human_joints_final.items():
                human_joints_final_augmented[name] = (joint - scene_center_xyz) @ xyz_augment_matrix
        except:
            print(f"Error: Non Human: {scene_id}: {object_id}")

        final_joint_matrix = []
        final_joint_name_idx_map = {}
        try:
            for pidx, [name, joints] in enumerate(human_joints_final_augmented.items()):
                final_joint_name_idx_map[name] = pidx
                final_joint_matrix.append(joints)
        except:
            print(f"Error: Non Human: {scene_id}: {object_id}")
        final_human_joint_matrix = np.stack([final_joint_matrix])
        data_dict["human_joint_matrix"] = final_human_joint_matrix
        data_dict["joint_name_idx_map"] = final_joint_name_idx_map
        point_normal = scene_data["normal"][choices] @ np.linalg.inv(xyz_augment_matrix).transpose()
        point_rgb = scene_data["rgb"][choices]
        data_dict["instance_ids"] = scene_data["instance_ids"][choices]
        data_dict["sem_labels"] = scene_data["sem_labels"][choices].astype(np.int64)

        data_dict["scene_center_xyz"] = scene_center_xyz  # used to recover the original pointcloud coordinates

        instance_num_point = []  # (nInst), int
        unique_instance_ids = np.unique(data_dict["instance_ids"])
        unique_instance_ids = unique_instance_ids[unique_instance_ids != -1]
        data_dict["num_instances"] = unique_instance_ids.shape[0]
        instance_centers = np.empty(shape=(data_dict["point_xyz"].shape[0], 3), dtype=np.float32)

        for index, i in enumerate(unique_instance_ids):
            assert index == i  # make sure it is consecutive
            inst_i_idx = np.where(data_dict["instance_ids"] == i)[0]
            mean_xyz_i = data_dict["point_xyz"][inst_i_idx].mean(0)  # instance_info
            instance_centers[inst_i_idx] = mean_xyz_i  # offset
            instance_num_point.append(inst_i_idx.size)  # instance_num_point

        data_dict["instance_num_point"] = np.array(instance_num_point, dtype=np.int32)
        data_dict["instance_centers"] = instance_centers

        # TODO
        data_dict["all_point_xyz"] = (scene_data["xyz"] - scene_center_xyz) @ xyz_augment_matrix
        data_dict["all_point_rgb"] = (scene_data["rgb"] + 1) / 2

        # augment axis-aligned bounding boxes in the scene
        augmented_gt_aabb_corners_tmp = (scene_data["aabb_corner_xyz"] - scene_center_xyz) @ xyz_augment_matrix
        data_dict["gt_aabb_min_max_bounds"] = np.stack(
            (augmented_gt_aabb_corners_tmp.min(1), augmented_gt_aabb_corners_tmp.max(1)), axis=1
        )
        data_dict["gt_aabb_obj_ids"] = scene_data["aabb_obj_ids"]

        # quantize points to voxels
        point_features = np.empty(shape=(data_dict["point_xyz"].shape[0], 0), dtype=np.float32)
        if self.data_cfg.point_features.use_rgb:
            point_features = np.concatenate((point_features, point_rgb), axis=1)
        if self.data_cfg.point_features.use_normal:
            point_features = np.concatenate((point_features, point_normal), axis=1)
        if self.data_cfg.point_features.use_multiview:
            if not hasattr(self, 'multiview_data'):
                self._open_hdf5()
            point_features = np.concatenate((point_features, self.multiview_data[scene_id][()][choices]), axis=1)

        point_features = np.concatenate((point_features, data_dict["point_xyz"]), axis=1)

        data_dict["voxel_xyz"], data_dict["voxel_features"], _, data_dict["voxel_point_map"] = ME.utils.sparse_quantize(
            coordinates=data_dict["point_xyz"] - data_dict["point_xyz"].min(axis=0), features=point_features, return_index=True,
            return_inverse=True, quantization_size=self.data_cfg.voxel_size
        )
        data_dict["scene_id"] = scene_id
        return data_dict

