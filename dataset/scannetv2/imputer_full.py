import torch
import trimesh
import hydra
import json
from trimesh.exchange.load import load
import numpy as np
import torch.multiprocessing as mp
import cv2
import os
import pickle
from tqdm import tqdm
import imputer
from copy import deepcopy as dcpy
import torch.nn.functional as F
from time import perf_counter
import open3d as o3d

print(imputer.visibility_grid)

#For multiprocessing with torch
mp.set_start_method('spawn', force=True)
mp.set_sharing_strategy('file_system')





MIN_FLOOR_HEIGHT = 4
BIN_OCC_THICKNESS = 12
VOXELIZATION_PITCH = 0.025



class TorchVoxelGrid:
    @classmethod
    def fromVoxels(cls, trimesh_voxel_grid:trimesh.voxel.VoxelGrid, device:str = "cuda"):
        return cls(torch.tensor(trimesh_voxel_grid.matrix),torch.tensor(trimesh_voxel_grid.transform), device=device)


    def __init__(self, grid, transform, device:str = "cuda"):
        self.transform:torch.Tensor = transform.to(device)
        self.inverse_transform:torch.Tensor = self.transform.inverse()
        self.matrix:torch.Tensor = grid.to(device)
        self.device = device

    def points_to_indices(self, points:torch.Tensor):
        points = self.homogenize(points)
        points = (self.inverse_transform@points.t()).t()[:,:3]
        return points.round().int()

    def indices_to_points(self, indices:torch.Tensor):
        if indices.dtype != torch.int:
            ValueError("indices must be int")
        indices = self.homogenize(indices)
        return (self.transform@indices.double().t()).t()[:,:3]

    def homogenize(self, points:torch.Tensor):
        ones = torch.ones(points.shape[0], device = self.device, dtype = points.dtype)[:,None]
        return torch.cat([points, ones], dim=1)
    @property
    def shape(self):
        return self.matrix.shape

    def to(self, device:str):
        return TorchVoxelGrid(self.matrix, self.transform, device)






def get_humans(human_base):
    full_human_data = {}
    reflect_trans = np.eye(4)
    reflect_trans[0,0] = -1
    reflect_trans3 = np.eye(3)
    reflect_trans3[0,0] = -1


    for human_type in os.listdir(human_base):
        human_type_list = []
        mesh_files_dir = f"{human_base}/{human_type}/meshes"
        joints_files_dir = f"{human_base}/{human_type}/joints"
        poses_files_dir = f"{human_base}/{human_type}/poses"
        for fname in sorted(os.listdir(mesh_files_dir)):
            #print(fname)
            human_pose_name = fname.split(".")[0]
            with open(f"{mesh_files_dir}/{fname}", "rb") as f:
                human_mesh_left:trimesh.Trimesh = trimesh.exchange.load.load(f,"ply")
                human_mesh_right = human_mesh_left.copy()
                human_mesh_right = human_mesh_right.apply_transform(reflect_trans)
            with open(f"{joints_files_dir}/{human_pose_name}.pkl","rb") as f:
                human_joints_left = pickle.load(f)
            vox_grid_left:trimesh.voxel.VoxelGrid = human_mesh_left.voxelized(VOXELIZATION_PITCH)
            vox_grid_left.fill()
            vox_grid_right:trimesh.voxel.VoxelGrid = human_mesh_right.voxelized(VOXELIZATION_PITCH)
            vox_grid_right.fill()
            human_joints_right = {}

            for key in human_joints_left:
                if "right" in key:
                    human_joints_right[key.replace("right", "left")] = reflect_trans3@human_joints_left[key]
                elif "left" in key:
                    human_joints_right[key.replace("left","right")] = reflect_trans3@human_joints_left[key]
                else:
                    human_joints_right[key] = reflect_trans3@human_joints_left[key]

            human_type_list.append({
                "left_mesh": human_mesh_left,
                "joints_left": human_joints_left,
                "voxel_tensor_left": TorchVoxelGrid.fromVoxels(vox_grid_left, device="cpu"),
                "right_mesh": human_mesh_right,
                "joints_right": human_joints_right,
                "voxel_tensor_right": TorchVoxelGrid.fromVoxels(vox_grid_right, device="cpu"),
            })
        full_human_data[human_type] = human_type_list
    return full_human_data








def close_scene(vox_mat:TorchVoxelGrid, start_idx:int,end_idx:int):
    mat_full=(torch.sum(vox_mat.matrix.permute(2,0,1),dim=0)>0).type(torch.uint8).numpy()
    mat_mat=(torch.sum(vox_mat.matrix.permute(2,0,1)[start_idx:end_idx],dim=0)<=0).type(torch.uint8).numpy()
    cnts,h=cv2.findContours(mat_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_cnt=[]
    for cnt in cnts:
        if len(cnt)>len(max_cnt):
            max_cnt=cnt
    mat_mat=cv2.drawContours(mat_mat,[max_cnt],-1,(0,0,0),2)
    filled=np.zeros_like(mat_mat)
    filled=cv2.drawContours(filled, [max_cnt], -1, (1,1,1), thickness=cv2.FILLED)
    mat_mat=mat_mat*filled
    return mat_mat,filled






def ret_3_planes(point_set_1, vert1):
    vecs1=point_set_1-vert1
    norms1=torch.linalg.cross(vecs1[0][None],vecs1, dim=1)
    norms1[0]=vecs1[0]
    planes1=torch.cat([norms1,-(norms1@vert1.t())],dim=1)
    return planes1



def ret_planes(points):
    vecs=points-points[0]
    dots=vecs@vecs.t()
    a=dots<=0.001
    b=dots>=-0.001
    c=a*b
    d=c.sum(1)-1
    if (d==0).shape[0]!=3 or (d==1).shape[0]!=3 or (d==7).shape[0]!=1 or (d==0).shape[0]!=1:
        a=dots<=0.00001
        b=dots>=-0.00001
        c=a*b
        d=c.sum(1)-1
    point_set_1=points[d==3]
    point_set_2=points[d==1]
    vert1=points[d==7]
    vert2=points[d==0]
    planes1=ret_3_planes(point_set_1, vert1)
    planes2=ret_3_planes(point_set_2, vert2)
    planes=torch.cat([planes1,planes2])
    return planes


def ret_bbox_mask(all_verts, points):
    bbox_planes=ret_planes(points)
    centroid=torch.sum(points,0)/8
    centroid_homo=torch.cat([centroid, torch.tensor([1]).cuda()])
    centroid_conds=bbox_planes@centroid_homo.t()
    homonizer=torch.ones(all_verts.shape[0])[:,None].cuda()
    all_verts_homo=all_verts.clone()
    all_verts_homo=torch.cat([all_verts_homo,homonizer],dim=1).t()
    vert_conds=bbox_planes@all_verts_homo
    ds=(vert_conds*centroid_conds.repeat(vert_conds.shape[1],1).t())
    mask=torch.prod(ds>0.0, dim=0)==0
    return mask






def ret_obj_removed_scene_mesh(scene_mesh, points):
    all_verts=torch.tensor(np.asarray(scene_mesh.vertices), device="cuda")
    all_faces=torch.tensor(np.asarray(scene_mesh.faces), device = "cuda")
    mask=ret_bbox_mask(all_verts, points)
    obj_removed_verts=all_verts[mask]
    removed_vert_indices=torch.where(mask==0)
    removed_vert_indices=removed_vert_indices[0]
    face_mask=torch.zeros(all_faces.shape[0],dtype=torch.bool, device='cuda')
    all_faces=all_faces.cuda()
    pbar=tqdm(removed_vert_indices)
    for removed_vert_idx in pbar:
        face_mask+=(all_faces[:,0]==removed_vert_idx)+(all_faces[:,1]==removed_vert_idx)+(all_faces[:,2]==removed_vert_idx)
    face_mask=face_mask<=0
    scene_mesh.update_faces(face_mask.cpu().numpy())
    scene_mesh.update_vertices(mask.cpu().numpy())
    return scene_mesh






def pad_obj_removed(obj_removed_scene_grid_tensor, centroid_indices, voxel_grid_scene):
    start_idx_y = 0
    start_idx_x = 0
    fin_shape = list(voxel_grid_scene.shape)
    fin_shape[2] = max(fin_shape[2],120)
    padded_obj_removed_scene_grid_tensor = torch.zeros((fin_shape), device = "cuda")
    if obj_removed_scene_grid_tensor.shape[0]<voxel_grid_scene.shape[0]:
        if centroid_indices[0]>voxel_grid_scene.shape[0]/2:
            start_idx_x = 0
        else:
            start_idx_x = voxel_grid_scene.shape[0]-obj_removed_scene_grid_tensor.shape[0]

    if obj_removed_scene_grid_tensor.shape[1]<voxel_grid_scene.shape[1]:
        if centroid_indices[1]>voxel_grid_scene.shape[1]/2:
            start_idx_y = 0
        else:
            start_idx_y = voxel_grid_scene.shape[1]-obj_removed_scene_grid_tensor.shape[1]


    padded_obj_removed_scene_grid_tensor[start_idx_x:start_idx_x+obj_removed_scene_grid_tensor.shape[0], start_idx_y:start_idx_y+obj_removed_scene_grid_tensor.shape[1], :obj_removed_scene_grid_tensor.shape[2]] = obj_removed_scene_grid_tensor
    return padded_obj_removed_scene_grid_tensor







def get_all_los(point, a):
    x0,y0,z0=point
    a =  imputer.visibility_grid(a, x0,y0,z0)
    return a.float()






def occupancy_checker(checking_grid, voxel_grid_human):
    out = torch.zeros((checking_grid.shape[0]-voxel_grid_human.shape[0],checking_grid.shape[1]-voxel_grid_human.shape[1]), device = "cuda")
    for i in tqdm(range(checking_grid.shape[0]-voxel_grid_human.shape[0])):
        for j in (range(checking_grid.shape[1]-voxel_grid_human.shape[1])):
            out[i,j] = (checking_grid[i:i+voxel_grid_human.shape[0],j:j+voxel_grid_human.shape[1],:]*voxel_grid_human).sum()
    return out






def get_bin_occ_old(voxel_grid_scene, voxel_grid_human, shoulder_index, floor_height_index):
    checking_grid = voxel_grid_scene.matrix[:,:, floor_height_index:floor_height_index+voxel_grid_human.shape[2]]==0
    occupancy1 = occupancy_checker(checking_grid.double(), voxel_grid_human.matrix.cuda())
    bin_occupancy1=occupancy1==voxel_grid_human.matrix.sum()
    return bin_occupancy1






def get_bin_occ(voxel_grid_scene, voxel_grid_human, shoulder_index, floor_height_index):
    print(f"floor_height_index: {floor_height_index}")
    checking_grid = torch.ones((voxel_grid_scene.shape[0],voxel_grid_scene.shape[1],max(120,voxel_grid_scene.shape[2])), device="cuda")
    checking_grid[:,:,:voxel_grid_scene.shape[2]] = voxel_grid_scene.matrix==0
    checking_grid = checking_grid[:,:, floor_height_index:floor_height_index+voxel_grid_human.shape[2]]
    occupancy1 = occupancy_checker(checking_grid.double(), voxel_grid_human.matrix.cuda())
    bin_occupancy1=occupancy1==voxel_grid_human.matrix.sum()
    return bin_occupancy1







def get_valid_coords(voxel_grid_scene, voxel_grid_human, final_vis_grid, shoulder_index, floor_height_index, bin_occupancy1):
    end_x, end_y= final_vis_grid.shape[0]-voxel_grid_human.shape[0]+shoulder_index[0], final_vis_grid.shape[1]-voxel_grid_human.shape[1]+shoulder_index[1]
    shoulder_vis_cond = final_vis_grid[shoulder_index[0]: end_x, shoulder_index[1]: end_y, shoulder_index[2]+floor_height_index]
    final_cond = torch.zeros(voxel_grid_scene.shape[:2], device = "cuda")
    newt = (bin_occupancy1*shoulder_vis_cond)
    final_cond[shoulder_index[0]: end_x, shoulder_index[1]: end_y] = newt
    # final_indices_2d = torch.where(final_cond)
    # final_z = torch.tensor([[shoulder_index[2]+self.floor_height_index]*final_indices_2d[0].shape[0]], device = "cuda")
    # final_indices_3d = torch.stack([final_indices_2d[0], final_indices_2d[1], final_z[0]]).t()
    # final_coords = self.voxel_grid_scene.indices_to_points(final_indices_3d.cpu().numpy())
    # plt.imshow(final_cond.cpu())
    # return torch.tensor(final_coords, device = "cuda"), final_indices_3d, bin_occupancy1[0,0],shoulder_vis_cond
    return final_cond







def thicken_occupancies(bin_occupancy):
    cnts, h = cv2.findContours(bin_occupancy.to(torch.uint8).cpu().numpy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bin_occupancy_thick = cv2.drawContours(np.ones(bin_occupancy.shape, dtype=np.uint8),cnts, -1,(0,0,0),BIN_OCC_THICKNESS)*bin_occupancy.cpu().numpy()
    bin_occupancy_thick = torch.tensor(bin_occupancy_thick, device="cuda")
    return bin_occupancy_thick

# %%
def process_obj(object_id, scene_mesh, voxel_grid_scene, closed, floor_height_index, points, bin_occupancies,full_human_data, retQ):
    v1 = perf_counter()
    points = points.cuda()
    voxel_grid_scene = voxel_grid_scene.to("cuda")
    closed = closed.cuda()

    centroid=points.mean(dim=0)
    centroid_indices=voxel_grid_scene.points_to_indices(centroid[None])[0]

    obj_removed_scene_mesh= ret_obj_removed_scene_mesh(scene_mesh.copy(),points)
    obj_removed_scene_grid=obj_removed_scene_mesh.voxelized(0.025)
    obj_removed_scene_grid_tensor = torch.tensor(obj_removed_scene_grid.matrix[:,:,:], device = "cuda")
    padded_obj_removed_scene_grid = pad_obj_removed(obj_removed_scene_grid_tensor, centroid_indices, voxel_grid_scene)
    obj_removed_scene_grid = TorchVoxelGrid(padded_obj_removed_scene_grid, voxel_grid_scene.transform)

    los_regions = get_all_los(centroid_indices, ((padded_obj_removed_scene_grid==0)).double())
    final_vis_grid=torch.zeros_like(padded_obj_removed_scene_grid, dtype=torch.float32, device = "cuda")
    final_vis_grid[:los_regions.shape[0],:los_regions.shape[1],:los_regions.shape[2]] = los_regions
    final_vis_grid = (((los_regions>0.31)))*closed

    valid_points_all = {}


    for name, human_list in full_human_data.items():
        valid_points_all[name]={}
        t1 = perf_counter()
        bin_occupancy_left = bin_occupancies[name]['left']

        human_data = human_list[6]
        human_mesh_left  = human_data['left_mesh']
        voxel_grid_human_left = human_data['voxel_tensor_left'].to("cuda")
        human_keypoints_left = human_data['joints_left']
        shoulder_left, index_1_left = human_keypoints_left['left_shoulder'], human_keypoints_left['left_index1']
        shoulder_idx_left = voxel_grid_human_left.points_to_indices(torch.tensor(shoulder_left, device = "cuda")[None])[0]

        region1_left = get_valid_coords(voxel_grid_scene, voxel_grid_human_left, final_vis_grid, shoulder_idx_left, floor_height_index, bin_occupancy_left)

        # human_data = human_list[-1]
        # voxel_grid_human = human_data['voxel_tensor'].to("cuda")
        # human_keypoints = human_data['joints']
        # shoulder, index_1 = human_keypoints['left_shoulder'], human_keypoints['left_index1']
        # shoulder_idx = voxel_grid_human.points_to_indices(torch.tensor(shoulder, device = "cuda")[None])[0]
        # region2 = get_valid_coords(voxel_grid_scene, voxel_grid_human, final_vis_grid, shoulder_idx, floor_height_index)
        # val1_end = perf_counter()

        final_cond_left = region1_left
        # val2= perf_counter()
        bin_occupancy_right = bin_occupancies[name]['right']

        final_indices_2d_left = torch.where(final_cond_left)
        final_z_left = torch.tensor([[shoulder_idx_left[2]+floor_height_index]*final_indices_2d_left[0].shape[0]], device = "cuda")
        final_indices_3d_left = torch.stack([final_indices_2d_left[0], final_indices_2d_left[1], final_z_left[0]]).t()
        final_coords_left = voxel_grid_scene.indices_to_points(final_indices_3d_left)
        valid_points_all[name]['left'] = final_coords_left.clone().cpu()


        # human_mesh_left  = human_data['left_mesh']
        voxel_grid_human_right = human_data['voxel_tensor_right'].to("cuda")
        human_keypoints_right = human_data['joints_right']
        shoulder_right, index_1_right = human_keypoints_right['right_shoulder'], human_keypoints_right['right_index1']
        shoulder_idx_right = voxel_grid_human_right.points_to_indices(torch.tensor(shoulder_right, device = "cuda")[None])[0]
        region1_right = get_valid_coords(voxel_grid_scene, voxel_grid_human_right, final_vis_grid, shoulder_idx_right, floor_height_index, bin_occupancy_right)

        # human_data = human_list[-1]
        # voxel_grid_human = human_data['voxel_tensor'].to("cuda")
        # human_keypoints = human_data['joints']
        # shoulder, index_1 = human_keypoints['left_shoulder'], human_keypoints['left_index1']
        # shoulder_idx = voxel_grid_human.points_to_indices(torch.tensor(shoulder, device = "cuda")[None])[0]
        # region2 = get_valid_coords(voxel_grid_scene, voxel_grid_human, final_vis_grid, shoulder_idx, floor_height_index)
        # val1_end = perf_counter()

        final_cond_right = region1_right
        # val2= perf_counter()

        final_indices_2d_right = torch.where(final_cond_right)
        final_z_right = torch.tensor([[shoulder_idx_right[2]+floor_height_index]*final_indices_2d_right[0].shape[0]], device = "cuda")
        final_indices_3d_right = torch.stack([final_indices_2d_right[0], final_indices_2d_right[1], final_z_right[0]]).t()
        final_coords_right = voxel_grid_scene.indices_to_points(final_indices_3d_right)
        valid_points_all[name]['right'] = final_coords_right.clone().cpu()
        print(f"human: {name} completed for obj: {object_id}")

    v2 = perf_counter()
    retQ.put({"object_id": object_id,
               "valid_points": valid_points_all,
               "obj_processing_time": v2-v1,
               "points": points.clone().cpu(),
               "final_vis_grid": final_vis_grid.to("cpu")
               })
# %%
def process_scene(scans_base_path, points_dir, scene_id, preprocess_save_path, human_data_path):


    points_agg=torch.load(f"{points_dir}/{scene_id}.pth")
    # humans_data:dict = full_human_data

    scene_path = f"{scans_base_path}/{scene_id}/{scene_id}_vh_clean_2.ply"
    with open(scene_path, "rb") as f:
        scene_mesh:trimesh.base.Trimesh = trimesh.exchange.load.load(f, "ply")

    full_human_data = get_humans(human_data_path)
    voxel_grid_scene:trimesh.voxel.VoxelGrid=scene_mesh.voxelized(VOXELIZATION_PITCH)
    voxel_grid_scene:TorchVoxelGrid = TorchVoxelGrid.fromVoxels(voxel_grid_scene, device = "cpu")
    closed = torch.tensor(close_scene(voxel_grid_scene,0,int(voxel_grid_scene.shape[2]))[1], device="cpu")
    closed = closed[:,:,None].repeat(1,1,max(voxel_grid_scene.shape[2],120))

    #TODO: Change Path
    meta_data = torch.load(f"dataset/scannetv2/train/{scene_id}.pth")

    o3d_mesh = o3d.io.read_triangle_mesh(scene_path)
    mesh_verts = np.asarray(o3d_mesh.vertices)
    floor_idxes = torch.where(torch.tensor(meta_data['sem_labels'])==1)
    floor_points = torch.tensor(mesh_verts)[floor_idxes]

    floor_height = min(floor_points[:,2].mean().item()+0.04, np.percentile(floor_points[:,2].cpu().numpy(), 85))

    floor_height_index = max(MIN_FLOOR_HEIGHT,voxel_grid_scene.points_to_indices(torch.tensor(np.array([[0,0,floor_height]])))[0,2])
    floor_height_vector = voxel_grid_scene.points_to_indices(torch.tensor([[0,0,floor_height]]).double())

    bin_occupancies = {}
    bin_occ_2 = {}
    for name, human_list in full_human_data.items():
        bin_occupancies[name] = {}
        bin_occ_2[name] = {}
        human_data = human_list[6]
        human_mesh_left  = human_data['left_mesh']
        voxel_grid_human_left = human_data['voxel_tensor_left'].to("cuda")
        human_keypoints_left = human_data['joints_left']
        shoulder_left, index_1_left = human_keypoints_left['left_shoulder'], human_keypoints_left['left_index1']
        shoulder_idx_left = voxel_grid_human_left.points_to_indices(torch.tensor(shoulder_left, device = "cuda")[None])[0]
        bin_occupnacy_left = get_bin_occ(voxel_grid_scene.to("cuda"), voxel_grid_human_left, shoulder_idx_left, floor_height_index)
        bin_occupnacy_left_thick = thicken_occupancies(bin_occupnacy_left)
        bin_occupancies[name]['left'] = bin_occupnacy_left_thick
        bin_occ_2[name]['left'] = bin_occupnacy_left

        voxel_grid_human_right = human_data['voxel_tensor_right'].to("cuda")
        human_keypoints_right = human_data['joints_right']
        shoulder_right, index_1_right = human_keypoints_right['right_shoulder'], human_keypoints_right['right_index1']
        shoulder_idx_right = voxel_grid_human_right.points_to_indices(torch.tensor(shoulder_right, device = "cuda")[None])[0]
        bin_occupnacy_right = get_bin_occ(voxel_grid_scene.to("cuda"), voxel_grid_human_right, shoulder_idx_right, floor_height_index)
        bin_occupnacy_right_thick = thicken_occupancies(bin_occupnacy_right)
        bin_occupancies[name]['right'] = bin_occupnacy_right_thick
        bin_occ_2[name]['right'] = bin_occupnacy_right
        print(f"{name}-left: {bin_occupnacy_left.shape}")


    max_workers = 10

    mq = mp.Manager().Queue()
    results = []
    obj_processes_batch = []
    obj_processes_batches = []

    processes_list=[]
    deck = {}

    for i in range(len(points_agg["bboxes_points"])):
        points = points_agg["bboxes_points"][i]
        pr = mp.Process(target=process_obj, args=[i, scene_mesh, voxel_grid_scene, closed, floor_height_index, points, bin_occupancies, dcpy(full_human_data), mq])
        obj_processes_batch.append(pr)
        if len(obj_processes_batch)>=max_workers:
                    obj_processes_batches.append(obj_processes_batch)
                    obj_processes_batch = []
    obj_processes_batches.append(obj_processes_batch)

    for i,batch in enumerate(obj_processes_batches):
        print(f"batch {i} started")

        for process in batch:
            process.start()

        for process in batch:
            ret = mq.get()
            ret["bin_occupancies"] = bin_occ_2
            ret['bin_occupancies_thick'] = bin_occupancies
            ret["floor_height_vector"] = floor_height_vector
            ret['voxel_grid_scene'] = voxel_grid_scene
            results.append(dcpy(ret))
            del ret
        for process in batch:
            process.join()
        print(f"batch {i} done")


    deck["bin_occupancies"] = bin_occ_2
    deck['bin_occupancies_thick'] = bin_occupancies
    torch.save(deck,f"{preprocess_save_path}/{scene_id}.pth")

    return results

def sample_point_indices(coords, points):
    centroid = points.mean(0)
    dists = (torch.tensor(coords)-centroid).norm(dim = 1)
    probabs = torch.nn.functional.softmax(-15*dists)
    choices1 = np.random.choice(coords.shape[0], size = 5, replace=False)
    maxes, choices2 = (-dists).topk(5)
    choices = torch.cat([torch.tensor(choices1), choices2])
    return choices

def arm_length_checker(coords, points, shoulder, hand):
    arm_length = (shoulder-hand).norm()
    centroid = points.mean(0)
    centroid[2] = shoulder[2]
    dists = (torch.tensor(coords)-centroid).norm(dim = 1)
    return coords[dists>(1.5*arm_length)]



def rotation_matrices_from_vectors(vecs1, vecs2):
    a, b = (vecs1/vecs1.norm(dim=1)[:, None]), (vecs2/vecs2.norm(dim=1)[:, None])
    v = torch.cross(a,b)
    c = torch.einsum("ni, ni-> n",a,b)
    s = v.norm(dim = 1)
    kmat = torch.zeros((v.shape[0],3,3))
    kmat[:,0,1] = -v[:,2]
    kmat[:,0,2] = v[:,1]
    kmat[:,1,0] = v[:,2]
    kmat[:,1,2] = -v[:,0]
    kmat[:,2,0] = -v[:,1]
    kmat[:,2,1] = v[:,0]
    rotation_matrices = torch.eye(3)[None].repeat(v.shape[0],1,1) + kmat + torch.einsum("nij,njk -> nik", kmat, kmat)* ((1 - c)/(s**2))[:,None,None]
    return rotation_matrices, a, b, kmat, v

def return_transform(pivot, source, dest):
    v1=pivot
    v2=source
    v3=dest
    a=v2-v1
    
    n = a.norm()
    a=a/a.norm()
    a[:,2] = 0
    b=v3-v1
    b[:,2] = 0

    m = b.norm()
    b=b/b.norm()
    r, x,y, kmat, cross=rotation_matrices_from_vectors(a,b)


    wheres = torch.where(r[:,2,2]!=1)[0]
    if len(wheres)>0:
        print(kmat)
    trans=torch.eye(4)[None].repeat(a.shape[0],1,1)
    trans[:,:3,:3]=r
    t=v1.double()-torch.einsum("nij,nj->ni", r,v1)
    trans[:,:3,3]=t
    return trans

def full_transform(source, pointer, dests, object_pts, floor_height_vector):
    vecs=dests-source
    translation_transform=torch.eye(4)
    translation_transform=translation_transform[None].repeat(vecs.shape[0],1,1)
    translation_transform[:,:3,3]=vecs
    vecs[:,2]=0
    obj_cen=torch.sum(object_pts,dim=0)/8
    sources=source[None].repeat(vecs.shape[0],1)
    pointers=pointer[None].repeat(vecs.shape[0],1)
    sources=torch.cat([sources, torch.ones(vecs.shape[0])[None].t()], dim=1)
    pointers=torch.cat([pointers, torch.ones(vecs.shape[0])[None].t()], dim=1)
    sources=torch.einsum("nij,nj->ni",translation_transform,sources)
    pointers=torch.einsum("nij,nj->ni",translation_transform,pointers)
    rot_transes=return_transform(sources[:,:3], pointers[:,:3], obj_cen)
    full_transes=torch.einsum("nij,njk->nik",rot_transes,translation_transform)
    return full_transes


@hydra.main(version_base=None, config_path="../../config", config_name="global_imputed_config")
def main(cfg):
    full_human_data = get_humans(cfg.data.human_dir)
    target_dir = cfg.data.imputer_target_dir
    scans_dir = cfg.data.raw_scene_path
    points_dir = cfg.data.points_dir
    target_dir = cfg.data.imputer_target_dir
    scans_path = cfg.data.raw_scene_path



    with open(cfg.data.scene_metadata.train_scene_ids) as f:
        scene_list_1 = f.read().splitlines()

    #TODO: Remove
    #scene_list_1 = [x.split(".")[0] for x in os.listdir("dataset/scannetv2/")]

    for scene_id in scene_list_1:
        scene_start_time = perf_counter()
        if "_00" not in scene_id:
            continue
        if scene_id+".pth" in os.listdir(target_dir):
            print(f"skipping {scene_id} cuz already done")
            continue

        print(f"started {scene_id}")
        obj_to_hum_to_valids = process_scene(scans_dir, points_dir, scene_id, "obj_to_human_to_valid_points", cfg.data.human_dir)
        scene_list = {}

        for object_id in range(len(obj_to_hum_to_valids)):
            obj_coord_list = []
            obj_id = obj_to_hum_to_valids[object_id]['object_id']
            points = obj_to_hum_to_valids[object_id]['points']

            for human_type in obj_to_hum_to_valids[object_id]['valid_points']:
                floor_height_vector = obj_to_hum_to_valids[object_id]['floor_height_vector']
                coords_left = obj_to_hum_to_valids[object_id]['valid_points'][human_type]['left']
                shoulder_left = torch.tensor(full_human_data[human_type][0]['joints_left']['left_shoulder'])
                hand_left = torch.tensor(full_human_data[human_type][0]['joints_left']['left_index1'])
                coords_left = arm_length_checker(coords_left, points, shoulder_left, hand_left)
                centroid = points.mean(dim=0)
                diff_left = centroid - coords_left
                angles_left = 180-((diff_left[:,2]/diff_left.norm(dim=1)).arccos()*180/torch.pi)
                angles_bin_left = (angles_left/10).round()*10
                final_transforms_left = full_transform(shoulder_left.float(), hand_left.float(), coords_left.float(), points.float(), floor_height_vector)
                fin = {
                    "human_name": human_type,
                }


                coords_right = obj_to_hum_to_valids[object_id]['valid_points'][human_type]['right']
                shoulder_right = torch.tensor(full_human_data[human_type][0]['joints_right']['right_shoulder'])
                hand_right = torch.tensor(full_human_data[human_type][0]['joints_right']['right_index1'])
                coords_right = arm_length_checker(coords_right, points, shoulder_right, hand_right)
                diff_right = centroid - coords_right
                angles_right = 180-((diff_right[:,2]/diff_right.norm(dim=1)).arccos()*180/torch.pi)
                angles_bin_right = (angles_right/10).round()*10
                final_transforms_right = full_transform(shoulder_right.float(), hand_right.float(), coords_right.float(), points.float(), floor_height_vector)
                fin = {
                    "human_name": human_type,
                    "right": {
                            "human_transform": final_transforms_right,
                            "angle": angles_right,
                            "angle_bin": angles_bin_right,
                            "coord": coords_right,
                        },
                    "left": {
                            "human_transform": final_transforms_left,
                            "angle": angles_left,
                            "angle_bin": angles_bin_left,
                            "coord": coords_left,
                        }
                }

                obj_coord_list.append(fin)
            scene_list[obj_id] = obj_coord_list
        print(f"{target_dir}/{scene_id}.pth")
        torch.save(scene_list, f"{target_dir}/{scene_id}.pth")
        torch.cuda.empty_cache()
        scene_end_time = perf_counter()
        print(f"scene_time: {scene_end_time-scene_start_time}")


# %%
if __name__=="__main__":
    main()



