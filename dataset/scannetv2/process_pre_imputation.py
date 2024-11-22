import numpy as np
import torch
from fire import Fire
import ast
import os
from collections import OrderedDict
import point_cloud_utils as pcu

def calc_all_scene_all_corners(scans_dir, pth_dir, save_dir):
    scene_names=sorted(os.listdir(pth_dir))
    for scene_name in scene_names:
        scene_name=scene_name.split('.')[0]
        file_path_transform=os.path.join(scans_dir,scene_name)
        file_path_transform=os.path.join(file_path_transform,scene_name+'.txt')
        file_path_pth=os.path.join(pth_dir,scene_name+'.pth')
        calc_single_scene_all_corners(scans_dir,scene_name,file_path_transform, file_path_pth,save_dir)


def calc_single_scene_all_corners(scan_dir,scene_name,file_path_transform, file_path_pth, save_dir):
    scene_dict=OrderedDict()
    m=OrderedDict()
    n=OrderedDict()
    o=OrderedDict()
    a=read_axis_align_matrix(file_path_transform)
    a=torch.tensor(a).inverse()
    x=torch.load(file_path_pth)
    save_path=os.path.join(save_dir,scene_name+'.pth')
    scene_dir=os.path.join(scan_dir,scene_name)
    scan_file=os.path.join(scene_dir,scene_name+"_vh_clean_2.ply")
    print(scene_name)
    for i in range(len(x['aabb_corner_xyz'])):
        p=x['aabb_corner_xyz'][i]
        p=torch.tensor(p)
        p=torch.cat([torch.tensor(p),torch.ones(8,1)],1).t()
        p=(a@p.double())[:3].t()
    #    v1=[3.35673380, 3.45864153, 1.32902181]
   #     v2=[3.35438776, 2.99998951, 1.27959979]
        centroid=torch.sum(p,0)/8
        m.update({i:p})
        n.update({i:centroid})
        o.update({i:x['aabb_obj_ids'][i]})
    scene_dict['bboxes_points']=m
    scene_dict['bboxes_centroids']=n
    scene_dict['bbox_obj_ids']=o
    #print('\n\n\n\n\n\n')
    torch.save(scene_dict,save_path)

def read_axis_align_matrix(file_path):
    axis_align_matrix = None
    with open(file_path, "r") as f:
        for line in f:
            line_content = line.strip()
            if 'axisAlignment' in line_content:
                axis_align_matrix = [float(x) for x in line_content.strip('axisAlignment = ').split(' ')]
                axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
                break
    return axis_align_matrix

if __name__=="__main__":
    Fire()
