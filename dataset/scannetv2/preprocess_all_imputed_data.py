import os
import csv
import json
import torch
import hydra
import numpy as np
import open3d as o3d
from os import cpu_count
from functools import partial
from tqdm.contrib.concurrent import process_map
import torch.multiprocessing as mp
from tqdm import tqdm
mp.set_start_method('spawn', force=True)

def get_semantic_mapping_file(file_path, mapping_name):
    label_mapping = {}
    mapping_col_idx = {
        "nyu40": 4,
        "eigen13": 5,
        "mpcat40": 16
    }
    with open(file_path, "r") as f:
        tsv_file = csv.reader(f, delimiter="\t")
        next(tsv_file)  # skip the header
        for line in tsv_file:
            label_mapping[line[1]] = int(line[mapping_col_idx[mapping_name]])
    return label_mapping

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


def read_mesh_file(file_path, axis_align_matrix):
    mesh = o3d.io.read_triangle_mesh(file_path)
    if axis_align_matrix is not None:
        mesh.transform(axis_align_matrix)  # align the mesh
    mesh.compute_vertex_normals()
    return np.asarray(mesh.vertices, dtype=np.float32), \
           np.rint(np.asarray(mesh.vertex_colors) * 255).astype(np.uint8), \
           np.asarray(mesh.vertex_normals, dtype=np.float32)


def get_semantic_labels(obj_name_to_segs, seg_to_verts, num_verts, label_map, valid_semantic_mapping):
    # create a map, skip invalid labels to make the final semantic labels consecutive
    filtered_label_map = {}
    for i, valid_id in enumerate(valid_semantic_mapping):
        filtered_label_map[valid_id] = i

    semantic_labels = np.full(shape=num_verts, fill_value=-1, dtype=np.int8)  # max value: 127
    for label, segs in obj_name_to_segs.items():
        for seg in segs:
            verts = seg_to_verts[seg]
            if label_map[label] not in filtered_label_map:
                semantic_labels[verts] = 20
            elif label_map[label] == 22:
                semantic_labels[verts] = -1
            else:
                semantic_labels[verts] = filtered_label_map[label_map[label]]
    return semantic_labels


def read_agg_file(file_path, label_map, invalid_ids):
    object_id_to_segs = {}
    obj_name_to_segs = {}
    with open(file_path, "r") as f:
        data = json.load(f)
    for group in data['segGroups']:
        object_name = group['label']
        if object_name not in label_map:
            object_name = "case"  # TODO: randomly assign a name mapped to "objects"
        if label_map[object_name] in invalid_ids:
            # skip room architecture
            continue
        segments = group['segments']
        object_id_to_segs[group["objectId"]] = segments
        if object_name in obj_name_to_segs:
            obj_name_to_segs[object_name].extend(segments)
        else:
            obj_name_to_segs[object_name] = segments.copy()
    return object_id_to_segs, obj_name_to_segs


def read_seg_file(file_path):
    seg2verts = {}
    with open(file_path, "r") as f:
        data = json.load(f)
    for vert, seg in enumerate(data['segIndices']):
        if seg not in seg2verts:
            seg2verts[seg] = []
        seg2verts[seg].append(vert)
    return seg2verts


def get_instance_ids(object_id2segs, seg2verts, sem_labels):
    instance_ids = np.full(shape=len(sem_labels), fill_value=-1, dtype=np.int16)
    for object_id, segs in object_id2segs.items():
        for seg in segs:
            verts = seg2verts[seg]
            instance_ids[verts] = object_id
    return instance_ids


def get_aabbs(xyz, instance_ids):
    unique_inst_ids = np.unique(instance_ids)
    unique_inst_ids = unique_inst_ids[unique_inst_ids != -1]  # skip the invalid id
    num_of_aabb = unique_inst_ids.shape[0]
    aabb_corner_points = np.empty(shape=(num_of_aabb, 8, 3), dtype=np.float32)
    aabb_obj_ids = np.empty(shape=num_of_aabb, dtype=np.int16)

    combinations = np.array(np.meshgrid([0, 1], [0, 1], [0, 1], copy=False), dtype=np.float32).T.reshape(-1, 3)
    for i, instance_id in enumerate(unique_inst_ids):
        point_indices = instance_ids == instance_id
        object_points = xyz[point_indices]
        min_corner = object_points.min(axis=0)
        max_corner = object_points.max(axis=0)
        corner_points = min_corner + (max_corner - min_corner) * combinations
        aabb_corner_points[i] = corner_points
        aabb_obj_ids[i] = instance_id
    return aabb_corner_points, aabb_obj_ids


def process_one_scene(scene, paths, split, label_map, invalid_ids, valid_semantic_mapping, translated_transforms_path, human_data_dir):
    mesh_file_path = os.path.join(paths["raw_scene_path"], scene, scene + '_vh_clean_2.ply')
    agg_file_path = os.path.join(paths["raw_scene_path"], scene, scene + '.aggregation.json')
    seg_file_path = os.path.join(paths["raw_scene_path"], scene, scene + '_vh_clean_2.0.010000.segs.json')
    meta_file_path = os.path.join(paths["raw_scene_path"], scene, scene + '.txt')

    # read meta_file
    axis_align_matrix = read_axis_align_matrix(meta_file_path)

    # read mesh file
    xyz, rgb, normal = read_mesh_file(mesh_file_path, axis_align_matrix)

    num_verts = len(xyz)

    sem_labels = None
    object_ids = None
    aabb_obj_ids = None
    aabb_corner_xyz = None
    if os.path.exists(agg_file_path) and os.path.exists(seg_file_path):
        # read seg_file
        seg2verts = read_seg_file(seg_file_path)
        # read agg_file
        object_id_to_segs, obj_name_to_segs = read_agg_file(agg_file_path, label_map, invalid_ids)
        # get semantic labels
        sem_labels = get_semantic_labels(obj_name_to_segs, seg2verts, num_verts, label_map, valid_semantic_mapping)
        # get instance ids
        object_ids = get_instance_ids(object_id_to_segs, seg2verts, sem_labels)
        # get axis-aligned bounding boxes
        aabb_corner_xyz, aabb_obj_ids = get_aabbs(xyz, object_ids)

    translated_transforms = torch.load(f"{translated_transforms_path}/{scene}.pth")
    human_class = valid_semantic_mapping.index(31)
    human_instance = max(object_ids) + 1
    axis_aligned_torch = torch.tensor(axis_align_matrix)
    combinations = torch.tensor(np.array(np.meshgrid([0, 1], [0, 1], [0, 1]), copy=False), device="cuda").permute(0,1,3,2).reshape(-1, 3)
    final_transforms = {}
    for i in tqdm(translated_transforms):
        obj_list = translated_transforms[i]
        final_transforms[i] = {}
        final_transforms[i]['left'] = []
        final_transforms[i]['right'] = []
        for j,human in enumerate(obj_list):
           human_path = f"{human_data_dir}/{human['human_name']}/meshes/l_h_20.ply"
           human_mesh = o3d.io.read_triangle_mesh(human_path)
           human_mesh_verts = torch.tensor(np.asarray(human_mesh.vertices), device = "cuda")
           human_mesh_verts_homo = torch.cat([human_mesh_verts, torch.tensor([[1]]*human_mesh_verts.shape[0],device="cuda")], dim = 1)
           left_transforms = human["left"]["human_transform"]
           if left_transforms.size(0)>5:
               left_perm = torch.randperm(left_transforms.size(0))
               left_idx = left_perm[:5]
               left_transform_samples = left_transforms[left_idx]
               left_angle_samples = translated_transforms[i][j]["left"]['angle'][left_idx]
               left_angle_bin_samples = translated_transforms[i][j]["left"]['angle_bin'][left_idx]
           else:
               left_transform_samples = left_transforms.clone()
               left_angle_samples = translated_transforms[i][j]["left"]['angle'].clone()
               left_angle_bin_samples = translated_transforms[i][j]["left"]['angle_bin'].clone()

           axis_aligned_left_transform_samples = torch.einsum("ij,njk->nik", axis_aligned_torch.double().cuda(), left_transform_samples.double().cuda())

           right_transforms = human["right"]["human_transform"]

           if right_transforms.size(0)>5:
               right_perm = torch.randperm(right_transforms.size(0))
               right_idx = right_perm[:5]
               right_transform_samples = right_transforms[right_idx]
               right_angle_samples = translated_transforms[i][j]["right"]['angle'][right_idx]
               right_angle_bin_samples = translated_transforms[i][j]["right"]['angle_bin'][right_idx]
           else:
               right_transform_samples = right_transforms.clone()
               right_angle_samples = translated_transforms[i][j]["right"]['angle']
               right_angle_bin_samples = translated_transforms[i][j]["right"]['angle_bin']

           axis_aligned_right_transform_samples = torch.einsum("ij,njk->nik", axis_aligned_torch.double().cuda(), right_transform_samples.double().cuda())

           left_transformed_meshes = torch.einsum("mij,nj->mni",axis_aligned_left_transform_samples,human_mesh_verts_homo)[:,:,:3]
           right_transformed_meshes = torch.einsum("mij,nj->mni",axis_aligned_right_transform_samples,human_mesh_verts_homo)[:,:,:3]

           min_corners_left = left_transformed_meshes.min(dim=1)[0]
           max_corners_left = left_transformed_meshes.max(dim=1)[0]

           corner_points_left = min_corners_left[:,None,:] + (max_corners_left - min_corners_left)[:,None,:] * combinations.repeat(min_corners_left.shape[0],1,1)

           min_corners_right = right_transformed_meshes.min(dim=1)[0]
           max_corners_right = right_transformed_meshes.max(dim=1)[0]

           corner_points_right = min_corners_right[:,None,:] + (max_corners_right - min_corners_right)[:,None,:] * combinations.repeat(min_corners_right.shape[0],1,1)


           if left_transforms.size(0)>0:
               final_transforms[i]['left'].append({
                        "human_name": human['human_name'],
                        "transforms": left_transform_samples.double().cpu().numpy(),
                        "axis_aligned_transforms": axis_aligned_left_transform_samples.double().cpu().numpy(),
                        "aabb_corner_xyz": corner_points_left.cpu().double().numpy(),
                        "angle": left_angle_samples,
                        "angle_bin": left_angle_bin_samples,
                        "instance_id": human_instance,
                        "sem_label": human_class
                   })

           if right_transforms.size(0)>0:
               final_transforms[i]['right'].append({
                       "human_name": human['human_name'],
                       "transforms": right_transform_samples.double().cpu().numpy(),
                       "axis_aligned_transforms": axis_aligned_right_transform_samples.double().cpu().numpy(),
                       "aabb_corner_xyz": corner_points_right.cpu().double().numpy(),
                       "angle": right_angle_samples,
                       "angle_bin": right_angle_bin_samples,
                       "instance_id": human_instance,
                       "sem_label": human_class
                   })

    torch.save(
        {"xyz": xyz, "rgb": rgb, "normal": normal, "sem_labels": sem_labels,
            "instance_ids": object_ids, "aabb_obj_ids": aabb_obj_ids, "aabb_corner_xyz": aabb_corner_xyz, "human_info": final_transforms},
        os.path.join(paths["scene_dataset_path"], f"{split}_imputed", f"{scene}.pth")
    )


@hydra.main(version_base=None, config_path="../../config", config_name="global_config")
def main(cfg):
    max_workers = cpu_count() if "workers" not in cfg else cfg.workers
    print(f"\nUsing {max_workers} CPU threads.")

    # read semantic label mapping file
    label_map = get_semantic_mapping_file(cfg.data.scene_metadata.label_mapping_file,
                                          cfg.data.scene_metadata.semantic_mapping_name)

    for split in ("val","test"):
        output_path = os.path.join(cfg.data.scene_dataset_path, split)
        if os.path.exists(output_path):
            print(f"\nWarning: output directory {os.path.abspath(output_path)} exists, "
                  f"existing files will be overwritten.")
        os.makedirs(output_path, exist_ok=True)
        with open(getattr(cfg.data.scene_metadata, f"{split}_scene_ids")) as f:
            split_list1 = [line.strip() for line in f]


        print(f"==> Processing {split} split ...")
        paths = {
                    "raw_scene_path": cfg.data.raw_scene_path,
                    "scene_dataset_path":cfg.data.scene_dataset_path
                }
        process_map(
            partial(
                process_one_scene, paths = paths, split=split, label_map=label_map,
                invalid_ids=cfg.data.scene_metadata.invalid_semantic_labels,
                valid_semantic_mapping=cfg.data.scene_metadata.valid_semantic_mapping, translated_transforms_path=f"{cfg.data.imputer_target_dir}",
                human_data_dir = f"{cfg.data.human_dir}"
            ), split_list, chunksize=1, max_workers=max_workers
        )

        print(f"==> Processing {split} split ...")
if __name__=="__main__":
    main()
