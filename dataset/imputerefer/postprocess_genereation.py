import json
import hydra
import random
import torch
import os
from tqdm import tqdm
if __name__=="__main__":
    for split in ["train","val", "test"]:
        unproc_data_path = f"metadata/PointRefer_unfiltered_{split}.json"
        proc_data_path = f"metadata/PointRefer_filtered_{split}.json"
        transforms_path = f"../scannetv2/val_imputed_newer" if split=="test" else f"../scannetv2/{split}_imputed_newer"
        with open(unproc_data_path) as f:
            unproc_data = json.load(f)
        if os.path.isfile(proc_data_path):
            with open(proc_data_path) as f:
                proc_data = json.load(f)
        #unproc_data = unproc_data[len(proc_data):]
        results = []
        for example in tqdm(unproc_data):
            trans_dat = torch.load(f"{transforms_path}/{example['scene_id']}.pth")
            obj_stuff = trans_dat['obj_stuff'][int(example['object_id'])]
            selection = random.randint(0,3)
            if len( example['potential_descriptions']) == 3 and selection <3 and not obj_stuff['human_transforms'][0] is None:
                desc = example['potential_descriptions'][selection][3:].strip()
                if len(desc)<=2:
                    desc = example['base_description']
            else:
                desc = example['base_description']
            desc = desc.replace(". "," . ").replace("  "," ").replace("(","").replace('"', '').replace("'","").lower()
            desc = desc[:-1]+" ." if desc[-1]=='.' else desc
            tokens = desc.split(" ")

            result = {
                    "scene_id": example['scene_id'],
                    "object_id": example['object_id'],
                    "object_name": example["object_name"],
                    "ann_id": example['ann_id'],
                    "description": desc,
                    "tokens": tokens
                    }
            results.append(result)
            with open(proc_data_path, "w+") as f:
                json.dump(results, f)
