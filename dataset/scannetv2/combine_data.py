import torch
import os
from tqdm import tqdm
import hydra
@hydra.main(version_base=None, config_path="../../config", config_name="global_imputed_config")
def main(cfg):
    if "split" not in cfg:
        raise RuntimeError("Please specify split using the +split=split argument")
    split = cfg.split
    for scene in tqdm(os.listdir(f"dataset/scannetv2/human_info_{split}")):
        non_imputed_data = torch.load(f"dataset/scannetv2/{split}/{scene}")
        human_info = torch.load(f"dataset/scannetv2/human_info_{split}/{scene}")
        imputed_data = non_imputed_data.copy()
        imputed_data['human_info'] = human_info['human_info']
        torch.save(imputed_data, f"dataset/scannetv2/{split}_imputed/{scene}")

if __name__=="__main__":
    main()
