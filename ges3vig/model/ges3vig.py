import os
import json
import clip
import torch
import hydra
import numpy as np
from tqdm import tqdm
import lightning.pytorch as pl
from ges3vig.common_ops.functions import common_ops
from ges3vig.model.vision_module.pointgroup import PointGroup
from ges3vig.model.vision_module.human_processor_early_matcher_plus_query_former import HumanProcessor
from ges3vig.model.cross_modal_module.match_module import MatchModule
from ges3vig.model.cross_modal_module.joint_module import JointModule
from ges3vig.model.vision_module.object_renderer import ObjectRenderer
from ges3vig.model.vision_module.clip_image_encoder import CLIPImageEncoder
from ges3vig.loss.human_losses import HumanLosses
import torch.nn as nn
from time import perf_counter
class Ges3Vig(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_name = cfg.data.lang_dataset

        self.train_pg = cfg.model.network.detector.train_pg
        self.train_pg_num_epochs = cfg.model.network.detector.train_pg_num_epochs

        # vision modules
        input_channel = 3 + 3 * cfg.data.point_features.use_rgb + 3 * cfg.data.point_features.use_normal + \
                        128 * cfg.data.point_features.use_multiview
        self.detector = PointGroup(
            input_channel=input_channel, output_channel=cfg.model.network.detector.output_channel,
            max_proposals=cfg.model.network.max_num_proposals, semantic_class=cfg.data.semantic_class,
            use_gt=cfg.model.network.detector.use_gt_proposal
        )
        self.human_processor = HumanProcessor(cfg.data.scene_metadata.valid_semantic_mapping,
                                in_channels = cfg.model.network.detector.output_channel,
                                out_channels = cfg.model.network.clip_word_encoder.output_channel)
        # loss
        #if self.train_pg:
            #return
        if self.dataset_name in ("ScanRefer", "Nr3D", "ImputedScanRefer", "PointRefer", "ImputeRefer"):
            self.ref_loss = hydra.utils.instantiate(
                cfg.model.loss.reference_ce_loss, chunk_size=cfg.data.chunk_size,
                max_num_proposals=cfg.model.network.max_num_proposals
            )
        elif self.dataset_name == "Multi3DRefer":
            self.ref_loss = hydra.utils.instantiate(
                cfg.model.loss.reference_bce_loss, chunk_size=cfg.data.chunk_size,
                max_num_proposals=cfg.model.network.max_num_proposals
            )
        else:
            raise NotImplementedError

        self.clip_model = clip.load(cfg.model.network.clip_model, device=self.device)[0]

        # freeze CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False

        if self.hparams.cfg.model.network.use_2d_feature:
            self.object_renderer = ObjectRenderer(**cfg.model.network.object_renderer)
            self.clip_image = CLIPImageEncoder(clip_model=self.clip_model, **cfg.model.network.clip_img_encoder)

        self.text_encoder = hydra.utils.instantiate(cfg.model.network.clip_word_encoder, clip_model=self.clip_model)

        self.joint_module = JointModule(
                feat_channel = 128,
                head = 4,
                depth = 2,
                input_channel=self.hparams.cfg.model.network.clip_img_encoder.output_channel
                )

        self.match_module = MatchModule(
            **cfg.model.network.matching_module,
            input_channel=cfg.model.network.detector.output_channel *
                          self.hparams.cfg.model.network.use_3d_features +
                          self.hparams.cfg.model.network.use_2d_feature *
                          self.hparams.cfg.model.network.clip_img_encoder.output_channel
        )

        self.contrastive_loss = hydra.utils.instantiate(cfg.model.loss.contrastive_loss)
        self.human_losses = HumanLosses()
        # evaluator
        self.evaluator = hydra.utils.instantiate(cfg.data.evaluator)
        self.val_test_step_outputs = []
        if self.hparams.cfg.model.network.adaptive_human_weight:
            self.score_det = nn.Linear(2,1)
    def forward(self, data_dict):
        output_dict = self.detector(data_dict)
        batch_size = len(data_dict["scene_id"])

        output_dict["joints"], joint_features, output_dict["human_lr"], bias_weights, has_human = self.human_processor(data_dict['point_xyz'],
                                                                                                         output_dict['point_features'],
                                                                                                         data_dict['vert_batch_ids'],
                                                                                                         output_dict['semantic_scores'],
                                                                                                         batch_size)

        human_pointer_bias = self.human_pointer_biasing(output_dict, data_dict, has_human, batch_size)
        if self.hparams.cfg.model.network.use_3d_features:
            aabb_features = output_dict["aabb_features"]
        else:
            aabb_features = torch.empty(
                size=(output_dict["aabb_features"].shape[0], 0),
                dtype=output_dict["aabb_features"].dtype, device=self.device
            )
        data_dict["lang_attention_mask"] = None
        if self.hparams.cfg.model.network.use_2d_feature:
            rendered_imgs = self.object_renderer(data_dict, output_dict)
            img_features = self.clip_image(rendered_imgs.permute(dims=(0, 3, 1, 2)))
            views = len(self.hparams.cfg.model.network.object_renderer.eye)
            aabb_img_features = torch.nn.functional.avg_pool1d(
                img_features.permute(1, 0), kernel_size=views, stride=views
            ).permute(1, 0)
            # TODO: adjust mask
           # data_dict["lang_attention_mask"] = data_dict["lang_attention_mask"][:, :77]  # CLIP context length
            # concatenate 2D and 3D features
            aabb_features = torch.nn.functional.normalize(torch.cat((aabb_features, aabb_img_features), dim=1), dim=1)

        output_dict["aabb_features"] = common_ops.convert_sparse_tensor_to_dense(
            aabb_features, output_dict["proposal_batch_offsets"],
            self.hparams.cfg.model.network.max_num_proposals
        )

        output_dict["pred_aabb_min_max_bounds"] = common_ops.convert_sparse_tensor_to_dense(
            output_dict["pred_aabb_min_max_bounds"].reshape(-1, 6), output_dict["proposal_batch_offsets"],
            self.hparams.cfg.model.network.max_num_proposals
        ).reshape(batch_size, self.hparams.cfg.model.network.max_num_proposals, 2, 3)


        self.text_encoder(data_dict, output_dict)

        #TODO maybe add a self attention block or 2 for human_query_features
        joint_features = self.joint_module(joint_features)
        wfeat_prev_shape = output_dict["word_features"].shape
        output_dict["word_features"] = torch.cat([output_dict["word_features"], joint_features], dim = 1)
        data_dict["lang_attention_mask"] = torch.zeros((output_dict["word_features"].shape[0],
                                            self.hparams.cfg.model.network.matching_module.head,
                                            self.hparams.cfg.model.network.max_num_proposals, output_dict["word_features"].shape[1]),
                                            dtype=torch.bool, device=self.device)

        data_dict["lang_attention_mask"][:,:,:,wfeat_prev_shape[1]:] = (has_human[:,None,None,None]==0)
        """
        cross-modal fusion
        """
        self.match_module(data_dict, output_dict)
        combined_pred = torch.cat([output_dict["pred_aabb_scores"][:,:, None],human_pointer_bias[:,:, None]], dim=2)

        output_dict["pred_aabb_scores"] = torch.einsum("ni,nji->nj",bias_weights,combined_pred)
        
        return output_dict

    def human_pointer_biasing(self, output_dict, data_dict, has_human, batch_size):

        human_pointing_biaser = torch.zeros((batch_size, self.hparams.cfg.model.network.max_num_proposals), dtype=torch.float32, device=self.device)

        for i in range(output_dict["proposal_batch_offsets"].shape[0] - 1):
            if has_human[i]:
                aabb_start_idx = output_dict["proposal_batch_offsets"][i]
                aabb_end_idx = output_dict["proposal_batch_offsets"][i + 1]
                bboxes = output_dict["pred_aabb_min_max_bounds"][aabb_start_idx:aabb_end_idx].mean(dim = 1)
                human_lr = nn.Softmax()(output_dict["human_lr"][i])
                human_joints = output_dict["joints"][i]
                joint_name_idx_map = data_dict["joint_name_idx_map"][i]
                left_shoulder = human_joints[joint_name_idx_map['left_shoulder']]
                right_shoulder = human_joints[joint_name_idx_map['right_shoulder']]
                left_index = human_joints[joint_name_idx_map['left_index1']]
                right_index = human_joints[joint_name_idx_map['right_index1']]

                pointing_vector_left = left_index - left_shoulder
                pointing_vector_right = right_index - right_shoulder


                obj_vector_left = bboxes - left_shoulder
                obj_vector_right = bboxes - right_shoulder

                obj_dist_left = obj_vector_left.norm(dim = 1)
                obj_dist_right= obj_vector_right.norm(dim = 1)

                dots_left = torch.einsum("ni,i->n",obj_vector_left, pointing_vector_left)
                dots_right = torch.einsum("ni,i->n",obj_vector_right, pointing_vector_right)

                pointing_biaser_left = dots_left*obj_dist_left.reciprocal()/pointing_vector_left.norm()
                pointing_biaser_right = dots_left*obj_dist_right.reciprocal()/pointing_vector_right.norm()

                pointing_biaser = human_lr[0]*pointing_biaser_left + human_lr[1]*pointing_biaser_right
#                print(f"aabb_start_idx,aabb_end_idx: {aabb_start_idx,aabb_end_idx}")
#                print(f"pointing_biaser: {pointing_biaser.shape}")
#                print(f"human_pointing_biaser[i][aabb_start_idx:aabb_end_idx]: {human_pointing_biaser[i][aabb_start_idx:aabb_end_idx].shape}")
                human_pointing_biaser[i][:aabb_end_idx-aabb_start_idx] = pointing_biaser
        return human_pointing_biaser

    def concat_human_feats(self, aabb_feats, batch_proposal_offsets, human_feats):
        new_feats = torch.zeros((aabb_feats.shape[0], aabb_feats.shape[1]+  self.hparams.cfg.model.network.use_human_early_num_feats), device = self.device)
        for i in range(len(batch_proposal_offsets)-1):
                start = batch_proposal_offsets[i]
                end = batch_proposal_offsets[i+1]
                h_feats = human_feats[i].flatten()[None].repeat(end-start,1)
                new_feats[start:end] += torch.cat([aabb_feats[start:end], h_feats], dim=1)
        return new_feats




    def _loss(self, data_dict, output_dict):
        loss_dict = self.detector.loss(data_dict, output_dict)
        # reference loss
        self.human_losses(data_dict, output_dict, loss_dict)
        #print(f"loss_dict['human_joint_loss']: {loss_dict['human_joint_loss']}")
        loss_dict["reference_loss"] = self.ref_loss(
            output_dict,
            output_dict["pred_aabb_min_max_bounds"],
            output_dict["pred_aabb_scores"],
            data_dict["gt_aabb_min_max_bounds"],
            data_dict["gt_target_obj_id_mask"].permute(dims=(1, 0)),
            data_dict["aabb_count_offsets"],
        )

        if self.hparams.cfg.model.network.use_contrastive_loss:
            # contrastive loss
            loss_dict["contrastive_loss"] = self.contrastive_loss(
                output_dict["aabb_features_inter"],
                output_dict["sentence_features"],
                output_dict["gt_labels"]
            )
        return loss_dict

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.hparams.cfg.model.optimizer, params=self.parameters())
        return optimizer

    def training_step(self, data_dict, idx):
        #x = torch.ones(5, requires_grad = True)
        #return x.sum()
        if self.current_epoch>self.train_pg_num_epochs:
            self.train_pg = False
        if self.train_pg:
            #print("trainin_pg")
            batch_size = len(data_dict["scene_id"])
            output_dict = self.detector(data_dict)

            #if self.hparams.cfg.model.network.use_human_pointer_scores and self.hparams.cfg.model.network.detector.train_human:
                #print("getting_human")

            #    output_dict["pred_aabb_min_max_bounds"] = common_ops.convert_sparse_tensor_to_dense(
            #        output_dict["pred_aabb_min_max_bounds"].reshape(-1, 6), output_dict["proposal_batch_offsets"],
            #        self.hparams.cfg.model.network.max_num_proposals
            #    ).reshape(batch_size, self.hparams.cfg.model.network.max_num_proposals, 2, 3)

             #   output_dict["human_inference_scores"] = common_ops.convert_sparse_tensor_to_dense(
            #        output_dict["human_inference_scores"], output_dict["proposal_batch_offsets"],
            #        self.hparams.cfg.model.network.max_num_proposals
                #)

           #     output_dict["pred_aabb_scores"]=self.hparams.cfg.model.network.human_pointer_scale*output_dict["human_inference_scores"]
           #     loss_dict = self.detector.loss(data_dict, output_dict)


           #     loss_dict["reference_loss"] = self.ref_loss(
           #         output_dict,
           #         output_dict["pred_aabb_min_max_bounds"],
           #         output_dict["pred_aabb_scores"],
           #         data_dict["gt_aabb_min_max_bounds"],
           #         data_dict["gt_target_obj_id_mask"].permute(dims=(1, 0)),
           #         data_dict["aabb_count_offsets"],
           #     )
           # else:
                #print("not getting human")
            loss_dict = self.detector.loss(data_dict, output_dict)
        else:
            #print(f"\n\n\n\ntraining Full\n\n\n\n")
            output_dict = self(data_dict)
            loss_dict = self._loss(data_dict, output_dict)

        # calculate the total loss and log
        total_loss = 0
        for loss_name, loss_value in loss_dict.items():
            total_loss += loss_value
            self.log(f"train_loss/{loss_name}", loss_value, on_step=True, on_epoch=False)
        self.log(f"train_loss/total_loss", total_loss, on_step=True, on_epoch=False)
        return total_loss

    def validation_step(self, data_dict, idx):
        #x = torch.ones(5, requires_grad = True)
        #return x.sum()
        output_dict = self(data_dict)
        loss_dict = self._loss(data_dict, output_dict)
        # calculate the total loss and log
        total_loss = 0
        for loss_name, loss_value in loss_dict.items():
            total_loss += loss_value
            self.log(f"val_loss/{loss_name}", loss_value, on_step=True, on_epoch=False)
        self.log(f"val_loss/total_loss", total_loss, on_step=True, on_epoch=False)

        # get predictions and gts
        self.val_test_step_outputs.append((self._parse_pred_results(data_dict, output_dict), self._parse_gt(data_dict)))

    def test_step(self, data_dict, idx):
        output_dict = self(data_dict)
        self.val_test_step_outputs.append(
            (self._parse_pred_results(data_dict, output_dict), self._parse_gt(data_dict))
        )

    def on_validation_epoch_end(self):
        total_pred_results = {}
        total_gt_results = {}
        for pred_results, gt_results in self.val_test_step_outputs:
            total_pred_results.update(pred_results)
            total_gt_results.update(gt_results)
        self.val_test_step_outputs.clear()
        self.evaluator.set_ground_truths(total_gt_results)
        results = self.evaluator.evaluate(total_pred_results)

        # log
        for metric_name, result in results.items():
            for breakdown, value in result.items():
                self.log(f"val_eval/{metric_name}_{breakdown}", value)

    def on_test_epoch_end(self):
        total_pred_results = {}
        total_gt_results = {}
        for pred_results, gt_results in self.val_test_step_outputs:
            total_pred_results.update(pred_results)
            total_gt_results.update(gt_results)
        self.val_test_step_outputs.clear()
        self._save_predictions(total_pred_results)

    def _parse_pred_results(self, data_dict, output_dict):
        batch_size, lang_chunk_size = data_dict["ann_id"].shape
        if self.dataset_name in ("ScanRefer", "Nr3D", "ImputedScanRefer", "PointRefer", "ImputeRefer"):
            pred_aabb_score_masks = (output_dict["pred_aabb_scores"].argmax(dim=1)).reshape(
                shape=(batch_size, lang_chunk_size, -1)
            )
        elif self.dataset_name == "Multi3DRefer":
            pred_aabb_score_masks = (
                    torch.sigmoid(output_dict["pred_aabb_scores"]) >= self.hparams.cfg.model.inference.output_threshold
            ).reshape(shape=(batch_size, lang_chunk_size, -1))
        else:
            raise NotImplementedError

        pred_results = {}
        for i in range(batch_size):
            for j in range(lang_chunk_size):
                pred_aabbs = output_dict["pred_aabb_min_max_bounds"][i][pred_aabb_score_masks[i, j]]
                pred_results[
                    (data_dict["scene_id"][i], data_dict["object_id"][i][j].item(),
                     data_dict["ann_id"][i][j].item())
                ] = {
                    "aabb_bound": (pred_aabbs + data_dict["scene_center_xyz"][i]).cpu().numpy()
                }
        return pred_results

    def _parse_gt(self, data_dict):
        batch_size, lang_chunk_size = data_dict["ann_id"].shape
        gts = {}
        gt_target_obj_id_masks = data_dict["gt_target_obj_id_mask"].permute(1, 0)
        for i in range(batch_size):
            aabb_start_idx = data_dict["aabb_count_offsets"][i]
            aabb_end_idx = data_dict["aabb_count_offsets"][i + 1]
            for j in range(lang_chunk_size):
                gts[
                    (data_dict["scene_id"][i], data_dict["object_id"][i][j].item(),
                     data_dict["ann_id"][i][j].item())
                ] = {
                    "aabb_bound":
                        (data_dict["gt_aabb_min_max_bounds"][aabb_start_idx:aabb_end_idx][gt_target_obj_id_masks[j]
                    [aabb_start_idx:aabb_end_idx]] + data_dict["scene_center_xyz"][i]).cpu().numpy(),
                    "eval_type": data_dict["eval_type"][i][j]
                }
        return gts

    def _save_predictions(self, predictions):
        scene_pred = {}
        for key, value in predictions.items():
            scene_id = key[0]
            if key[0] not in scene_pred:
                scene_pred[scene_id] = []
            corners = np.empty(shape=(value["aabb_bound"].shape[0], 8, 3), dtype=np.float32)
            for i, aabb in enumerate(value["aabb_bound"]):
                min_point = aabb[0]
                max_point = aabb[1]
                corners[i] = np.array([
                    [x, y, z]
                    for x in [min_point[0], max_point[0]]
                    for y in [min_point[1], max_point[1]]
                    for z in [min_point[2], max_point[2]]
                ], dtype=np.float32)

            if self.dataset_name in ("ScanRefer", "Nr3D", "ImputedScanRefer", "PointRefer", "ImputeRefer"):
                scene_pred[scene_id].append({
                    "object_id": key[1],
                    "ann_id": key[2],
                    "aabb": corners.tolist()
                })
            elif self.dataset_name == "Multi3DRefer":
                scene_pred[scene_id].append({
                    "ann_id": key[2],
                    "aabb": corners.tolist()
                })
        prediction_output_root_path = os.path.join(
            self.hparams.cfg.pred_path, self.hparams.cfg.data.inference.split
        )
        os.makedirs(prediction_output_root_path, exist_ok=True)
        for scene_id in tqdm(scene_pred.keys(), desc="Saving predictions"):
            with open(os.path.join(prediction_output_root_path, f"{scene_id}.json"), "w") as f:
                json.dump(scene_pred[scene_id], f, indent=2)
        self.print(f"==> Complete. Saved at: {os.path.abspath(prediction_output_root_path)}")
