# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
from collections import defaultdict

import numpy as np
import torch,os, time
import torch.nn as nn
from tqdm import tqdm

from lab4d.engine.train_utils import get_local_rank
from lab4d.nnutils.intrinsics import IntrinsicsMLP
from lab4d.nnutils.multifields import MultiFields
from lab4d.utils.geom_utils import K2inv, K2mat
from lab4d.utils.numpy_utils import interp_wt
from lab4d.utils.render_utils import render_pixel
from lab4d.utils.camera_utils import get_object_to_camera_matrix, se3_to_quaternion_translation
import torchvision.utils as vutils
import threestudio
from threestudio.data.uncond import RandomCameraIterableDataset,RandomCameraDataModuleConfig
from threestudio.data.random_multiview import RandomMultiviewCameraIterableDataset, RandomMultiviewCameraDataModuleConfig
from lab4d.utils.quat_transform import dual_quaternion_to_quaternion_translation

VRAM_DBG = False

class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x, grad):
        ctx.grad = grad
        return torch.mean(x)  ## just some random value

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def backward(ctx, *grad_outputs):
        return ctx.grad, None

straight_through_estimator = StraightThroughEstimator.apply




class dvr_model(nn.Module):
    """A model that contains a collection of static/deformable neural fields

    Args:
        config (Dict): Command-line args
        data_info (Dict): Dataset metadata from get_data_info()
    """

    def __init__(self, config, data_info):
        super().__init__()
        self.config = config
        self.round_count = 0
        self.total_iters = 0
        self.device = get_local_rank()
        self.data_info = data_info

        self.fields = MultiFields(
            data_info=data_info,
            field_type=config["field_type"],
            fg_motion=config["fg_motion"],
            num_inst=1
            if config["single_inst"]
            else len(data_info["frame_info"]["frame_offset"]) - 1,
            opts=config,
        )
        self.intrinsics = IntrinsicsMLP(
            self.data_info["intrinsics"],
            frame_info=self.data_info["frame_info"],
            num_freq_t=0,
            D=1 if "hash" in config["fg_motion"] else 5,
        )
        if self.config["gen3d_wt"] > 0:
            self.gen3d_res = self.config['gen3d_res']
            self.gen3d_freq = self.config['gen3d_freq']

            prompt = self.config['prompt'].replace('_', ' ')
            print("---- gen3d_prompt ----",prompt)
            assert "_" not in prompt and not prompt == 'A'
            if self.config['gen3d_guidance'] == 'if':
                cfg = {
                    'prompt_processor_type': 'deep-floyd-prompt-processor',
                    'prompt_processor': {
                        'prompt': prompt,
                    },
                    'guidance_type': 'deep-floyd-guidance',
                    'guidance': {
                        'half_precision_weights': True,
                        'guidance_scale': self.config['gs']*1.0,
                        'pretrained_model_name_or_path': 'DeepFloyd/IF-I-XL-v1.0',
                        'grad_clip': None,
                        "view_dependent_prompting": True,
                    },
                    'data': {
                        'batch_size': 1,
                        'width': self.gen3d_res,
                        'height': self.gen3d_res,
                        'camera_distance_range': [1.5, 2],
                        'fovy_range': [40, 70],
                        'elevation_range': [-10, 40],
                        'light_sample_strategy': "dreamfusion",
                        'eval_camera_distance': 2.0,
                        'eval_fovy_deg': 70.,
                        
                        # 'elevation_range': [0,0],
                        # 'azimuth_range':[0, 0],
                    },
                    # 'data': {
                    #     'batch_size': [4,4],
                    #     'relative_radius': True,
                    #     'n_view': 4,
                    #     'zoom_range': [1.0, 1.0],
                    #     'resolution_milestones': [0],
                    #     'width': [self.gen3d_res, self.gen3d_res],
                    #     'height': [self.gen3d_res, self.gen3d_res],
                    #     'camera_perturb': 0.,
                    #     'center_perturb': 0.,
                    #     'up_perturb': 0.,
                    #     # 'camera_distance_range': [1, 1.5], #[1, 1.5 too large]
                    #     'fovy_range': [40, 70],
                    #     'elevation_range': [-10, 40],

                    #     # 'elevation_range': [0,0],
                    #     # 'azimuth_range':[0, 0],

                    #     'eval_camera_distance': 2.0,
                    #     'eval_fovy_deg': 70.,
                    # },
                }
                self.data_cfg = RandomCameraDataModuleConfig(**cfg['data'])
                self.gen3d_dataloader = RandomCameraIterableDataset(self.data_cfg)
                mvd_data = {
                            'batch_size': [4,4],
                            'relative_radius': True,
                            'n_view': 4,
                            'zoom_range': [1.0, 1.0],
                            'resolution_milestones': [0],
                            'width': [self.gen3d_res, self.gen3d_res],
                            'height': [self.gen3d_res, self.gen3d_res],
                            'camera_perturb': 0.,
                            'center_perturb': 0.,
                            'up_perturb': 0.,
                            # 'camera_distance_range': [1, 1.5], #[1, 1.5 too large]
                            'fovy_range': [40, 70],
                            'elevation_range': [-10, 40],

                            # 'elevation_range': [0,0],
                            # 'azimuth_range':[0, 0],

                            'eval_camera_distance': 2.0,
                            'eval_fovy_deg': 70.,
                            }
                self.mvddata_cfg = RandomMultiviewCameraDataModuleConfig(**mvd_data)
                self.gen3d_dataloader = RandomMultiviewCameraIterableDataset(self.mvddata_cfg)

                # from threestudio.data.random_multiview import RandomMultiviewCameraIterableDataset, RandomMultiviewCameraDataModuleConfig
                # self.data_cfg = RandomMultiviewCameraDataModuleConfig(**cfg['data'])
                # self.gen3d_dataloader = RandomMultiviewCameraIterableDataset(self.data_cfg)
                # print("borrrowing dataset from mvd!!!!!!!")
                self.guidance = threestudio.find(cfg['guidance_type'])(cfg['guidance'])

            elif self.config['gen3d_guidance'] == 'mvd':
                negative_prompt = "" if self.config['no_negative_prompt'] else "ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions"
                cfg = {
                    'prompt_processor_type': "stable-diffusion-prompt-processor",
                    'prompt_processor': {
                        'pretrained_model_name_or_path': "stabilityai/stable-diffusion-2-1-base",
                        'prompt': prompt,
                        'negative_prompt': negative_prompt,
                        'front_threshold': 30.,
                        'back_threshold': 30.,
                    },
                    'guidance_type': "multiview-diffusion-guidance",
                    'guidance': {
                        'model_name': "sd-v2.1-base-4view",
                        'ckpt_path': None,
                        'guidance_scale': self.config['gs']*1.0,
                        'min_step_percent': 0.02,   # dummy!
                        'max_step_percent': 0.50,   # dummy!
                        'recon_loss': True,
                        'recon_std_rescale': 0.5,
                        'view_dependent_prompting': False,
                    },

                    'data': {
                        'batch_size': [4,4],
                        'relative_radius': True,
                        'n_view': 4,
                        'zoom_range': [1.0, 1.0],
                        'resolution_milestones': [0],
                        'width': [self.gen3d_res, self.gen3d_res],
                        'height': [self.gen3d_res, self.gen3d_res],
                        'camera_perturb': 0.,
                        'center_perturb': 0.,
                        'up_perturb': 0.,
                        # 'camera_distance_range': [1, 1.5], #[1, 1.5 too large]
                        'fovy_range': [40, 70],
                        'elevation_range': [-10, 40],

                        # 'elevation_range': [0,0],
                        # 'azimuth_range':[0, 0],

                        'eval_camera_distance': 2.0,
                        'eval_fovy_deg': 70.,
                    },
                }
                self.data_cfg = RandomMultiviewCameraDataModuleConfig(**cfg['data'])
                self.gen3d_dataloader = RandomMultiviewCameraIterableDataset(self.data_cfg)
                

                self.guidance = threestudio.find(cfg['guidance_type'])(cfg['guidance'])
            

            assert self.config['gen3d_sds_t_max'] >= 0.5 and self.config['gen3d_sds_t_max'] < 1

            if self.config["new_ts"]:
                final_point = self.config["num_rounds"] * self.config["iters_per_round"] * 0.8
            else:
                final_point = 8000
            
            self.min_step_schedule = [0.0, self.config['gen3d_sds_t_max'], 0.02, final_point]
            self.max_step_schedule = [0.0, self.config['gen3d_sds_t_max'], 0.50, final_point]

            self.prompt_processor = threestudio.find(cfg['prompt_processor_type'])(cfg['prompt_processor'])
            self.prompt_processor.configure_text_encoder()
    
    def set_iters(self, round_count, total_iters):
        self.round_count = round_count
        self.total_iters = total_iters

    def mlp_init(self):
        """Initialize camera transforms, geometry, articulations, and camera
        intrinsics for all neural fields from external priors
        """
        self.fields.mlp_init()
        self.intrinsics.mlp_init()

    def forward(self, batch, log_dict=None):
        """Run forward pass and compute losses

        Args:
            batch (Dict): Batch of dataloader samples. Keys: "rgb" (M,2,N,3),
                "mask" (M,2,N,1), "depth" (M,2,N,1), "feature" (M,2,N,16),
                "flow" (M,2,N,2), "flow_uct" (M,2,N,1), "vis2d" (M,2,N,1),
                "crop2raw" (M,2,4), "dataid" (M,2), "frameid_sub" (M,2),
                "hxy" (M,2,N,3), and "is_detected" (M,2)
        Returns:
            loss_dict (Dict): Computed losses. Keys: "mask" (M,N,1),
                "rgb" (M,N,3), "depth" (M,N,1), "flow" (M,N,1), "vis" (M,N,1),
                "feature" (M,N,1), "feat_reproj" (M,N,1),
                "reg_gauss_mask" (M,N,1), "reg_visibility" (0,),
                "reg_eikonal" (0,), "reg_deform_cyc" (0,),
                "reg_soft_deform" (0,), "reg_gauss_skin" (0,),
                "reg_cam_prior" (0,), and "reg_skel_prior" (0,).
        """

        config = self.config
        self.process_frameid(batch) # frameid local to global

        # # watch some view when training
        # if self.total_iters % 50 == 0 and 'cat-pikachu' in self.config['seqname']:
        #     with torch.no_grad():
                
        #         eval_batch = torch.load('eval_batch.pth')
        #         # import ipdb; ipdb.set_trace()
        #         for k in eval_batch.keys():
        #             eval_batch[k] = eval_batch[k][14:16, ...]
        #         # eval_batch['frameid_sub']=torch.tensor([340,342],device='cuda:0')
        #         angle=90
        #         elev=0
        #         cam_rot = get_object_to_camera_matrix(angle, [0, 1, 0], 0)
        #         cam_elev = get_object_to_camera_matrix(elev, [1, 0, 0], 0)[None]
        #         # cam_rot = get_object_to_camera_matrix(20, [0,1,0], 0)
        #         cam_3d_gen = np.zeros((1, 4, 4)) * 1.0
        #         cam_3d_gen[0,3,3]=1
        #         cam_3d_gen[0,:3,:3]=np.array([[1,0,0],[0,1,0],[0,0,1]])
        #         cam_3d_gen[0,:3,3]=np.array([0,0,2])
        #         cam_3d_gen = cam_3d_gen @ cam_rot

        #         field2cam = {}
        #         field2cam["fg"] = cam_3d_gen # FIXME: any problem set 'fg'?
        #         for k, v in field2cam.items():
        #             field2cam[k] = torch.tensor(v, dtype=torch.float32, device=self.device)
        #             field2cam[k] = se3_to_quaternion_translation(field2cam[k], tuple=False)

        #         eval_batch["field2cam"] = field2cam #quaternion # 如果有则不从mlp获取
        #         new_view_results = self.evaluate(eval_batch, is_pair=True)
        #         new_view_rendered = new_view_results['rgb'].reshape(64,64,3) # HWC
        #         vutils.save_image(new_view_rendered.permute(2,0,1), f'look_gradinfo/{self.config["logname"]}_rgb_eval.png')
        #         # import ipdb; ipdb.set_trace()

        #         if self.config['render_uncert']:
        #             new_view_rendered = new_view_results['entropy'].reshape(64,64,1)

        #             vutils.save_image(new_view_rendered.permute(2,0,1), f'look_gradinfo/{self.config["logname"]}_entropy_eval.png')

        if batch["hxy"].shape[0] > 1:
            self.reshape_batch(batch) # merge pair dimension into batch dimension    hxy: [2,128,16,3] -> [256,16,3]  128:img samples per it,  16:pixel samples per img
        if self.config["gen3d_wt"] > 0 and batch["hxy"].shape[0] == 1: # 2 means batchsize=1 , means gen3d_dataloader hxy: [1, 2, 256, 256, 3])
            
            if self.config["ablation_lockframeid"] > 0:
                batch["frameid"] = torch.tensor([self.config["lock_frameid"]], device=self.device).repeat(batch["frameid"].shape[0])
                batch['frameid_sub'] = (batch['frameid'] - self.data_info["frame_info"]["frame_offset_raw"][batch["dataid"]]).to(self.device)

            self.guidance.update_step(self.round_count / self.config['num_rounds'], self.total_iters)
            self.gen3d_dataloader.update_step(self.round_count / self.config['num_rounds'], self.total_iters)

            if self.config['gen3d_guidance'] == 'mvd':
                self.guidance.interp_value(self.min_step_schedule, self.max_step_schedule, self.round_count / self.config['num_rounds'], self.total_iters)
            # -----------------------------------------------
            batch["is_gen3d"] = torch.tensor([True]).repeat(batch["frameid"].shape[0])
            aabb = self.fields.get_aabb()
            obj_size = (aabb["fg"][1, :] - aabb["fg"][0, :]).max().detach().cpu().numpy()

            cam_3d_gen_list = []
            # ---------------new cam_3d_gen--------------------
            # self.gen3d_dataloader.camera_distance_range = [0.5 * self.config['gen3d_dist'], 1.5 * self.config['gen3d_dist']]
            self.gen3d_dataloader.camera_distance_range = [0.42 * self.config['gen3d_dist'] * obj_size, 0.7 * self.config['gen3d_dist'] * obj_size]
            

            threestudio_batch = self.gen3d_dataloader.collate(None)

            if self.config['gen3d_guidance'] == 'if':
                batch_size = 1

                idx = np.random.randint(0, 4)
                for k,v in threestudio_batch.items():
                    if k in ['height', 'width']:
                        continue
                    threestudio_batch[k] = v[idx][None]
                # import ipdb; ipdb.set_trace()
            else:
                batch_size = threestudio_batch['rays_o'].shape[0]
            # print("angle:", threestudio_batch['azimuth'].cpu().numpy(), "elev:", threestudio_batch['elevation'].cpu().numpy())
            # print("threestudio c2w:", threestudio_batch['c2w'].cpu().numpy())
            # import ipdb; ipdb.set_trace()
            for i in range(batch_size):
                ts=threestudio_batch['c2w'].cpu().numpy()[i]
                c2wp=ts[[1,2,0,3],:]*np.array([[1],[-1],[-1],[1]])
                wp2c=np.linalg.inv(c2wp)
                wp2cp=wp2c*np.array([[1],[-1],[-1],[1]])
                cam_3d_gen_list.append(wp2cp[None,...])
            # import ipdb; ipdb.set_trace()
                # print("threestudio wp2cp:", wp2cp)
            # cam_3d_gen = threestudio_batch['c2w']  # 4 4
            # print(cam_3d_gen)
            # cam_3d_gen = torch.inverse(cam_3d_gen)
            Kmat = torch.zeros(1, 4, dtype=torch.float32, device=self.device)
            Kmat[:, 0] = threestudio_batch["fx"][0, 0]
            Kmat[:, 1] = threestudio_batch["fy"][0, 0]
            Kmat[:, 2] = threestudio_batch['width'] / 2
            Kmat[:, 3] = threestudio_batch['height'] / 2
            batch['Kinv'] = K2inv(Kmat)


            # --------------old cam_3d_gen---------------------
            
            if self.config["gen3d_guidance"] == 'if' and False :
                if False:

                    for k, v in threestudio_batch.items():
                        if k in ['height', 'width']:
                            continue
                        random_idx = np.random.randint(0, 4)
                        threestudio_batch[k] = v[random_idx:random_idx+1]
                else:
                    cam_3d_gen_list = []
                    axis = [0, 1, 0]
                    # print("angle:", threestudio_batch['azimuth'].cpu().numpy(), "elev:", threestudio_batch['elevation'].cpu().numpy())
                    elevs = threestudio_batch['elevation'].cpu().numpy()
                    angles = threestudio_batch['azimuth'].cpu().numpy()
                    # 
                    for i in range(1):
                        elev = elevs[i:i+1]
                        angle = angles[i:i+1]

                        cam_rot = get_object_to_camera_matrix(angle , axis, 0)
                        cam_elev = get_object_to_camera_matrix(elev , [1, 0, 0], 0)[None]

                        # field2cam_fr = self.fields.get_cameras(inst_id=batch["inst_id"])
                        quat, trans = self.fields.field_params.fg.camera_mlp.get_vals(frame_id=batch["frameid"])
                        trans = trans / self.fields.field_params.fg.logscale.exp()
                        field2cam_fr = quaternion_translation_to_se3(quat, trans).detach().cpu().numpy()

                        # dbg!!!!!!!!!!!!
                        field2cam_fr[0,:3,:3]=np.array([[1,0,0],[0,1,0],[0,0,1]])
                        field2cam_fr[0,:3,3]=np.array([0,0,2 * self.config["gen3d_dist"] ]) 

                        cam_3d_gen = field2cam_fr @ cam_rot @ cam_elev  # 4 4
                        cam_3d_gen_list.append(cam_3d_gen)
                        # print("cam_3d_gen:\n", cam_3d_gen)
                        
                        # cam_3d_gen[0,:3,:3]=np.array([[1,0,0],[0,1,0],[0,0,1]])
                        # cam_3d_gen[0,:3,3]=np.array([0,0,2])
                        del batch['Kinv']
            # ------------------------------------------

            
            field2cam = {}
            field2cam["fg"] = np.concatenate(cam_3d_gen_list, axis=0) # FIXME: any problem set 'fg'?
            for k, v in field2cam.items():
                field2cam[k] = torch.tensor(v, dtype=torch.float32, device=self.device)
                field2cam[k] = se3_to_quaternion_translation(field2cam[k], tuple=False)

            batch["field2cam"] = field2cam #quaternion # 如果有则不从mlp获取

            # Kmat = self.intrinsics.get_vals(batch["frameid"]) # intrinsics (focal is from mlp.forward(frameid))
            # batch['Kinv'] = K2inv(Kmat) @ K2mat(batch["crop2raw"])# 如果有则不从mlp获取+套用crop2raw

            # -----------------------------------------
            batch["crop2raw"][..., :2] *= self.train_res / self.gen3d_res

            # -----------------------------------------

            eval_range = torch.arange(self.gen3d_res, dtype=torch.float32, device=self.device)
            hxy = torch.cartesian_prod(eval_range, eval_range)
            hxy = torch.stack([hxy[:, 1], hxy[:, 0], torch.ones_like(hxy[:, 0])], -1)
            hxy = hxy[None, ...].repeat(batch["frameid"].shape[0], 1, 1)
            batch["hxy"] = hxy
            
            keys_to_delete = ['rgb', 'mask', 'depth', 'feature', 'flow', 'flow_uct', 'vis2d']
            for key in keys_to_delete:
                del batch[key]

            if self.config["gen3d_frameid"] >= 0:
                batch["frameid"] = torch.tensor([self.config["gen3d_frameid"]], device=self.device).repeat(batch["frameid"].shape[0])
                batch['frameid_sub'] = (batch['frameid'] - self.data_info["frame_info"]["frame_offset_raw"][batch["dataid"]]).to(self.device)
            
            # keys: 'crop2raw', 'is_detected', 'dataid', 'frameid_sub', 'hxy', 'frameid', 'is_gen3d', 'field2cam'
            # ----------------------------------------
            # print("sds frameid: ", batch["frameid"])

            for k, v in batch.items():
                if k == 'field2cam':
                    continue
                else:        
                    # import ipdb; ipdb.set_trace()
                    if len(v.shape) == 1:
                        batch[k] = v.repeat(batch_size, 1)
                    elif len(v.shape) == 2:
                        batch[k] = v.repeat(batch_size, 1)
                    else:
                        batch[k] = v.repeat(batch_size, 1, 1) # repeat batch_size times
                    if batch[k].shape[-1] == 1:
                        batch[k] = batch[k].squeeze(-1)
        

            if VRAM_DBG:
                import ipdb; ipdb.set_trace()
                import time 
                start_time = time.time()
            
            if self.config["save_vram"] or (self.config["gen3d_guidance"] == 'mvd' and self.config["gen3d_res"] > 88):
                with torch.no_grad():
                    new_view_results = self.render(batch)

            else:
                new_view_results = self.render(batch) # flow_thresh=config["train_res"]
            if VRAM_DBG:
                print("res", self.config["gen3d_res"], "render time:", time.time() - start_time)
                import ipdb; ipdb.set_trace()
            new_view_rendered, collage = process_results(self, new_view_results, batch_size)
            # import ipdb; ipdb.set_trace()
            # for i in range(batch_size):           
            #     vutils.save_image(new_view_rendered[i].permute(2,0,1), f'look_gradinfo/{self.config["logname"]}_rgb{i}_train.png')
            if self.config['render_uncert']:
                new_view_rendered = new_view_results["rendered"]['entropy'].reshape(self.gen3d_res,self.gen3d_res,1)
                vutils.save_image(new_view_rendered.permute(2,0,1), f'look_gradinfo/{self.config["seqname"]}{self.config["logname"]}_entropy_train.png')

            if VRAM_DBG:
                import ipdb; ipdb.set_trace()
                import time 
                start_time = time.time()
            if self.config["save_vram"] or (self.config["gen3d_guidance"] == 'mvd' and self.config["gen3d_res"] > 88):
                new_view_rendered_clone = new_view_rendered.detach().clone()
                new_view_rendered_clone.requires_grad = True
                loss_3d_gen = self.guidance(new_view_rendered_clone, self.prompt_processor(), **threestudio_batch, rgb_as_latents=False)['loss_sds']
                loss_3d_gen.backward()

                pred_grad = new_view_rendered_clone.grad.data
                pred_grad = pred_grad.reshape(batch_size, -1,3)

                patch_size = (self.gen3d_res * self.gen3d_res) // 8
                try:
                    assert (self.gen3d_res * self.gen3d_res) % patch_size == 0
                except:
                    import ipdb; ipdb.set_trace()
                num_patch = (self.gen3d_res * self.gen3d_res) // patch_size
                # 
                for patch_idx in range(num_patch):
                    # print("patch_idx", patch_idx, "num_patch", num_patch)
                    batch_chunk = defaultdict(list)
                    for k, v in batch.items():
                        if k == "hxy":
                            batch_chunk[k] = v[
                                :, patch_size * patch_idx : patch_size * (patch_idx + 1)
                            ]
                        else:
                            batch_chunk[k] = v

                    this_pred = self.render(batch_chunk)
                    this_pred_rgb = this_pred['rendered']['rgb']
                    this_pred_mask = this_pred['rendered']['mask']
                    assert self.config['w_bg']
                    this_pred_rgb = this_pred_rgb * this_pred_mask + (1 - this_pred_mask)

                    this_loss = straight_through_estimator(this_pred_rgb, pred_grad[:, patch_size * patch_idx : patch_size * (patch_idx + 1)] * self.config['gen3d_wt'])
                    # this_loss = 0.5 * F.mse_loss(
                    #     this_pred, this_pred.detach() - pred_grad[:, v : v + 1]
                    # )
                    # this_pred_rgb.retain_grad()
                    this_loss.backward()#retain_graph=True)
                
                placeholder_loss = torch.tensor(1.0, requires_grad=True)
                loss_dict={"gen3d": placeholder_loss}

            else:


                loss_3d_gen = self.guidance(new_view_rendered, self.prompt_processor(), **threestudio_batch, rgb_as_latents=False)['loss_sds']


                # print("loss_3d_gen", loss_3d_gen)
                loss_dict={"gen3d": loss_3d_gen}
                if self.config['gen3d_jacobloss'] or self.config['gen3d_cycloss']:
                    self.compute_reg_loss(loss_dict, new_view_results)
                self.apply_loss_weights(loss_dict, config)

            results = new_view_results

            if VRAM_DBG:
                print("SDS time:", time.time() - start_time)
                import ipdb; ipdb.set_trace()
            # ------log------

            if False:
                print("rounds", self.round_count, "iters", self.total_iters, "objsize", obj_size, "cam_dis", threestudio_batch["camera_distances"][0])
                print("angle", threestudio_batch['azimuth'][0], "elev", threestudio_batch['elevation'][0], "frameid", batch['frameid'][0])
            
            
            view_logger = {"angle": threestudio_batch['azimuth'][0], "elev": threestudio_batch['elevation'][0], "obj_size": obj_size, "camera_distances": threestudio_batch["camera_distances"][0]}
            # torch.save(view_logger, f"wxz_tmp/view_logger_{self.config['seqname']}{self.config['logname']}.pt")
            if self.total_iters % 200 == 0 or (self.config["debug"] and self.total_iters % 50 == 0):
                if not os.path.exists(f'wxz_tmp/{self.config["seqname"]}-{self.config["logname"]}'):
                    os.mkdir(f'wxz_tmp/{self.config["seqname"]}-{self.config["logname"]}')
                vutils.save_image(collage.permute(2,0,1), f'wxz_tmp/{self.config["seqname"]}-{self.config["logname"]}/{self.total_iters}_dist{threestudio_batch["camera_distances"][0]}.png')

            if self.total_iters % 50 == 0 :
                pass
                # print("objsize", obj_size, "cam_dis", threestudio_batch["camera_distances"][0])
                # torch.save(new_view_results, f"wxz_tmp/rendered_{self.config['seqname']}{self.config['logname']}.pt")
            if self.total_iters % 100 == 0:
                with torch.no_grad():
                    cano_view_results = self.render(batch, no_warp=True) # flow_thresh=config["train_res"]
                    cano_view_rendered, collage = process_results(self, cano_view_results, batch_size, suffix="cano")
        else:
            # if have batch["field2cam"] , transform it from quat to mat, or get it from mlp by frameid
            results = self.render(batch, flow_thresh=config["train_res"])
            loss_dict = self.compute_loss(batch, results)
        
        if self.config['dreamo_wt'] > 0 or self.config["skel_preserve_wt"] > 0:

            aux_dict = results["aux_dict"]['fg']
            bones_qr = aux_dict["bones_qr"]
            bones_trans = aux_dict["bones_trans"]
            bones_sdf = aux_dict["bones_sdf"]
            assert bones_sdf.shape[1] == 25
            if self.config['dreamo_wt'] > 0:
                loss_dict["surf_loss"] = 0.1 * torch.max(bones_sdf-0.01, torch.zeros_like(bones_sdf)).norm(dim=1).mean() * self.config['dreamo_wt']  # 0.1
            if False and batch['frameid_sub'].shape[0] > 0 and batch['frameid_sub'][0] != batch['frameid_sub'][1] : # when batch has multipy frames
                # loss_dict["smooth_loss"] = 2 * torch.arccos(torch.abs(torch.dot(bones_qr[0], bones_qr[1]))).mean() 
                # bones_qr[0] /= bones_qr[0].norm(dim=1, keepdim=True)
                # bones_qr[1] /= bones_qr[1].norm(dim=1, keepdim=True)
                dot_product = (bones_qr[::2] * bones_qr[1::2]).sum(dim=2)
                dot_product[dot_product > 1] = 0.999
                arccos_mean = torch.acos(dot_product).mean() # 0.1
                trans_diff_mean = (bones_trans[::2] - bones_trans[1::2]).norm(dim=-1).mean() # 0.01
                loss_dict["smooth_loss"] = 0.0001 * (arccos_mean * 0.1 + trans_diff_mean) * self.config['dreamo_wt']
            if self.config["skel_preserve_wt"] > 0:
                import ipdb; ipdb.set_trace()
                # Not finish yet
                # TODO: 应该是先正规空间生成skel，然后遍历所有帧，找出①骨骼长度变化范围，②关节相对骨骼的旋转角度变化范围
                if self.total_iters == 0:
                    self.skel, self.rest_articulation = self.fields.generate_fg_skel()

                    self.skel_qr = bones_qr
                    self.skel_trans = bones_trans
                else:
                    loss_trans = torch.zeros(1, device=self.device)
                    loss_rotation = torch.zeros(1, device=self.device)
                    for i in range(self.skel.shape[-1]):
                        for j in range(i+1, self.skel.shape[-1]):
                            if self.skel[i,j] > 0:
                                loss_trans += ((bones_trans[i] - bones_trans[j]) - (self.skel_trans[i] - self.skel_trans[j])) * self.skel[i,j]
                                loss_rotation += ((bones_qr[i] * bones_qr[j]) - (self.skel_qr[i] * self.skel_qr[j])).norm(dim=-1) * self.skel[i,j]
         
                    loss_dict["skel_preserve_loss"] = (loss_trans + loss_rotation) * self.config["skel_preserve_wt"]
                # print("surf_loss", loss_dict["surf_loss"].item(), "smooth_loss", loss_dict["smooth_loss"].item())
            # dot = np.dot(q1, q2)
            
            # # 确保点积的绝对值不超过1
            # dot = max(min(dot, 1.0), -1.0)
            
            # # 计算角度（弧度）
            # theta = 2 * np.arccos(abs(dot))

        self.total_iters += 1
        self.round_count = self.total_iters // self.config["iters_per_round"]
        # import ipdb; ipdb.set_trace()
        if self.total_iters % 100 < 2 or self.config["debug"]:

            print("----iters", self.total_iters," ----")
            # time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
            print("loss dict is",[f"{k}:{v.item():.5f}" for k,v in loss_dict.items()], 2*"\n")

        if self.total_iters % 25 == 0 and self.config["debug"]:
            self.export_geometry_aux("%s/it%03d" % ( os.path.join(self.config["logroot"], self.config["logname"]), self.total_iters))

        return loss_dict

    def process_frameid(self, batch):
        """Convert frameid within each video to overall frame id

        Args:
            batch (Dict): Batch of input metadata. Keys: "dataid" (M,),
                "frameid_sub" (M,), "crop2raw" (M,4), "feature" (M,N,16), and
                "hxy" (M,N,3). This function modifies it in place to add key
                "frameid" (M,)
        """
        if not hasattr(self, "offset_cuda"):
            self.offset_cache = torch.tensor(
                self.data_info["frame_info"]["frame_offset_raw"],
                device=self.device,
                dtype=torch.long,
            )
        # import ipdb; ipdb.set_trace()
        # convert frameid_sub to frameid
        batch["frameid"] = batch["frameid_sub"] + self.offset_cache[batch["dataid"]]

    def set_progress(self, current_steps, is_fine=False, is_coarse=False):
        """Adjust loss weights and other constants throughout training

        Args:wosang
            current_steps (int): Number of optimization steps so far
        """
        total_iters = self.config["num_rounds"] * self.config["iters_per_round"] 
        
        # positional encoding annealing
        anchor_x = (0, total_iters)
        anchor_y = (0.6, 1)
        type = "linear"

        if "update_step" in dir(self.fields.field_params['fg'].pos_embedding):
            # print("NOT PERFECT ANNEALING")
            self.fields.field_params['fg'].pos_embedding.update_step(0, current_steps)
            # if current_steps % self.fields.field_params['fg'].pos_embedding.update_steps == 0:
            #     print("set level to", self.fields.field_params['fg'].pos_embedding.current_level)

        alpha = interp_wt(anchor_x, anchor_y, current_steps, type=type)
        if alpha >= 1:
            alpha = None
        self.fields.set_alpha(alpha)
        
        assert not (is_coarse and is_fine)
        if is_fine:
            self.fields.set_alpha(self.config["top_alpha"])
        if is_coarse:
            self.fields.set_alpha(0.7)
        # beta_prob: steps(0->2k, 1->0.2), range (0.2,1)
        anchor_x = (0, total_iters / 2)
        anchor_y = (1.0, 0.2)
        type = "linear"
        beta_prob = interp_wt(anchor_x, anchor_y, current_steps, type=type)
        self.fields.set_beta_prob(beta_prob)

        # camera prior wt: steps(0->800, 1->0), range (0,1)
        loss_name = "reg_cam_prior_wt"
        anchor_x = (0, total_iters / 5)
        anchor_y = (1, 0)
        type = "linear"
        self.set_loss_weight(loss_name, anchor_x, anchor_y, current_steps, type=type)

        # reg_eikonal_wt: steps(0->24000, 1->100), range (1,100)
        loss_name = "reg_eikonal_wt"
        anchor_x = (0, total_iters * self.config['reg_anneal'])
        anchor_y = (1, 100)
        type = "log"
        self.set_loss_weight(loss_name, anchor_x, anchor_y, current_steps, type=type)

        # skel prior wt: steps(0->4000, 1->0), range (0,1)
        loss_name = "reg_skel_prior_wt"
        anchor_x = (0, total_iters * self.config['reg_anneal'])
        anchor_y = (1, 0)
        type = "linear"
        self.set_loss_weight(loss_name, anchor_x, anchor_y, current_steps, type=type)

        # gauss mask wt: steps(0->4000, 1->0), range (0,1)
        loss_name = "reg_gauss_mask_wt"
        anchor_x = (0, total_iters * self.config['reg_anneal'])
        anchor_y = (1, 0)
        type = "linear"
        self.set_loss_weight(loss_name, anchor_x, anchor_y, current_steps, type=type)

        loss_name = "dreamo_wt"
        anchor_x = (0, total_iters * self.config['reg_anneal'])
        anchor_y = (1, 0)
        type = "linear"
        self.set_loss_weight(loss_name, anchor_x, anchor_y, current_steps, type=type)

        #
        # loss_name = "gen3d_wt"
        # anchor_x = (0, 4000)
        # anchor_y = (0, 1)
        # type = "linear"
        # self.set_loss_weight(loss_name, anchor_x, anchor_y, current_steps, type=type)

        if self.config['gen3d_loss_anneal'] > 0:
            loss_name = "gen3d_wt"
            anchor_x = (0, total_iters * self.config['gen3d_loss_anneal'])
            anchor_y = (1e-10, 1)
            type = "linear"
            self.set_loss_weight(loss_name, anchor_x, anchor_y, current_steps, type=type)
        elif self.config['gen3d_loss_anneal'] < 0:
            loss_name = "gen3d_wt"
            anchor_x = (0, total_iters * self.config['gen3d_loss_anneal'])
            anchor_y = (1, 1e-10)
            type = "linear"
            self.set_loss_weight(loss_name, anchor_x, anchor_y, current_steps, type=type)

        if self.config['all_reconloss_anneal'] >= 0:
            anchor_x = (0, total_iters * self.config['all_reconloss_anneal'])
            anchor_y = (1, 1e-6)
            type = self.config["all_anneal_type"]
            for loss_name in ['mask_wt','feature_wt','feat_reproj_wt','rgb_wt','depth_wt','flow_wt','vis_wt']:
                self.set_loss_weight(loss_name, anchor_x, anchor_y, current_steps, type=type)

        if self.config['rgb_loss_anneal'] > 0:
            loss_name = "rgb_wt"
            anchor_x = (0, total_iters * self.config['rgb_loss_anneal'])
            anchor_y = (1, 1e-6)
            type = self.config["rgb_anneal_type"]
            self.set_loss_weight(loss_name, anchor_x, anchor_y, current_steps, type=type)

        if self.config['mask_loss_anneal'] > 0:
            loss_name = "mask_wt"
            anchor_x = (0, total_iters * self.config['mask_loss_anneal'])
            anchor_y = (1, 1e-6)
            type = self.config["anneal_type"]
            self.set_loss_weight(loss_name, anchor_x, anchor_y, current_steps, type=type)

        
    def set_loss_weight(
        self, loss_name, anchor_x, anchor_y, current_steps, type="linear"
    ):
        """Set a loss weight according to the current training step

        Args:
            loss_name (str): Name of loss weight to set
            anchor_x: Tuple of optimization steps [x0, x1]
            anchor_y: Tuple of loss values [y0, y1]
            current_steps (int): Current optimization step
            type (str): Interpolation type ("linear" or "log")
        """
        if "%s_init" % loss_name not in self.config.keys():
            self.config["%s_init" % loss_name] = self.config[loss_name]
        factor = interp_wt(anchor_x, anchor_y, current_steps, type=type)
        self.config[loss_name] = self.config["%s_init" % loss_name] * factor

    def evaluate(self, batch, is_pair=True, no_warp=False):
        """Evaluate a Lab4D model

        Args:
            batch (Dict): Dataset metadata from `construct_eval_batch()`. Keys:
                "dataid" (M,), "frameid_sub" (M,), "crop2raw" (M,4),
                "feature" (M,N,16), and "hxy" (M,N,3)
            is_pair (bool): Whether to evaluate by rendering pairs
        Returns:
            rendered (Dict): Dict of rendered outputs. Keys: "mask" (M,H,W,1),
                "vis" (M,H,W,1), "depth" (M,H,W,1), "flow" (M,H,W,2),
                "feature" (M,H,W,16), "normal" (M,H,W,3), and
                "eikonal" (M,H,W,1)
        """
        if is_pair:
            div_factor = 2
        else:
            div_factor = 1
        self.process_frameid(batch)

        rendered = defaultdict(list)
        # split batch
        for i in tqdm(range(0, len(batch["frameid"]) // div_factor)):
            batch_sub = {}
            for k, v in batch.items():
                if isinstance(v, dict):
                    batch_sub[k] = {}
                    for k2, v2 in v.items():
                        batch_sub[k][k2] = v2[i * div_factor : (i + 1) * div_factor]
                else:
                    batch_sub[k] = v[i * div_factor : (i + 1) * div_factor]

            rendered_sub = self.render(batch_sub, no_warp=no_warp)["rendered"]

            for k, v in rendered_sub.items():
                res = int(np.sqrt(v.shape[1]))
                rendered[k].append(v.view(div_factor, res, res, -1)[0])
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        for k, v in rendered.items():
            rendered[k] = torch.stack(v, 0)

        # blend with mask: render = render * mask + 0*(1-mask)
        # import ipdb; ipdb.set_trace()
        
        WHITE_BKGD = False
        WHITE_BKGD = True
        # del rendered["xyz_matches"]
        # del rendered["xyz_reproj"]
        for k, v in rendered.items():
            if k in ["mask", "xyz_matches", "xyz_reproj"]:
                continue
            else:
                if WHITE_BKGD:
                    # rendered["mask"] = 0
                    # import ipdb; ipdb.set_trace()
                    try:
                        rendered[k] = rendered[k] * rendered["mask"] + (1 - rendered["mask"])
                    except:
                        rendered[k] = torch.nn.functional.interpolate(rendered[k].permute(0,3,1,2), size=rendered["mask"].shape[0:2], mode='bilinear', align_corners=False).permute(0,2,3,1) * rendered["mask"]
                    # rendered[k][rendered[k]==0] = 1
                else:
                    try:
                        rendered[k] = rendered[k] * rendered["mask"]
                    except:
                        rendered[k] = torch.nn.functional.interpolate(rendered[k].permute(0,3,1,2), size=rendered["mask"].shape[0:2], mode='bilinear', align_corners=False).permute(0,2,3,1) * rendered["mask"]
            # rendered[k][rendered[k]==0] = 1
        return rendered

    def update_geometry_aux(self):
        """Extract proxy geometry for all neural fields"""
        self.fields.update_geometry_aux()

    def export_geometry_aux(self, path):
        """Export proxy geometry for all neural fields"""
        return self.fields.export_geometry_aux(path)

    def render(self, batch, flow_thresh=None, no_warp=False):
        """Render model outputs

        Args:
            batch (Dict): Batch of input metadata. Keys: "dataid" (M,),
                "frameid_sub" (M,), "crop2raw" (M,4), "feature" (M,N,16),
                "hxy" (M,N,3), and "frameid" (M,)
            flow_thresh (float): Flow magnitude threshold, for `compute_flow()`
        Returns:
            results: Rendered outputs. Keys: "rendered", "aux_dict"
            results["rendered"]: "mask" (M,N,1), "rgb" (M,N,3),
                "vis" (M,N,1), "depth" (M,N,1), "flow" (M,N,2),
                "feature" (M,N,16), "normal" (M,N,3), and "eikonal" (M,N,1)
            results["aux_dict"]["fg"]: "xy_reproj" (M,N,2) and "feature" (M,N,16) 
        """
        samples_dict = self.get_samples(batch)  # Homogeneous pixel coords on the image plane
        if no_warp:
            samples_dict['fg']['no_warp'] = True
        results = self.render_samples_chunk(samples_dict, flow_thresh=flow_thresh)
        # print(results['rendered']['rgb'].shape)
        # import ipdb; ipdb.set_trace()

        return results

    def get_samples(self, batch):
        """Compute time-dependent camera and articulation parameters for all
        neural fields.

        Args:
            batch (Dict): Batch of input metadata. Keys: "dataid" (M,),
                "frameid_sub" (M,), "crop2raw" (M,4), "feature" (M,N,16),
                "hxy" (M,N,3), and "frameid" (M,)
        Returns:
            samples_dict (Dict): Input metadata and time-dependent outputs.
                Keys: "Kinv" (M,3,3), "field2cam" (M,4,4), "frame_id" (M,),
                "inst_id" (M,), "near_far" (M,2), "hxy" (M,N,3), and
                "feature" (M,N,16).  
        M = 2 N = 4096
        """
        # if batch["hxy"].shape[0] < 10:
        # import ipdb; ipdb.set_trace() # Kinv should be M 3 3
        if "Kinv" in batch.keys():
            Kinv = batch["Kinv"]
        else:
            Kmat = self.intrinsics.get_vals(batch["frameid"]) # intrinsics (focal is from mlp.forward(frameid))
            Kinv = K2inv(Kmat) @ K2mat(batch["crop2raw"])

        samples_dict = self.fields.get_samples(Kinv, batch)

        return samples_dict

    def render_samples_chunk(self, samples_dict, flow_thresh=None, chunk_size=8192):
        """Render outputs from all neural fields. Divide in chunks along pixel
        dimension N to avoid running out of memory.

        Args:
            samples_dict (Dict): Maps neural field types ("bg" or "fg") to
                dicts of input metadata and time-dependent outputs.
                Each dict has keys: "Kinv" (M,3,3), "field2cam" (M,4,4),
                "frame_id" (M,), "inst_id" (M,), "near_far" (M,2),
                "hxy" (M,N,3), and "feature" (M,N,16).
            flow_thresh (float): Flow magnitude threshold, for `compute_flow()`
            chunk_size (int): Number of pixels to render per chunk
        Returns:
            results: Rendered outputs. Keys: "rendered", "aux_dict"
        """
        # get chunk size
        # print("chunck hxy shape", samples_dict['fg']['hxy'].shape)
        category = list(samples_dict.keys())[0]
        total_pixels = (
            samples_dict[category]["hxy"].shape[0]
            * samples_dict[category]["hxy"].shape[1]
        )
        num_chunks = int(np.ceil(total_pixels / chunk_size))
        chunk_size_n = int(
            np.ceil(chunk_size // samples_dict[category]["hxy"].shape[0])
        )  # at n dimension

        results = {
            "rendered": defaultdict(list),
            "aux_dict": defaultdict(defaultdict),
        }
        for i in range(num_chunks):
            # construct chunk input
            samples_dict_chunk = defaultdict(list)
            for category, category_v in samples_dict.items():
                samples_dict_chunk[category] = defaultdict(list)
                for k, v in category_v.items():
                    if k == "hxy":
                        samples_dict_chunk[category][k] = v[
                            :, i * chunk_size_n : (i + 1) * chunk_size_n
                        ]
                    else:
                        samples_dict_chunk[category][k] = v

            # get chunk output
            # if samples_dict_chunk['fg']['hxy'].shape[0] == 1:

            results_chunk = self.render_samples(
                samples_dict_chunk, flow_thresh=flow_thresh
            )

            # merge chunk output
            for k, v in results_chunk["rendered"].items():
                if k not in results["rendered"].keys():
                    results["rendered"][k] = []
                results["rendered"][k].append(v)

            for cate in results_chunk["aux_dict"].keys():
                for k, v in results_chunk["aux_dict"][cate].items():
                    if k not in results["aux_dict"][cate].keys():
                        results["aux_dict"][cate][k] = []
                    results["aux_dict"][cate][k].append(v)

        # concat chunk output
        for k, v in results["rendered"].items():
            # import ipdb;ipdb.set_trace()
            results["rendered"][k] = torch.cat(v, 1)

        for cate in results["aux_dict"].keys():
            for k, v in results["aux_dict"][cate].items():
                results["aux_dict"][cate][k] = torch.cat(v, 1)
        # import ipdb; ipdb.set_trace()
        if self.config['dreamo_wt'] > 0 or self.config["skel_preserve_wt"] > 0:
            bones_qr, bones_trans = dual_quaternion_to_quaternion_translation(samples_dict['fg']['t_articulation'])
            bones_sdf = self.fields.field_params['fg'].forward(bones_trans, inst_id=samples_dict['fg']["inst_id"], get_density=False)
            # import ipdb; ipdb.set_trace()
            results["aux_dict"]["fg"]["bones_qr"] = bones_qr
            results["aux_dict"]["fg"]["bones_trans"] = bones_trans
            results["aux_dict"]["fg"]["bones_sdf"] = bones_sdf
        # SDF at gaussian center and motion smothness
        
        return results

    def render_samples(self, samples_dict, flow_thresh=None):
        """Render outputs from all neural fields.

        Args:
            samples_dict (Dict): Maps neural field types ("bg" or "fg") to
                dicts of input metadata and time-dependent outputs.
                Each dict has keys: "Kinv" (M,3,3), "field2cam" (M,4,4),
                "frame_id" (M,), "inst_id" (M,), "near_far" (M,2),
                "hxy" (M,N,3), and "feature" (M,N,16).
            flow_thresh (float): Flow magnitude threshold, for `compute_flow()`
        Returns:
            results: Rendered outputs. Keys: "rendered", "aux_dict"
        """
        
        # if VRAM_DBG:
        #     import ipdb; ipdb.set_trace()
        multifields_dict, deltas_dict, aux_dict = self.fields.query_multifields(
            samples_dict, flow_thresh=flow_thresh
        )


        field_dict, deltas = self.fields.compose_fields(multifields_dict, deltas_dict)
        rendered = render_pixel(field_dict, deltas, render_uncert=self.config["render_uncert"])
        for cate in multifields_dict.keys():
            # render each field and put into aux_dict
            rendered_cate = render_pixel(multifields_dict[cate], deltas_dict[cate])
            for k, v in rendered_cate.items():
                aux_dict[cate][k] = v

        if "fg" in aux_dict.keys():
            # move for visualization
            if "xyz_matches" in aux_dict["fg"].keys():
                rendered["xyz_matches"] = aux_dict["fg"]["xyz_matches"]
                rendered["xyz_reproj"] = aux_dict["fg"]["xyz_reproj"]

        results = {"rendered": rendered, "aux_dict": aux_dict}
        return results

    @staticmethod
    def reshape_batch(batch):
        """Reshape a batch to merge the pair dimension into the batch dimension

        Args:
            batch (Dict): Arbitrary dataloader outputs (M, 2, ...). This is
                modified in place to reshape each value to (M*2, ...)
        """
        for k, v in batch.items():
            batch[k] = v.view(-1, *v.shape[2:])

    def compute_loss(self, batch, results, is_gen3d=False):
        """Compute model losses

        Args:
            batch (Dict): Batch of dataloader samples. Keys: "rgb" (M,2,N,3),
                "mask" (M,2,N,1), "depth" (M,2,N,1), "feature" (M,2,N,16),
                "flow" (M,2,N,2), "flow_uct" (M,2,N,1), "vis2d" (M,2,N,1),
                "crop2raw" (M,2,4), "dataid" (M,2), "frameid_sub" (M,2), and
                "hxy" (M,2,N,3)
            results: Rendered outputs. Keys: "rendered", "aux_dict"
        Returns:
            loss_dict (Dict): Computed losses. Keys: "mask" (M,N,1),
                "rgb" (M,N,3), "depth" (M,N,1), "flow" (M,N,1), "vis" (M,N,1),
                "feature" (M,N,1), "feat_reproj" (M,N,1),
                "reg_gauss_mask" (M,N,1), "reg_visibility" (0,),
                "reg_eikonal" (0,), "reg_deform_cyc" (0,),
                "reg_soft_deform" (0,), "reg_gauss_skin" (0,),
                "reg_cam_prior" (0,), and "reg_skel_prior" (0,).
        """


        # if is_gen3d: 

        #     loss_3d_gen = self.guidance(new_view_rendered, self.prompt_processor(), **threestudio_batch, rgb_as_latents=False)['loss_sds']
        #     loss_3d_gen *= self.config["gen3d_wt"]
        #     loss_dict={"3d_generate_loss": loss_3d_gen}
        #     if config['gen3d_regloss']:
        #         self.compute_reg_loss(loss_dict, results)

        config = self.config
        loss_dict = {}
        self.compute_recon_loss(loss_dict, results, batch, config)
        self.mask_losses(loss_dict, batch, config)
        self.compute_reg_loss(loss_dict, results)
        self.apply_loss_weights(loss_dict, config)
        return loss_dict

    @staticmethod
    def get_mask_balance_wt(mask, vis2d, is_detected):
        """Balance contribution of positive and negative pixels in mask.

        Args:
            mask: (M,N,1) Object segmentation mask
            vis2d: (M,N,1) Whether each pixel is visible in the video frame
            is_detected: (M,) Whether there is segmentation mask in the frame
        Returns:
            mask_balance_wt: (M,N,1) Balanced mask
        """
        # all the positive labels
        mask = mask.float()
        # all the labels
        vis2d = vis2d.float() * is_detected.float()[:, None, None]
        if mask.sum() > 0 and (1 - mask).sum() > 0:
            pos_wt = vis2d.sum() / mask[vis2d > 0].sum()
            neg_wt = vis2d.sum() / (1 - mask[vis2d > 0]).sum()
            mask_balance_wt = 0.5 * pos_wt * mask + 0.5 * neg_wt * (1 - mask)
        else:
            mask_balance_wt = 1
        return mask_balance_wt

    
    @staticmethod
    def compute_recon_loss(loss_dict, results, batch, config):
        """Compute reconstruction losses.

        Args:
            loss_dict (Dict): Updated in place to add keys: "mask" (M,N,1),
                "rgb" (M,N,3), "depth" (M,N,1), "flow" (M,N,1), "vis" (M,N,1),
                "feature" (M,N,1), "feat_reproj" (M,N,1), and
                "reg_gauss_mask" (M,N,1)
            results: Rendered outputs. Keys: "rendered", "aux_dict"
            batch (Dict): Batch of dataloader samples. Keys: "rgb" (M,2,N,3),
                "mask" (M,2,N,1), "depth" (M,2,N,1), "feature" (M,2,N,16),
                "flow" (M,2,N,2), "flow_uct" (M,2,N,1), "vis2d" (M,2,N,1),
                "crop2raw" (M,2,4), "dataid" (M,2), "frameid_sub" (M,2), and
                "hxy" (M,2,N,3)
            config (Dict): Command-line options
        """
        rendered = results["rendered"]
        aux_dict = results["aux_dict"]
        # reconstruction loss
        # get rendered fg mask
        if config["field_type"] == "fg":
            rendered_fg_mask = rendered["mask"]
        elif config["field_type"] == "comp":
            rendered_fg_mask = rendered["mask_fg"]
        elif config["field_type"] == "bg":
            rendered_fg_mask = None
        else:
            raise ("field_type %s not supported" % config["field_type"])
        # get fg mask balance factor
        mask_balance_wt = dvr_model.get_mask_balance_wt(
            batch["mask"], batch["vis2d"], batch["is_detected"]
        )
        if config["field_type"] == "bg":
            loss_dict["mask"] = (rendered["mask"] - 1).pow(2)
        elif config["field_type"] == "fg":
            loss_dict["mask"] = (rendered_fg_mask - batch["mask"].float()).pow(2)
            loss_dict["mask"] *= mask_balance_wt
        elif config["field_type"] == "comp":
            loss_dict["mask"] = (rendered_fg_mask - batch["mask"].float()).pow(2)
            loss_dict["mask"] *= mask_balance_wt
            loss_dict["mask"] += (rendered["mask"] - 1).pow(2)
        else:
            raise ("field_type %s not supported" % config["field_type"])

        if config['gen3d_wt'] == 0 and (config["field_type"] == "fg" or config["field_type"] == "comp"):
            loss_dict["feature"] = (aux_dict["fg"]["feature"] - batch["feature"]).norm(
                2, -1, keepdim=True
            )
            loss_dict["feat_reproj"] = (
                aux_dict["fg"]["xy_reproj"] - batch["hxy"][..., :2]
            ).norm(2, -1, keepdim=True)

        loss_dict["rgb"] = (rendered["rgb"] - batch["rgb"]).pow(2)
        loss_dict["depth"] = (
            (rendered["depth"] - batch["depth"]).norm(2, -1, keepdim=True).clone()
        )
        loss_dict["flow"] = (rendered["flow"] - batch["flow"]).norm(2, -1, keepdim=True)

        # visibility: supervise on fg and bg separately
        vis_loss = []
        # for aux_cate_dict in aux_dict.values():
        for cate, aux_cate_dict in aux_dict.items():
            if cate == "bg":
                # use smaller weight for bg
                aux_cate_dict["vis"] *= 0.01
            vis_loss.append(aux_cate_dict["vis"])
        vis_loss = torch.stack(vis_loss, 0).sum(0)
        loss_dict["vis"] = vis_loss

        # weighting
        loss_dict["flow"] = loss_dict["flow"] * (batch["flow_uct"] > 0).float()

        # consistency between rendered mask and gauss mask
        if "gauss_mask" in rendered.keys():
            loss_dict["reg_gauss_mask"] = (
                aux_dict["fg"]["gauss_mask"] - rendered_fg_mask.detach()
            ).pow(2)

    def compute_reg_loss(self, loss_dict, results):
        """Compute regularization losses.

        Args:
            loss_dict (Dict): Updated in place to add keys:
                "reg_visibility" (0,), "reg_eikonal" (0,),
                "reg_deform_cyc" (0,), "reg_soft_deform" (0,),
                "reg_gauss_skin" (0,), "reg_cam_prior" (0,), and
                "reg_skel_prior" (0,).
            results: Rendered outputs. Keys: "rendered", "aux_dict"
        """
        rendered = results["rendered"]
        aux_dict = results["aux_dict"]
        # regularization loss
        loss_dict["reg_visibility"] = self.fields.visibility_decay_loss()
        if "eikonal" in rendered.keys():
            loss_dict["reg_eikonal"] = rendered["eikonal"]
        if "fg" in aux_dict.keys():
            if "cyc_dist" in aux_dict["fg"].keys():
                loss_dict["reg_deform_cyc"] = aux_dict["fg"]["cyc_dist"]
                loss_dict["reg_delta_skin"] = aux_dict["fg"]["delta_skin"]
                loss_dict["reg_skin_entropy"] = aux_dict["fg"]["skin_entropy"]
        loss_dict["reg_soft_deform"] = self.fields.soft_deform_loss()
        loss_dict["reg_gauss_skin"] = self.fields.gauss_skin_consistency_loss()
        loss_dict["reg_cam_prior"] = self.fields.cam_prior_loss()
        loss_dict["reg_skel_prior"] = self.fields.skel_prior_loss()

    @staticmethod
    def mask_losses(loss_dict, batch, config):
        """Apply segmentation mask on dense losses

        Args:
            loss_dict (Dict): Dense losses. Keys: "mask" (M,N,1), "rgb" (M,N,3),
                "depth" (M,N,1), "flow" (M,N,1), "vis" (M,N,1), "feature" (M,N,1),
                "feat_reproj" (M,N,1), and "reg_gauss_mask" (M,N,1). Modified in
                place to multiply loss_dict["mask"] with the other losses
            batch (Dict): Batch of dataloader samples. Keys: "rgb" (M,2,N,3),
                "mask" (M,2,N,1), "depth" (M,2,N,1), "feature" (M,2,N,16),
                "flow" (M,2,N,2), "flow_uct" (M,2,N,1), "vis2d" (M,2,N,1),
                "crop2raw" (M,2,4), "dataid" (M,2), "frameid_sub" (M,2), and
                "hxy" (M,2,N,3)
            config (Dict): Command-line options
        """
        # ignore the masking step
        keys_ignore_masking = ["reg_gauss_mask"]
        # always mask-out non-visible (out-of-frame) pixels
        keys_allpix = ["mask"]
        # always mask-out non-object pixels
        keys_fg = ["feature", "feat_reproj"]
        # field type specific keys
        keys_type_specific = ["rgb", "depth", "flow", "vis"]

        # type-specific masking rules
        vis2d = batch["vis2d"].float()
        maskfg = batch["mask"].float()
        if config["field_type"] == "bg":
            mask = (1 - maskfg) * vis2d
        elif config["field_type"] == "fg":
            mask = maskfg * vis2d
        elif config["field_type"] == "comp":
            mask = vis2d
        else:
            raise ("field_type %s not supported" % config["field_type"])

        # apply mask
        for k, v in loss_dict.items():
            if k in keys_ignore_masking:
                continue
            elif k in keys_allpix:
                loss_dict[k] = v * vis2d
            elif k in keys_fg:
                loss_dict[k] = v * maskfg
            elif k in keys_type_specific:
                loss_dict[k] = v * mask
            else:
                raise ("loss %s not defined" % k)

        # mask out the following losses if obj is not detected
        keys_mask_not_detected = ["mask", "feature", "feat_reproj"]
        for k, v in loss_dict.items():
            if k in keys_mask_not_detected:
                loss_dict[k] = v * batch["is_detected"].float()[:, None, None]

    @staticmethod
    def apply_loss_weights(loss_dict, config):
        """Weigh each loss term according to command-line configs

        Args:
            loss_dict (Dict): Computed losses. Keys: "mask" (M,N,1),
                "rgb" (M,N,3), "depth" (M,N,1), "flow" (M,N,1), "vis" (M,N,1),
                "feature" (M,N,1), "feat_reproj" (M,N,1),
                "reg_gauss_mask" (M,N,1), "reg_visibility" (0,),
                "reg_eikonal" (0,), "reg_deform_cyc" (0,),
                "reg_soft_deform" (0,), "reg_gauss_skin" (0,),
                "reg_cam_prior" (0,), and "reg_skel_prior" (0,). Modified in
                place to multiply each term with a scalar weight.
            config (Dict): Command-line options
        """
        px_unit_keys = ["flow", "feat_reproj"]
        for k, v in loss_dict.items():
            # average over non-zero pixels

            loss_dict[k] = v[v > 0].mean()

            # scale with image resolution
            if k in px_unit_keys:
                loss_dict[k] /= config["train_res"]

            # scale with loss weights
            wt_name = k + "_wt"
            if wt_name in config.keys():
                # print("loss weight", wt_name, config[wt_name])
                loss_dict[k] *= config[wt_name]

def quaternion_translation_to_se3(q: torch.Tensor, t: torch.Tensor):
    rmat = quaternion_to_matrix(q)
    rt4x4 = torch.cat((rmat, t[..., None]), -1)  # (..., 3, 4)
    rt4x4 = torch.cat((rt4x4, torch.zeros_like(rt4x4[..., :1, :])), -2)  # (..., 4, 4)
    rt4x4[..., 3, 3] = 1
    return rt4x4

@torch.jit.script
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        o: Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    q2 = quaternions**2
    rr, ii, jj, kk = torch.unbind(q2, -1)
    two_s = 2.0 / q2.sum(-1)
    ij = i * j
    ik = i * k
    ir = i * r
    jk = j * k
    jr = j * r
    kr = k * r

    o1 = 1 - two_s * (jj + kk)
    o2 = two_s * (ij - kr)
    o3 = two_s * (ik + jr)
    o4 = two_s * (ij + kr)

    o5 = 1 - two_s * (ii + kk)
    o6 = two_s * (jk - ir)
    o7 = two_s * (ik - jr)
    o8 = two_s * (jk + ir)
    o9 = 1 - two_s * (ii + jj)

    o = torch.stack((o1, o2, o3, o4, o5, o6, o7, o8, o9), -1)

    return o.view(quaternions.shape[:-1] + (3, 3))

def process_results(self, new_view_results, batch_size, suffix="train"):
    for key_name in ["normal", "rgb"]:#"mask","depth","vis",
        # import ipdb; ipdb.set_trace()
        if key_name not in new_view_results["rendered"].keys():
            continue
        channel = new_view_results["rendered"][key_name].shape[-1]
        new_view_rendered = new_view_results["rendered"][key_name].reshape(batch_size, self.gen3d_res,self.gen3d_res,channel) # gen3d_res,gen3d_res,3HWC  # new_view_results['aux_dict']['fg'].keys() = dict_keys(['mask', 'rgb', 'cyc_dist', 'delta_skin', 'skin_entropy', 'normal', 'xyz', 'xyz_cam', 'depth', 'feature', 'flow', 'mask_fg', 'eikonal', 'vis', 'gauss_mask'])
        tmask = new_view_results["rendered"]["mask"].reshape(batch_size, self.gen3d_res,self.gen3d_res,1)
        if self.config['rd_bg']:
            new_view_rendered = new_view_rendered * tmask + (1 - tmask) * torch.tensor(np.random.rand(channel),device=new_view_rendered.device, dtype=torch.float).view(1,3)
        elif self.config['w_bg']:
            new_view_rendered = new_view_rendered * tmask + (1 - tmask)
        else:
            new_view_rendered = new_view_rendered * new_view_results["rendered"]["mask"].reshape(batch_size, self.gen3d_res,self.gen3d_res,1)

        # new_view_rendered_merged = new_view_rendered.permute(0,3,1,2).reshape(1,3,self.gen3d_res*2,self.gen3d_res*2)
        if new_view_rendered.shape[0] == 4:
            collage = torch.zeros(self.gen3d_res * 2, self.gen3d_res * 2, channel, device=new_view_rendered.device)
            for i in range(2):
                for j in range(2):
                    collage[i * self.gen3d_res : (i + 1) * self.gen3d_res, j * self.gen3d_res : (j + 1) * self.gen3d_res] = new_view_rendered[i * 2 + j]
            vutils.save_image(collage.permute(2,0,1), f'look_gradinfo/{self.config["seqname"]}{self.config["logname"]}_{key_name}_{suffix}.png')
        else:
            collage = new_view_rendered[0]
            vutils.save_image(new_view_rendered[0].permute(2,0,1), f'look_gradinfo/{self.config["seqname"]}{self.config["logname"]}_{key_name}_{suffix}.png')

    return new_view_rendered, collage