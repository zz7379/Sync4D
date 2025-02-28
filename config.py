# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import os

from absl import flags

opts = flags.FLAGS


class TrainModelConfig:
    # weights of reconstruction terms
    flags.DEFINE_float("mask_wt", 0.1, "weight for silhouette loss")
    flags.DEFINE_float("rgb_wt", 0.1, "weight for color loss")
    flags.DEFINE_float("depth_wt", 1e-4, "weight for depth loss")
    flags.DEFINE_float("flow_wt", 0.5, "weight for flow loss")
    flags.DEFINE_float("vis_wt", 1e-2, "weight for visibility loss")
    flags.DEFINE_float("feature_wt", 1e-2, "weight for feature reconstruction loss")
    flags.DEFINE_float("feat_reproj_wt", 5e-2, "weight for feature reprojection loss")

    flags.DEFINE_boolean("debug", False,"")
    flags.DEFINE_boolean("freeze_warp", False,"")
    flags.DEFINE_boolean("freeze_skin", False,"")
    flags.DEFINE_boolean("freeze_art", False,"")
    flags.DEFINE_boolean("test_in_train", False,"")
    flags.DEFINE_boolean("recon_keep_coarse", False,"")
    flags.DEFINE_boolean("gen3d_optim_all", False, "False not optimize embbeding")
    flags.DEFINE_boolean("gen3d_optim_new", False, "seems right")


    flags.DEFINE_integer('ablation_lockframeid', -1,"")
    flags.DEFINE_boolean('w_bg', True,"white backgound")
    flags.DEFINE_boolean('rd_bg', False,"random backgound")
    flags.DEFINE_boolean('no_negative_prompt', False,"")
    flags.DEFINE_boolean('gen3d_nosetprogress', False, "gen3d set_progress like recon")
    
    flags.DEFINE_boolean('motion_retarget', False,"")
    
    flags.DEFINE_float("top_alpha", 1,"")
    flags.DEFINE_float("gs", 50.0, "guidance_scale")
    flags.DEFINE_string("gen3d_guidance", 'mvd', "if or mvd")
    flags.DEFINE_boolean("recon_no_coarsetofine", False,"")
    flags.DEFINE_boolean('use_wide_near_far', False, "")
    flags.DEFINE_boolean('rgb_timefree', False, "not use time dependet nerf rgb")
    flags.DEFINE_boolean('rgb_dirfree', False, "not use dir dependet nerf rgb")
    flags.DEFINE_boolean('decoup_nerf', False, "decoup_nerf")

    flags.DEFINE_float("gen3d_wt", 0, "weight from 3d generation loss")
    flags.DEFINE_float("dreamo_wt", 0, "weight from dreamo")
    flags.DEFINE_float("skel_preserve_wt", 0, "weight of skel from recon")
    flags.DEFINE_float("bone_threshold", 0.25, "bone to skeleton threshold")

    
    
    flags.DEFINE_integer("gen3d_res", 64, "")
    flags.DEFINE_float("gen3d_dist", 1, 'render distance')
    flags.DEFINE_float("gen3d_freq", 2, "? images one sds")
    flags.DEFINE_boolean('low_pos_emb', False, "if true, nerf's position embedding dim is 1/2")


    # flags.DEFINE_integer("gen3d_start_iters", 0, "for sds_t and camera angle anneling")
    # flags.DEFINE_boolean("gen3d_dirprompt", False, "")
    flags.DEFINE_boolean("gen_in_canonical", False, "")
    flags.DEFINE_boolean("rac_no_template", False, "rac_no_template")
    flags.DEFINE_boolean("gen_skip_feat", True, "skip feature query in generation")
    flags.DEFINE_boolean("new_ts", False, "last 20% epoches t=[0.02,0.5]")
    flags.DEFINE_boolean("render_uncert", False, "")
    flags.DEFINE_integer("gen3d_frameid", -1, "-1 for random")


    flags.DEFINE_integer("seed",-1, "-1 for not seed anything")
    flags.DEFINE_boolean("gen3d_random_bkgd", False, "")
    flags.DEFINE_string("prompt", '', "!!!NOTE: split by _ name of the sequence")
    flags.DEFINE_boolean("reset_rgb_mlp", False, "")
    flags.DEFINE_float("gen3d_sds_t_max", 0.98, "sds_t_max")
    flags.DEFINE_float('rgb_loss_anneal', -1, "x:(0, total_iters * self.config['rgb_loss_anneal']) y:(1, 0)")
    flags.DEFINE_float('mask_loss_anneal', -1, "x:(0, total_iters * self.config['mask_loss_anneal']) y:(1, 0)")
    flags.DEFINE_float('all_reconloss_anneal', -1, "x:(0, total_iters * self.config['mask_loss_anneal']) y:(1, 0)")
    flags.DEFINE_float('gen3d_loss_anneal', 0, "<0 from 1 to 0 >0 from 0 to 1")

    flags.DEFINE_boolean("rgb_only", False, "")
    flags.DEFINE_boolean("geo_only", False, "")
    
    flags.DEFINE_string('rgb_anneal_type', "log", "linear or log")
    flags.DEFINE_string('anneal_type', "linear", "mask linear or log")
    flags.DEFINE_string('all_anneal_type', "linear", "all linear or log")

    flags.DEFINE_float('reg_anneal', 1,'') 


    
    # ["gen3d_jacobloss"] or self.opts["gen3d_sds_normal"]
    # flags.DEFINE_boolean("gen3d_regloss", False, "")

    flags.DEFINE_boolean("gen3d_jacobloss", False, "")
    flags.DEFINE_boolean("gen3d_cycloss", False, "")
    flags.DEFINE_boolean("gen3d_sds_normal", False, "")
    # flags.DEFINE_integer("num_rounds", 20, "number of rounds to train")
        

    flags.DEFINE_integer("lock_frameid", -1, "lock frameid for rgb querying")
    # weights of regularization terms
    flags.DEFINE_float(
        "reg_visibility_wt", 1e-4, "weight for visibility regularization"
    )
    flags.DEFINE_float("reg_eikonal_wt", 1e-3, "weight for eikonal regularization")
    flags.DEFINE_float(
        "reg_deform_cyc_wt", 0.01, "weight for deform cyc regularization"
    )
    flags.DEFINE_float("reg_delta_skin_wt", 5e-3, "weight for delta skinning reg")
    flags.DEFINE_float("reg_skin_entropy_wt", 5e-4, "weight for delta skinning reg")
    flags.DEFINE_float(
        "reg_gauss_skin_wt", 1e-3, "weight for gauss skinning consistency"
    )
    flags.DEFINE_float("reg_cam_prior_wt", 0.1, "weight for camera regularization")
    flags.DEFINE_float("reg_skel_prior_wt", 0.1, "weight for skeleton regularization")
    flags.DEFINE_float(
        "reg_gauss_mask_wt", 0.01, "weight for gauss mask regularization"
    )
    flags.DEFINE_float("reg_soft_deform_wt", 100.0, "weight for soft deformation reg")

    # model
    flags.DEFINE_string("field_type", "fg", "{bg, fg, comp}")
    flags.DEFINE_string(
        "fg_motion", "bob", "{rigid, dense, bob, skel-human, skel-quad}"
    )
    flags.DEFINE_bool("single_inst", True, "assume the same morphology over objs")


class TrainOptConfig:
    # io-related
    flags.DEFINE_string("seqname", "cat", "name of the sequence")
    flags.DEFINE_string("logname", "tmp", "name of the saved log")
    flags.DEFINE_string(
        "data_prefix", "crop", "prefix of the data entries, {crop, full}"
    )
    flags.DEFINE_integer("train_res", 256, "size of preprocessed training images")
    flags.DEFINE_string("logroot", "logdir/", "root directory for log files")
    flags.DEFINE_string("load_suffix", "", "sufix of params, {latest, 0, 10, ...}")
    flags.DEFINE_string("feature_type", "dinov2", "{dinov2, cse}")
    flags.DEFINE_string("load_path", "", "path to load pretrained model")

    # accuracy-related
    flags.DEFINE_float("learning_rate", 5e-4, "learning rate")
    flags.DEFINE_integer("num_rounds", 20, "number of rounds to train")
    flags.DEFINE_integer("iters_per_round", 200, "number of iterations per round")
    flags.DEFINE_integer("imgs_per_gpu", 128, "images samples per iter, per gpu")
    flags.DEFINE_integer("pixels_per_image", 32, "pixel samples per image")

    # vram
    flags.DEFINE_integer("n_depth", 64, "sampling points")
    flags.DEFINE_boolean("save_vram", False, "")
    flags.DEFINE_boolean("no_vis_mlp", False, "")

    flags.DEFINE_boolean(
        "freeze_bone_len", False, "do not change bone length of skeleton"
    )
    flags.DEFINE_boolean(
        "reset_steps",
        True,
        "reset steps of loss scheduling, set to False if resuming training",
    )

    # efficiency-related
    flags.DEFINE_integer("ngpu", 1, "number of gpus to use")
    flags.DEFINE_integer("num_workers", 2, "Number of workers for dataloading")
    flags.DEFINE_integer("eval_res", 64, "size used for eval visualizations")
    flags.DEFINE_integer("save_freq", 10, "params saving frequency")
    flags.DEFINE_boolean("profile", False, "profile the training loop")
    


def get_config():
    return opts.flag_values_dict()


def save_config():
    save_dir = os.path.join(opts.logroot, "%s-%s" % (opts.seqname, opts.logname))
    os.makedirs(save_dir, exist_ok=True)
    opts_path = os.path.join(save_dir, "opts.log")
    if os.path.exists(opts_path):
        os.remove(opts_path)
    opts.append_flags_into_file(opts_path)
