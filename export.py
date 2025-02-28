# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
"""
python lab4d/export.py --flagfile=logdir/cat-85-sub-sub-bob-pika-cate-b02/opts.log --load_suffix latest --inst_id 0
"""

import os, sys
import json
from typing import NamedTuple, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import trimesh
from absl import app, flags

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)


from lab4d.config import get_config
from lab4d.dataloader import data_utils
from lab4d.engine.trainer import Trainer
from lab4d.nnutils.warping import SkinningWarp
from lab4d.nnutils.pose import ArticulationSkelMLP
from lab4d.utils.io import make_save_dir, save_rendered
from lab4d.utils.quat_transform import (
    dual_quaternion_to_se3,
    quaternion_translation_to_se3,
    dual_quaternion_to_quaternion_translation,
)

cudnn.benchmark = True


class ExportMeshFlags:
    flags.DEFINE_integer("inst_id", 0, "video/instance id")
    flags.DEFINE_integer("grid_size", 128, "grid size of marching cubes")
    flags.DEFINE_integer("num_frames", -1, "number of frames to render")
    flags.DEFINE_float(
        "level", 0.0, "contour value of marching cubes use to search for isosurfaces"
    )


class MotionParamsExpl(NamedTuple):
    """
    explicit motion params for reanimation and transfer
    """

    field2cam: Tuple[torch.Tensor, torch.Tensor]  # (quaternion, translation)
    t_articulation: Tuple[
        torch.Tensor, torch.Tensor
    ]  # dual quaternion, applies to skinning
    so3: torch.Tensor  # so3, applies to skeleton
    mesh_t: trimesh.Trimesh  # mesh at time t
    bone_t: trimesh.Trimesh  # bone center at time t


def extract_deformation(field, mesh_rest, inst_id, render_length,bone_filter=None):
    device = next(field.parameters()).device
    xyz = torch.tensor(mesh_rest.vertices, dtype=torch.float32, device=device)
    offset = field.frame_offset_raw[inst_id]
    inst_id = torch.tensor([inst_id], dtype=torch.long, device=device)

    motion_tuples = {}
    # import ipdb; ipdb.set_trace()
    for frame_id in range(render_length):
        frame_id_torch = torch.tensor(
            [offset + frame_id], dtype=torch.long, device=device
        )
        field2cam = field.camera_mlp.get_vals(frame_id_torch)

        samples_dict = {}
        if isinstance(field.warp, SkinningWarp):
            (
                samples_dict["t_articulation"],
                samples_dict["rest_articulation"],
            ) = field.warp.articulation.get_vals_and_mean(frame_id_torch)
            t_articulation = samples_dict["t_articulation"]

            if isinstance(field.warp.articulation, ArticulationSkelMLP):
                so3 = field.warp.articulation.get_vals(frame_id_torch, return_so3=True)
            else:
                so3 = None

            # bones
            
            mesh_bones_t = field.warp.skinning_model.draw_gaussian(
                (
                    samples_dict["t_articulation"][0][0],
                    samples_dict["t_articulation"][1][0],
                    
                ),
                field.warp.articulation.edges,
                bone_filter=bone_filter
            )
            se3_mat = quaternion_translation_to_se3(field2cam[0], field2cam[1])[0]
            mesh_bones_t.apply_transform(se3_mat.cpu().numpy())
        else:
            t_articulation = None
            so3 = None
            mesh_bones_t = None

        xyz_t = field.forward_warp(
            xyz[None, None],
            field2cam,
            frame_id_torch,
            inst_id,
            samples_dict=samples_dict,
        )
        xyz_t = xyz_t[0, 0]
        mesh_t = trimesh.Trimesh(vertices=xyz_t.cpu().numpy(), faces=mesh_rest.faces)

        field2cam[1][:] /= field.logscale.exp()  # to world scale
        motion_expl = MotionParamsExpl(
            field2cam=field2cam,
            t_articulation=t_articulation,
            so3=so3,
            mesh_t=mesh_t,
            bone_t=mesh_bones_t,
        )
        motion_tuples[frame_id] = motion_expl

    if isinstance(field.warp, SkinningWarp):
        # modify rest mesh based on instance morphological changes on bones
        # idendity transformation of cameras
        field2cam_rot_idn = torch.zeros_like(field2cam[0])
        field2cam_rot_idn[..., 0] = 1.0
        field2cam_idn = (field2cam_rot_idn, torch.zeros_like(field2cam[1]))
        # bone stretching from rest to instance id
        samples_dict["t_articulation"] = field.warp.articulation.get_mean_vals(
            inst_id=inst_id
        )
        xyz_i = field.forward_warp(
            xyz[None, None],
            field2cam_idn,
            None,
            inst_id,
            samples_dict=samples_dict,
        )
        xyz_i = xyz_i[0, 0]
        mesh_rest = trimesh.Trimesh(vertices=xyz_i.cpu().numpy(), faces=mesh_rest.faces)

    return mesh_rest, motion_tuples


def save_motion_params(meshes_rest, motion_tuples, save_dir, skel=None):
    for cate, mesh_rest in meshes_rest.items():
        os.chdir("/mnt/mfs/xinzhou.wang/repo/DreamBANMo/")
        mesh_rest.export("%s/%s.obj" % (save_dir, cate))
        motion_params = {"field2cam": [], "t_articulation": [], "joint_so3": []}
        for frame_id, motion_expl in motion_tuples[cate].items():
            # save mesh
            motion_expl.mesh_t.export("%s/%s-%05d.obj" % (save_dir, cate, frame_id))
            if motion_expl.bone_t is not None:
                motion_expl.bone_t.export(
                    "%s/%s-%05d-bone.obj" % (save_dir, cate, frame_id)
                )

            # save motion params
            field2cam = quaternion_translation_to_se3(
                motion_expl.field2cam[0], motion_expl.field2cam[1]
            )  # 1,4,4
            motion_params["field2cam"].append(field2cam.cpu().numpy()[0].tolist())

            if motion_expl.t_articulation is not None:
                t_articulation = dual_quaternion_to_se3(
                    motion_expl.t_articulation
                )  # 1,K,4,4
                motion_params["t_articulation"].append(
                    t_articulation.cpu().numpy()[0].tolist()
                )
            if motion_expl.so3 is not None:
                motion_params["joint_so3"].append(
                    motion_expl.so3.cpu().numpy()[0].tolist()
                )  # K,3

        with open("%s/%s-motion.json" % (save_dir, cate), "w") as fp:
            json.dump(motion_params, fp)


@torch.no_grad()
def extract_motion_params(model, opts, data_info, bone_filter=None):
    # get rest mesh
    meshes_rest = model.fields.extract_canonical_meshes(
        grid_size=opts["grid_size"],
        level=opts["level"],
        inst_id=opts["inst_id"],
        use_visibility=False,
        # use_extend_aabb=False, ## wxz dbg
        use_extend_aabb=True
    )

    # get length of the seq
    vid_length = data_utils.get_vid_length(opts["inst_id"], data_info)
    if opts["num_frames"] > 0:
        render_length = opts["num_frames"]
    else:
        render_length = vid_length

    # get deformation
    motion_tuples = {}
    for cate, field in model.fields.field_params.items():
        meshes_rest[cate], motion_tuples[cate] = extract_deformation(
            field, meshes_rest[cate], opts["inst_id"], render_length, bone_filter=bone_filter
        )
    return meshes_rest, motion_tuples


def export(opts):
    model, data_info, ref_dict = Trainer.construct_test_model(opts)
    save_dir = make_save_dir(opts, sub_dir="export_%04d" % (opts["inst_id"]))
    # save motion paramters
    meshes_rest, motion_tuples = extract_motion_params(model, opts, data_info)
    save_motion_params(meshes_rest, motion_tuples, save_dir)

    # for frame_id, motion_expl in motion_tuples["fg"].items()
    
    # save reference images
    raw_size = data_info["raw_size"][opts["inst_id"]]  # full range of pixels
    save_rendered(ref_dict, save_dir, raw_size, data_info["apply_pca_fn"])
    print("Saved to %s" % save_dir)


    articulations = model.fields.field_params['fg'].warp.articulation.get_mean_vals()  # (1,K,4,4)
    articulation_rest = (articulations[0][0], articulations[1][0])
    # import ipdb; ipdb.set_trace()
    skel, mesh_geo = model.fields.generate_fg_skel(articulation_rest, mesh_geo=meshes_rest['fg'], return_geo=True)
    # print("skel", skel)
    bone_filter = np.zeros(skel.shape[0])
    # import ipdb; ipdb.set_trace()
    for i in range(skel.shape[0]):
        for j in range(i, skel.shape[1]):
            if skel[i, j] > 0:
                bone_filter[i] = 1
                bone_filter[j] = 1
    # print("bone_filter", bone_filter)
    meshes_rest, motion_tuples = extract_motion_params(model, opts, data_info, bone_filter=bone_filter)
    for frame_id, motion_expl in motion_tuples["fg"].items():
        # save mesh
        print(save_dir, frame_id)
        motion_expl.mesh_t.export("%s/%s-%05d.obj" % (save_dir, "fg", frame_id))
        motion_expl.mesh_t.visual.vertex_colors = mesh_geo.visual.vertex_colors
        motion_expl.mesh_t.export("%s/%s-%05d-geo-color.obj" % (save_dir, "fg", frame_id))
        skel_mesh = motion_expl.bone_t
        field2cam = motion_expl.field2cam
        field2cam[1][:] *= model.fields.field_params['fg'].logscale.exp()
        se3_mat = quaternion_translation_to_se3(field2cam[0], field2cam[1])[0].detach().cpu().numpy()
        if motion_expl.bone_t is not None:
            # import ipdb; ipdb.set_trace()
            qr, trans = dual_quaternion_to_quaternion_translation(motion_expl.t_articulation)
            trans = trans[0].cpu().numpy()
            for i in range(skel.shape[0]):
                for j in range(i, skel.shape[1]):
                    if skel[i, j] > 0:
                        cylinder = create_cylinder_between_points(trans[i], trans[j])
                        pct = skel[i, j]
                        color = [255 * pct, 0, 255 * (1-pct), 128]  
                        vertex_colors = np.tile(color, (len(cylinder.vertices), 1))
                        cylinder.visual.vertex_colors = vertex_colors
                        cylinder.apply_transform(se3_mat)
                        # cylinder.visual.face_colors = vertex_colors
                        skel_mesh = trimesh.util.concatenate([skel_mesh, cylinder])
            skel_mesh.export("%s/%s-%05d-skel.obj" % (save_dir, "fg", frame_id))








def main(_):
    opts = get_config()
    export(opts)

def create_cylinder_between_points(p1, p2, radius=0.002, segments=32):
    # 计算两点之间的向量和距离
    vec = p2 - p1
    height = np.linalg.norm(vec)
    direction = vec / height
    
    # 创建一个圆柱
    cylinder = trimesh.creation.cylinder(radius, height, sections=segments)
    
    # 计算旋转矩阵
    align_vector = [0, 0, 1] # 假设圆柱初始方向为 z 轴
    rotation_axis = np.cross(align_vector, direction)
    rotation_angle = np.arccos(np.dot(align_vector, direction))
    rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, rotation_axis)
    
    # 应用旋转和平移
    cylinder.apply_transform(rotation_matrix)
    cylinder.apply_translation((p1+p2)/2)
    
    return cylinder

if __name__ == "__main__":
    app.run(main)
