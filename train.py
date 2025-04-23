#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, kl_divergence
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from pytorch3d.ops import knn_points

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def arap_loss_for_gaussians(xyz, d_xyz, prev_d_xyz=None, knn_idx=None, knn_dists=None, knn_weights=None):
    """
    更完整的ARAP损失实现，比较当前帧和前一帧的变形
    
    参数:
        xyz: 当前点云位置 (canonical位置)
        d_xyz: 当前点云变形向量
        prev_d_xyz: 前一帧的变形向量，None表示与canonical帧比较
        knn_idx: KNN索引 
        knn_dists: KNN距离
        knn_weights: KNN权重
    """
    N = xyz.shape[0]
    
    # 计算当前帧变形后的位置
    pos_after = xyz + d_xyz  # [N, 3]
    
    # 如果前一帧变形向量不存在，则使用零变形（即与canonical帧比较）
    prev_d = torch.zeros_like(d_xyz) if prev_d_xyz is None else prev_d_xyz
    prev_pos = xyz + prev_d  # [N, 3]
    
    # 获取邻居点位置
    nb_xyz = xyz[knn_idx]  # [N, K, 3]
    nb_pos_after = pos_after[knn_idx]  # [N, K, 3]
    nb_prev_pos = prev_pos[knn_idx]  # [N, K, 3]
    
    # 计算邻居点相对于中心点的偏移向量
    offset_xyz = nb_xyz - xyz.unsqueeze(1)  # [N, K, 3]
    offset_after = nb_pos_after - pos_after.unsqueeze(1)  # [N, K, 3]
    offset_prev = nb_prev_pos - prev_pos.unsqueeze(1)  # [N, K, 3]
    
    # 计算距离保持损失 (iso loss)：每个点与其邻居之间的距离应在变形前后保持一致
    dist_after = torch.norm(offset_after, dim=2)  # [N, K]
    dist_prev = torch.norm(offset_prev, dim=2)  # [N, K]
    iso_loss = torch.abs(dist_after - dist_prev)  # [N, K]
    
    # 计算旋转一致性损失：当前帧和前一帧的变形方向应当一致
    K = knn_idx.shape[1]
    
    # 构建二阶邻居结构
    neighbor_indices_2d = knn_idx.reshape(-1)  # [N*K]
    
    # 获取每个邻居点的邻居点
    neighbor_of_neighbors = knn_idx[neighbor_indices_2d].reshape(N, K, K)  # [N, K, K]
    
    # 计算相对邻居关系：对所有局部邻居之间的关系
    nb_offset_xyz = xyz[neighbor_of_neighbors] - nb_xyz.unsqueeze(2)  # [N, K, K, 3]
    nb_offset_after = pos_after[neighbor_of_neighbors] - nb_pos_after.unsqueeze(2)  # [N, K, K, 3]
    nb_offset_prev = prev_pos[neighbor_of_neighbors] - nb_prev_pos.unsqueeze(2)  # [N, K, K, 3]
    
    # 归一化偏移向量以关注方向变化
    norm_xyz = torch.norm(nb_offset_xyz, dim=3, keepdim=True) + 1e-6
    norm_after = torch.norm(nb_offset_after, dim=3, keepdim=True) + 1e-6
    norm_prev = torch.norm(nb_offset_prev, dim=3, keepdim=True) + 1e-6
    
    unit_xyz = nb_offset_xyz / norm_xyz
    unit_after = nb_offset_after / norm_after
    unit_prev = nb_offset_prev / norm_prev
    
    # 计算方向一致性：当前帧和前一帧的局部变形方向应该相似
    dir_change_curr = 1.0 - torch.sum(unit_xyz * unit_after, dim=3)  # [N, K, K]
    dir_change_prev = 1.0 - torch.sum(unit_xyz * unit_prev, dim=3)  # [N, K, K]
    
    # 方向变化的差异作为旋转一致性损失
    rot_loss = torch.abs(dir_change_curr - dir_change_prev).mean(dim=2)  # [N, K]
    
    # 应用权重
    if knn_weights is not None:
        weighted_iso_loss = (iso_loss * knn_weights).sum() / (knn_weights.sum() + 1e-8)
        weighted_rot_loss = (rot_loss * knn_weights).sum() / (knn_weights.sum() + 1e-8)
    else:
        weighted_iso_loss = iso_loss.mean()
        weighted_rot_loss = rot_loss.mean()
    
    # 返回综合损失，权重比例参考Dynamic3DGaussians
    return 4.0 * weighted_rot_loss + 2.0 * weighted_iso_loss


def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel(dataset.is_blender, dataset.is_6dof)
    deform.train_setting(opt)

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # 分阶段训练设置
    densify_phase = True  # 初始阶段为密集化阶段
    densify_until_iter = 10000  # 密集化阶段结束于10000次迭代
    arap_gap = 1000  # 密集化结束后，等待1000次迭代再应用ARAP约束
    arap_start_iter = densify_until_iter + arap_gap

    # KNN相关变量
    knn_idx = None
    knn_dists = None
    knn_weights = None
    
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
    
    for iteration in range(1, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    # 为缺少的参数提供默认值
                    d_xyz_default = torch.zeros_like(gaussians.get_xyz)
                    d_rotation_default = torch.zeros((gaussians.get_xyz.shape[0], 4), device="cuda")
                    d_scaling_default = torch.zeros((gaussians.get_xyz.shape[0], 3), device="cuda")
                    net_image = render(custom_cam, gaussians, pipe, background, d_xyz_default, d_rotation_default, d_scaling_default, False)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        # 每1000次迭代增加SH级别，最多到指定的最大值
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 随机选择一个摄像机视角
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        # 计算变形
        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)

            ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration)
            d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise)
        
        # 密集化阶段结束后，等待arap_gap次迭代，再初始化ARAP约束的KNN
        if iteration == densify_until_iter:
            print(f"[迭代 {iteration}] 密集化阶段结束，进入过渡期")
            densify_phase = False

        # 在过渡期结束后初始化ARAP约束的KNN
        if iteration == arap_start_iter:
            try:
                from pytorch3d.ops import knn_points
                
                # 确保使用contiguous的张量来构建KNN
                xyz = gaussians.get_xyz.detach().contiguous()
                knn_result = knn_points(xyz.unsqueeze(0), xyz.unsqueeze(0), K=20+1)
                knn_idx = knn_result.idx.squeeze(0)[:, 1:].contiguous()  # [N, K]
                knn_dists = knn_result.dists.squeeze(0)[:, 1:].sqrt().contiguous()  # [N, K]
                knn_weights = torch.exp(-2000 * knn_dists**2).contiguous()
                
                print(f"[迭代 {iteration}] 过渡期结束，初始化ARAP约束的邻居关系")
                print(f"KNN 索引形状: {knn_idx.shape}, 是否连续: {knn_idx.is_contiguous()}")
                print(f"当前点云大小: {xyz.shape[0]}")
            except ImportError:
                print("无法导入pytorch3d.ops.knn_points，无法应用ARAP约束")
            except Exception as e:
                print(f"初始化KNN时出错: {str(e)}")

        # 渲染
        render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]

        # 损失计算
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # 在密集化阶段结束后且KNN初始化完成后应用ARAP约束
        if not densify_phase and iteration >= arap_start_iter and knn_idx is not None:
            try:
                # 安全检查
                if knn_idx.shape[0] == gaussians.get_xyz.shape[0]:
                    # 随着训练的进行逐渐增加ARAP约束的权重
                    progress = min(1.0, (iteration - arap_start_iter) / 10000)
                    arap_weight = 0.1 * progress
                    
                    # 计算前一帧的变形向量（如果当前帧不是第一帧）
                    N = gaussians.get_xyz.shape[0]
                    curr_fid = fid
                    # 对于不是blender的数据集，前一帧简单地使用时间-0.1
                    # 对于blender数据集，我们假设前一帧是上一个索引
                    if dataset.is_blender:
                        # 对于blender数据集，前一帧的fid就是当前帧-1，但确保不小于0
                        prev_fid_value = max(0, curr_fid.item() - 1)
                        prev_fid = torch.tensor([prev_fid_value], device=curr_fid.device)
                    else:
                        # 对于非blender数据集，前一帧的fid是当前帧-时间间隔，但确保不小于0
                        prev_fid_value = max(0, curr_fid.item() - time_interval)
                        prev_fid = torch.tensor([prev_fid_value], device=curr_fid.device)
                    
                    # 扩展fid以匹配点云大小
                    prev_time_input = prev_fid.unsqueeze(0).expand(N, -1)
                    
                    # 使用deform模型计算前一帧的变形
                    prev_d_xyz, _, _ = deform.step(gaussians.get_xyz.detach(), prev_time_input)
                    
                    # 计算ARAP损失
                    loss_arap = arap_loss_for_gaussians(
                        xyz=gaussians.get_xyz.detach(),
                        d_xyz=d_xyz,
                        prev_d_xyz=prev_d_xyz,
                        knn_idx=knn_idx,
                        knn_dists=knn_dists,
                        knn_weights=knn_weights
                    )
                    
                    arap_term = arap_weight * loss_arap
                    loss = loss + arap_term
                    
                    if iteration % 100 == 0:
                        print(f"[迭代 {iteration}] ARAP损失: {loss_arap.item():.6f}, 权重: {arap_weight:.6f}")
                        print(f"当前fid: {curr_fid.item():.3f}, 前一帧fid: {prev_fid_value:.3f}")
                else:
                    print(f"[警告] KNN索引大小 ({knn_idx.shape[0]}) 与点云大小 ({gaussians.get_xyz.shape[0]}) 不匹配，跳过ARAP损失")
            except Exception as e:
                print(f"计算ARAP损失时出错: {str(e)}")

        loss.backward()

        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # 进度条
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # 记录用于剪枝的最大2D半径
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            # 记录和保存
            cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render, (pipe, background), deform,
                                       dataset.load2gpu_on_the_fly, dataset.is_6dof)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                model_path = opt.model_path
                deform.save_weights(model_path, iteration)

            # 仅在密集化阶段进行密集化操作
            if densify_phase and iteration < densify_until_iter:
                viewspace_point_tensor_densify = render_pkg_re["viewspace_points_densify"]
                gaussians.add_densification_stats(viewspace_point_tensor_densify, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    old_point_count = gaussians.get_xyz.shape[0]
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    new_point_count = gaussians.get_xyz.shape[0]
                    if new_point_count != old_point_count:
                        print(f"[迭代 {iteration}] 密集化操作：点数从 {old_point_count} 变为 {new_point_count}")

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # 优化器步骤
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                deform.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, deform, load2gpu_on_the_fly, is_6dof=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof)["render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[1000, 3000, 5000, 6000, 7000] + list(range(10000, 40001, 2000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 3_000, 5_000, 6_000,7_000, 10_000, 20_000, 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations)

    # All done
    print("\nTraining complete.")
