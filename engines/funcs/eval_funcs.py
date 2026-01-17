import os
from tqdm.auto import tqdm
import torch
import numpy as np
from utils.evaluation import cal_3d_position_error, match_2d_greedy, get_matching_dict, compute_prf1, vectorize_distance, calculate_iou, select_and_align
from utils.transforms import unNormalize
from utils.visualization import tensor_to_BGR, pad_img
from utils.visualization import vis_meshes_img, vis_boxes, vis_sat, vis_scale_img, get_colors_rgb
from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from utils.constants import H36M_EVAL_JOINTS
import time
import datetime
import scipy.io as sio
import cv2
import zipfile
import pickle


# Modified from agora_evaluation
def evaluate_agora(model, eval_dataloader, conf_thresh,
                        vis = True, vis_step = 40, results_save_path = None,
                        distributed = False, accelerator = None):
    assert results_save_path is not None
    assert accelerator is not None

    # whether the dataset contains 'kid' label
    has_kid = ('train' in eval_dataloader.dataset.split and eval_dataloader.dataset.ds_name == 'agora')
    
    os.makedirs(results_save_path,exist_ok=True)
    if vis:
        imgs_save_dir = os.path.join(results_save_path, 'imgs')
        os.makedirs(imgs_save_dir, exist_ok = True)
    
    step = 0
    total_miss_count = 0
    total_count = 0
    total_fp = 0


    mve, mpjpe = [], []
    if has_kid:
        kid_total_miss_count = 0
        kid_total_count = 0
        kid_mve, kid_mpjpe = [], []

    cur_device = next(model.parameters()).device
    smpl_layer = model.human_model
    body_verts_ind = smpl_layer.body_vertex_idx
    
    progress_bar = tqdm(total=len(eval_dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description('evaluate')
    for itr, (samples, targets) in enumerate(eval_dataloader):
        samples=[sample.to(device = cur_device, non_blocking = True) for sample in samples]
        with torch.no_grad():
            outputs = model(samples, targets)

        # per-batch aggregators
        batch_count = []
        batch_miss_count = []
        batch_fp = []
        batch_mve = []
        batch_mpjpe = []
        if has_kid:
            batch_kid_count = []
            batch_kid_miss_count = []
            batch_kid_mve = []
            batch_kid_mpjpe = []
        
        bs = len(targets)
        for idx in range(bs):
            batch_count.append(0)
            batch_miss_count.append(0)
            batch_fp.append(0)
            # avoid empty gather
            sample_mve = [float('inf')]
            sample_mpjpe = [float('inf')]
            if has_kid:
                batch_kid_count.append(0)
                batch_kid_miss_count.append(0)
                sample_kid_mve = [float('inf')]
                sample_kid_mpjpe = [float('inf')]

            # gt
            gt_j2ds = targets[idx]['j2ds'].cpu().numpy()[:,:24,:]
            gt_j3ds = targets[idx]['j3ds'].cpu().numpy()[:,:24,:]
            gt_verts = targets[idx]['verts'].cpu().numpy()

            # pred
            select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
            pred_j2ds = outputs['pred_j2ds'][idx][select_queries_idx].detach().cpu().numpy()[:,:24,:]
            pred_j3ds = outputs['pred_j3ds'][idx][select_queries_idx].detach().cpu().numpy()[:,:24,:]
            pred_verts = outputs['pred_verts'][idx][select_queries_idx].detach().cpu().numpy()

            # matching
            matched_verts_idx = []
            assert len(gt_j2ds.shape) == 3 and len(pred_j2ds.shape) == 3
            greedy_match = match_2d_greedy(pred_j2ds, gt_j2ds)  # tuples are (idx_pred_kps, idx_gt_kps)
            matchDict, falsePositive_count = get_matching_dict(greedy_match)

            # align matched pairs
            gt_verts_list, pred_verts_list, gt_joints_list, pred_joints_list = [], [], [], []
            gtIdxs = np.arange(len(gt_j3ds))
            miss_flag = []
            for gtIdx in gtIdxs:
                gt_verts_list.append(gt_verts[gtIdx])
                gt_joints_list.append(gt_j3ds[gtIdx])
                if matchDict[str(gtIdx)] == 'miss' or matchDict[str(gtIdx)] == 'invalid':
                    miss_flag.append(1)
                    pred_verts_list.append([])
                    pred_joints_list.append([])
                else:
                    miss_flag.append(0)
                    pred_joints_list.append(pred_j3ds[matchDict[str(gtIdx)]])
                    pred_verts_list.append(pred_verts[matchDict[str(gtIdx)]])
                    matched_verts_idx.append(matchDict[str(gtIdx)])

            if has_kid:
                gt_kid_list = targets[idx]['kid']

            # compute 3D errors
            for i, (gt3d, pred) in enumerate(zip(gt_joints_list, pred_joints_list)):
                batch_count[-1] += 1
                if has_kid and gt_kid_list[i]:
                    batch_kid_count[-1] += 1

                # Get corresponding ground truth and predicted 3d joints and verts
                if miss_flag[i] == 1:
                    # miss case
                    batch_miss_count[-1] += 1
                    if has_kid and gt_kid_list[i]:
                        batch_kid_miss_count[-1] += 1
                    continue

                gt3d = gt3d.reshape(-1, 3)
                pred3d = pred.reshape(-1, 3)
                gt3d_verts = gt_verts_list[i].reshape(-1, 3)
                pred3d_verts = pred_verts_list[i].reshape(-1, 3)
                
                gt3d, gt3d_verts = select_and_align(gt3d, gt3d_verts, body_verts_ind)
                pred3d, pred3d_verts = select_and_align(pred3d, pred3d_verts, body_verts_ind)

                # joints
                error_j, _ = cal_3d_position_error(pred3d, gt3d)
                sample_mpjpe.append(float(error_j))
                if has_kid and gt_kid_list[i]:
                    sample_kid_mpjpe.append(float(error_j))
                # vertices
                error_v, _ = cal_3d_position_error(pred3d_verts, gt3d_verts)
                sample_mve.append(float(error_v))
                if has_kid and gt_kid_list[i]:
                    sample_kid_mve.append(float(error_v))

            # counting and visualization
            step += 1
            batch_fp[-1] += falsePositive_count

            # stash per-sample arrays (with leading inf)
            batch_mve.append(np.array(sample_mve))
            batch_mpjpe.append(np.array(sample_mpjpe))
            if has_kid:
                batch_kid_mve.append(np.array(sample_kid_mve))
                batch_kid_mpjpe.append(np.array(sample_kid_mpjpe))

            img_idx = step + accelerator.process_index*len(eval_dataloader)*bs
            if vis and (img_idx%vis_step == 0):
                img_name = targets[idx]['img_path'].split('/')[-1].split('.')[0]
                ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())

                # render mesh
                colors = [(1.0, 1.0, 0.9)] * len(gt_verts)
                gt_mesh_img = vis_meshes_img(img = ori_img.copy(),
                                            verts = gt_verts,
                                            smpl_faces = smpl_layer.faces,
                                            cam_intrinsics = targets[idx]['cam_intrinsics'].reshape(3,3).detach().cpu(),
                                            colors = colors)

                colors = [(1.0, 0.6, 0.6)] * len(pred_verts)   
                for i in matched_verts_idx:
                    colors[i] = (0.7, 1.0, 0.4)
                pred_mesh_img = vis_meshes_img(img = ori_img.copy(),
                                            verts = pred_verts,
                                            smpl_faces = smpl_layer.faces,
                                            cam_intrinsics = outputs['pred_intrinsics'][idx].reshape(3,3).detach().cpu(),
                                            colors = colors)

                # scale map
                if 'enc_outputs' not in outputs:
                    pred_scale_img = np.zeros_like(pred_mesh_img)
                else:
                    enc_out = outputs['enc_outputs']
                    h, w = enc_out['hw'][idx]
                    flatten_map = enc_out['scale_map'].split(enc_out['lens'])[idx].detach().cpu()
                    ys = enc_out['pos_y'].split(enc_out['lens'])[idx]
                    xs = enc_out['pos_x'].split(enc_out['lens'])[idx]
                    scale_map = torch.zeros((h,w,2))
                    scale_map[ys,xs] = flatten_map
                    pred_scale_img = vis_scale_img(img = ori_img.copy(),
                                                   scale_map = scale_map,
                                                   conf_thresh = model.sat_cfg['conf_thresh'],
                                                   patch_size=28)

                # boxes
                pred_boxes = outputs['pred_boxes'][idx][select_queries_idx].detach().cpu()
                pred_boxes = box_cxcywh_to_xyxy(pred_boxes) * model.input_size
                pred_box_img = vis_boxes(ori_img.copy(), pred_boxes, color = (255,0,255))

                # sat
                sat_img = vis_sat(ori_img.copy(),
                                  input_size = model.input_size,
                                  patch_size = 14,
                                  sat_dict = outputs['sat'],
                                  bid = idx)

                ori_img = pad_img(ori_img, model.input_size)
                full_img = np.vstack([np.hstack([ori_img, sat_img]),
                                      np.hstack([pred_scale_img, pred_box_img]),
                                      np.hstack([gt_mesh_img, pred_mesh_img])])
                cv2.imwrite(os.path.join(imgs_save_dir, f'{img_idx}_{img_name}.png'), full_img)

        # distributed gather at batch level
        if distributed:
            batch_count = accelerator.gather_for_metrics(batch_count)
            batch_miss_count = accelerator.gather_for_metrics(batch_miss_count)
            batch_fp = accelerator.gather_for_metrics(batch_fp)
            batch_mve = accelerator.gather_for_metrics(batch_mve)
            batch_mpjpe = accelerator.gather_for_metrics(batch_mpjpe)
            if has_kid:
                batch_kid_count = accelerator.gather_for_metrics(batch_kid_count)
                batch_kid_miss_count = accelerator.gather_for_metrics(batch_kid_miss_count)
                batch_kid_mve = accelerator.gather_for_metrics(batch_kid_mve)
                batch_kid_mpjpe = accelerator.gather_for_metrics(batch_kid_mpjpe)

        # update totals
        total_count += sum(batch_count)
        total_miss_count += sum(batch_miss_count)
        total_fp += sum(batch_fp)
        mve += batch_mve
        mpjpe += batch_mpjpe
        if has_kid:
            kid_total_count += sum(batch_kid_count)
            kid_total_miss_count += sum(batch_kid_miss_count)
            kid_mve += batch_kid_mve
            kid_mpjpe += batch_kid_mpjpe

        progress_bar.update(1)

    # collect all results and drop placeholder inf
    mve = np.concatenate([item[1:] for item in mve], axis=0).tolist() if len(mve) > 0 else []
    mpjpe = np.concatenate([item[1:] for item in mpjpe], axis=0).tolist() if len(mpjpe) > 0 else []
    if has_kid:
        kid_mve = np.concatenate([item[1:] for item in kid_mve], axis=0).tolist() if len(kid_mve) > 0 else []
        kid_mpjpe = np.concatenate([item[1:] for item in kid_mpjpe], axis=0).tolist() if len(kid_mpjpe) > 0 else []

    if len(mpjpe) <= 0:
        return "Failed to evaluate. Keep training!"
    if has_kid and len(kid_mpjpe) <= 0:
        return "Failed to evaluate. Keep training!"
    
    precision, recall, f1 = compute_prf1(total_count, total_miss_count, total_fp)
    error_dict = {}
    error_dict['precision'] = precision
    error_dict['recall'] = recall
    error_dict['f1'] = f1

    error_dict['MPJPE'] = round(float(sum(mpjpe)/len(mpjpe)), 1)
    #error_dict['NMJE'] = round(error_dict['MPJPE'] / (f1), 1)
    if f1 != 0:
        error_dict['NMJE'] = round(error_dict['MPJPE'] / f1, 1)
    else:
         error_dict['NMJE'] = float('inf')  # 或者根据需要设为0
    error_dict['MVE'] = round(float(sum(mve)/len(mve)), 1)
    #error_dict['NMVE'] = round(error_dict['MVE'] / (f1), 1)
    if f1 != 0:
        error_dict['NMVE'] = round(error_dict['MVE'] / f1, 1)
    else:
        error_dict['NMVE'] = float('inf')

    if has_kid:
        kid_precision, kid_recall, kid_f1 = compute_prf1(kid_total_count, kid_total_miss_count, total_fp)
        error_dict['kid_precision'] = kid_precision
        error_dict['kid_recall'] = kid_recall
        error_dict['kid_f1'] = kid_f1

        error_dict['kid-MPJPE'] = round(float(sum(kid_mpjpe)/len(kid_mpjpe)), 1)
        error_dict['kid-NMJE'] = round(error_dict['kid-MPJPE'] / (kid_f1), 1)
        error_dict['kid-MVE'] = round(float(sum(kid_mve)/len(kid_mve)), 1)
        error_dict['kid-NMVE'] = round(error_dict['kid-MVE'] / (kid_f1), 1)

    if accelerator.is_main_process:
        with open(os.path.join(results_save_path,'results.txt'),'w') as f:
            for k,v in error_dict.items():
                f.write(f'{k}: {v}\n')

    return error_dict


def test_agora(model, eval_dataloader, conf_thresh, 
                vis = True, vis_step = 400, results_save_path = None,
                distributed = False, accelerator = None):
    assert results_save_path is not None
    assert accelerator is not None

    os.makedirs(os.path.join(results_save_path,'predictions'),exist_ok=True)
    if vis:
        imgs_save_dir = os.path.join(results_save_path, 'imgs')
        os.makedirs(imgs_save_dir, exist_ok = True)
    step = 0
    cur_device = next(model.parameters()).device
    smpl_layer = model.human_model
    
    progress_bar = tqdm(total=len(eval_dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description('testing')
    for itr, (samples, targets) in enumerate(eval_dataloader):
        samples=[sample.to(device = cur_device, non_blocking = True) for sample in samples]
        with torch.no_grad():    
           outputs = model(samples, targets)
        bs = len(targets)
        for idx in range(bs):
            #gt
            img_name = targets[idx]['img_name'].split('.')[0]
            #pred
            select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
            pred_j2ds = np.array(outputs['pred_j2ds'][idx][select_queries_idx].detach().to('cpu'))[:,:24,:]*(3840/model.input_size)
            pred_j3ds = np.array(outputs['pred_j3ds'][idx][select_queries_idx].detach().to('cpu'))[:,:24,:]
            pred_verts = np.array(outputs['pred_verts'][idx][select_queries_idx].detach().to('cpu'))
            pred_poses = np.array(outputs['pred_poses'][idx][select_queries_idx].detach().to('cpu'))
            pred_betas = np.array(outputs['pred_betas'][idx][select_queries_idx].detach().to('cpu'))

            #visualization
            step+=1
            img_idx = step + accelerator.process_index*len(eval_dataloader)*bs
            if vis and (img_idx%vis_step == 0):
                ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())
                ori_img = pad_img(ori_img, model.input_size)

                sat_img = vis_sat(ori_img.copy(),
                                    input_size = model.input_size,
                                    patch_size = 14,
                                    sat_dict = outputs['sat'],
                                    bid = idx)
                
                colors = get_colors_rgb(len(pred_verts))
                mesh_img = vis_meshes_img(img = ori_img.copy(),
                                          verts = pred_verts,
                                          smpl_faces = smpl_layer.faces,
                                          colors = colors,
                                          cam_intrinsics = outputs['pred_intrinsics'][idx].detach().cpu())
                
                if 'enc_outputs' not in outputs:
                    pred_scale_img = np.zeros_like(ori_img)
                else:
                    enc_out = outputs['enc_outputs']
                    h, w = enc_out['hw'][idx]
                    flatten_map = enc_out['scale_map'].split(enc_out['lens'])[idx].detach().cpu()

                    ys = enc_out['pos_y'].split(enc_out['lens'])[idx]
                    xs = enc_out['pos_x'].split(enc_out['lens'])[idx]
                    scale_map = torch.zeros((h,w,2))
                    scale_map[ys,xs] = flatten_map
                    pred_scale_img = vis_scale_img(img = ori_img.copy(),
                                                   scale_map = scale_map,
                                                   conf_thresh = model.sat_cfg['conf_thresh'],
                                                   patch_size=28)

                full_img = np.vstack([np.hstack([ori_img, mesh_img]),
                                      np.hstack([pred_scale_img, sat_img])])
                cv2.imwrite(os.path.join(imgs_save_dir, f'{img_idx}_{img_name}.png'), full_img)

            
            # submit
            for pnum in range(len(pred_j2ds)):
                smpl_dict = {}
                # smpl_dict['age'] = 'kid'
                smpl_dict['joints'] = pred_j2ds[pnum].reshape(24,2)
                smpl_dict['params'] = {'transl': np.zeros((1,3)),
                                        'betas': pred_betas[pnum].reshape(1,10),
                                        'global_orient': pred_poses[pnum][:3].reshape(1,1,3),
                                        'body_pose': pred_poses[pnum][3:].reshape(1,23,3)}
                # smpl_dict['verts'] = pred_verts[pnum].reshape(6890,3)
                # smpl_dict['allSmplJoints3d'] = pred_j3ds[pnum].reshape(24,3)
                with open(os.path.join(results_save_path,'predictions',f'{img_name}_personId_{pnum}.pkl'), 'wb') as f:
                    pickle.dump(smpl_dict, f)
 
        progress_bar.update(1)

    accelerator.print('Packing...')

    folder_path = os.path.join(results_save_path,'predictions')
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(results_save_path,f'pred_{timestamp}.zip')
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(folder_path))
                zipf.write(file_path, arcname)


    return 'Results saved at: ' + os.path.join(results_save_path,'predictions')

def evaluate_3dpw(model, eval_dataloader, conf_thresh,
                        vis = True, vis_step = 40, results_save_path = None,
                        distributed = False, accelerator = None):
    assert results_save_path is not None
    assert accelerator is not None
    
    os.makedirs(results_save_path,exist_ok=True)
    if vis:
        imgs_save_dir = os.path.join(results_save_path, 'imgs')
        os.makedirs(imgs_save_dir, exist_ok = True)
    
    step = 0
    total_miss_count = 0
    total_count = 0
    total_fp = 0

    # store arrays per sample then concatenate once at the end
    mve, mpjpe, pa_mpjpe, pa_mve = [], [], [], []
    cur_device = next(model.parameters()).device
    smpl_layer = model.human_model
    smpl2h36m_regressor = torch.from_numpy(smpl_layer.smpl2h36m_regressor).float().to(cur_device)
    
    progress_bar = tqdm(total=len(eval_dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description('evaluate')
    for itr, (samples, targets) in enumerate(eval_dataloader):
        samples=[sample.to(device = cur_device, non_blocking = True) for sample in samples]
        with torch.no_grad():
           outputs = model(samples, targets)

        # per-batch aggregators
        batch_count = []
        batch_miss_count = []
        batch_fp = []
        batch_mve = []
        batch_mpjpe = []
        batch_pa_mve = []
        batch_pa_mpjpe = []

        bs = len(targets)
        for idx in range(bs):
            batch_count.append(0)
            batch_miss_count.append(0)
            batch_fp.append(0)
            # avoid empty gather
            sample_mve = [float('inf')]
            sample_pa_mve = [float('inf')]
            sample_mpjpe = [float('inf')]
            sample_pa_mpjpe = [float('inf')]

            # gt 
            gt_verts = targets[idx]['verts']
            gt_transl = targets[idx]['transl']
            gt_j3ds = torch.einsum('bik,ji->bjk', [gt_verts - gt_transl[:,None,:], smpl2h36m_regressor]) + gt_transl[:,None,:]

            gt_verts = gt_verts.cpu().numpy()
            gt_j3ds = gt_j3ds.cpu().numpy()
            gt_j2ds = targets[idx]['j2ds'].cpu().numpy()[:,:24,:]

            # pred
            select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
            pred_verts = outputs['pred_verts'][idx][select_queries_idx].detach()
            pred_transl = outputs['pred_transl'][idx][select_queries_idx].detach()
            pred_j3ds = torch.einsum('bik,ji->bjk', [pred_verts - pred_transl[:,None,:], smpl2h36m_regressor]) + pred_transl[:,None,:]
            
            pred_verts = pred_verts.cpu().numpy()
            pred_j3ds = pred_j3ds.cpu().numpy()
            pred_j2ds = outputs['pred_j2ds'][idx][select_queries_idx].detach().cpu().numpy()[:,:24,:]

            # matching
            matched_verts_idx = []
            assert len(gt_j2ds.shape) == 3 and len(pred_j2ds.shape) == 3
            greedy_match = match_2d_greedy(pred_j2ds, gt_j2ds)  # tuples are (idx_pred_kps, idx_gt_kps)
            matchDict, falsePositive_count = get_matching_dict(greedy_match)

            # align with matching result
            gt_verts_list, pred_verts_list, gt_joints_list, pred_joints_list = [], [], [], []
            gtIdxs = np.arange(len(gt_j3ds))
            miss_flag = []
            for gtIdx in gtIdxs:
                gt_verts_list.append(gt_verts[gtIdx])
                gt_joints_list.append(gt_j3ds[gtIdx])
                if matchDict[str(gtIdx)] == 'miss' or matchDict[str(gtIdx)] == 'invalid':
                    miss_flag.append(1)
                    pred_verts_list.append([])
                    pred_joints_list.append([])
                else:
                    miss_flag.append(0)
                    pred_joints_list.append(pred_j3ds[matchDict[str(gtIdx)]])
                    pred_verts_list.append(pred_verts[matchDict[str(gtIdx)]])
                    matched_verts_idx.append(matchDict[str(gtIdx)])

            # compute 3D errors
            for i, (gt3d, pred) in enumerate(zip(gt_joints_list, pred_joints_list)):
                batch_count[-1] += 1

                if miss_flag[i] == 1:
                    batch_miss_count[-1] += 1
                    continue

                gt3d = gt3d.reshape(-1, 3)
                pred3d = pred.reshape(-1, 3)
                gt3d_verts = gt_verts_list[i].reshape(-1, 3)
                pred3d_verts = pred_verts_list[i].reshape(-1, 3)

                gt_pelvis = gt3d[[0],:].copy()
                pred_pelvis = pred3d[[0],:].copy()

                gt3d = (gt3d - gt_pelvis)[H36M_EVAL_JOINTS, :].copy()
                gt3d_verts = (gt3d_verts - gt_pelvis).copy()
                
                pred3d = (pred3d - pred_pelvis)[H36M_EVAL_JOINTS, :].copy()
                pred3d_verts = (pred3d_verts - pred_pelvis).copy()

                # joints
                error_j, pa_error_j = cal_3d_position_error(pred3d, gt3d)
                sample_mpjpe.append(float(error_j))
                sample_pa_mpjpe.append(float(pa_error_j))
                # vertices
                error_v, pa_error_v = cal_3d_position_error(pred3d_verts, gt3d_verts)
                sample_mve.append(float(error_v))
                sample_pa_mve.append(float(pa_error_v))

            # counting and visualization
            step += 1
            batch_fp[-1] += falsePositive_count

            # stash per-sample arrays (with leading inf)
            batch_mve.append(np.array(sample_mve))
            batch_pa_mve.append(np.array(sample_pa_mve))
            batch_mpjpe.append(np.array(sample_mpjpe))
            batch_pa_mpjpe.append(np.array(sample_pa_mpjpe))

            img_idx = step + accelerator.process_index*len(eval_dataloader)*bs
            if vis and (img_idx%vis_step == 0) and len(matched_verts_idx) > 0:
                img_name = targets[idx]['img_path'].split('/')[-1].split('.')[0]
                ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())
                ori_img = pad_img(ori_img, model.input_size)

                selected_verts = pred_verts[matched_verts_idx]
                colors = get_colors_rgb(len(selected_verts))
                mesh_img = vis_meshes_img(img = ori_img.copy(),
                                          verts = selected_verts,
                                          smpl_faces = smpl_layer.faces,
                                          colors = colors,
                                          cam_intrinsics = outputs['pred_intrinsics'][idx].detach().cpu())
                full_img = np.hstack([ori_img, mesh_img])
                cv2.imwrite(os.path.join(imgs_save_dir, f'{img_idx}_{img_name}.png'), full_img)

        # distributed gather at batch level
        if distributed:
            batch_count = accelerator.gather_for_metrics(batch_count)
            batch_miss_count = accelerator.gather_for_metrics(batch_miss_count)
            batch_fp = accelerator.gather_for_metrics(batch_fp)
            batch_mve = accelerator.gather_for_metrics(batch_mve)
            batch_pa_mve = accelerator.gather_for_metrics(batch_pa_mve)
            batch_mpjpe = accelerator.gather_for_metrics(batch_mpjpe)
            batch_pa_mpjpe = accelerator.gather_for_metrics(batch_pa_mpjpe)

        # update totals
        total_count += sum(batch_count)
        total_miss_count += sum(batch_miss_count)
        total_fp += sum(batch_fp)
        mve += batch_mve
        pa_mve += batch_pa_mve
        mpjpe += batch_mpjpe
        pa_mpjpe += batch_pa_mpjpe

        progress_bar.update(1)

    # collect all results and drop placeholder inf
    mve = np.concatenate([item[1:] for item in mve], axis=0).tolist() if len(mve) > 0 else []
    pa_mve = np.concatenate([item[1:] for item in pa_mve], axis=0).tolist() if len(pa_mve) > 0 else []
    mpjpe = np.concatenate([item[1:] for item in mpjpe], axis=0).tolist() if len(mpjpe) > 0 else []
    pa_mpjpe = np.concatenate([item[1:] for item in pa_mpjpe], axis=0).tolist() if len(pa_mpjpe) > 0 else []

    if len(mpjpe) <= 0:
        return "Failed to evaluate. Keep training!"
    
    precision, recall, f1 = compute_prf1(total_count,total_miss_count,total_fp)
    error_dict = {}
    error_dict['recall'] = recall

    error_dict['MPJPE'] = round(float(sum(mpjpe)/len(mpjpe)), 1)
    error_dict['PA-MPJPE'] = round(float(sum(pa_mpjpe)/len(pa_mpjpe)), 1)
    error_dict['MVE'] = round(float(sum(mve)/len(mve)), 1)
    error_dict['PA-MVE'] = round(float(sum(pa_mve)/len(pa_mve)), 1)

    if accelerator.is_main_process:
        with open(os.path.join(results_save_path,'results.txt'),'w') as f:
            for k,v in error_dict.items():
                f.write(f'{k}: {v}\n')

    return error_dict




