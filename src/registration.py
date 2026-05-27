import os
import copy
import numpy as np
import open3d as o3d
from scipy.ndimage import maximum_filter
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree

from config import *
from utils import (
    force_estimate_normals, extract_smooth_surface, extract_outer_surface,
    compute_fpfh_feature, preprocess_point_cloud, mirror_point_cloud,
    mask_target_region, evaluate_registration_detailed, classify_and_order_fragments
)

# ================= PCA 全局配准 =================

def _finite_points(pcd, label="point cloud"):
    points = np.asarray(pcd.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{label} must contain Nx3 points, got shape {points.shape}")

    finite_mask = np.isfinite(points).all(axis=1)
    if not finite_mask.all():
        points = points[finite_mask]
    if len(points) < 3:
        raise ValueError(f"{label} must contain at least 3 finite points, got {len(points)}")

    return points


def compute_pca(pcd, label="point cloud"):
    """
    计算点云的主成分分析 (PCA)。
    
    原理：
    通过计算点云的协方差矩阵并进行特征值分解，提取出点云的质心（中心位置）和
    主轴方向（特征向量）。在骨骼中，最大特征值对应的主轴通常代表骨骼的长轴。
    
    参数:
        pcd: 输入的 Open3D 点云对象。
    返回:
        centroid (质心坐标), eigenvectors (降序排列的特征向量矩阵), eigenvalues (降序排列的特征值)
    """
    points = _finite_points(pcd, label)
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    cov = centered.T @ centered / max(len(points) - 1, 1)
    if not np.all(np.isfinite(cov)):
        raise ValueError(f"{label} covariance contains NaN or Inf")
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    return centroid, eigenvectors[:, idx], eigenvalues[idx]


def align_by_pca(source_combined, target, verbose=False):
    """
    基于 PCA 的全局粗配准。
    
    原理：
    将患侧整体粗略对齐到健侧模板。PCA 提取的特征向量具有方向上的符号模糊性
    （正反向都是合法的特征向量），因此该函数会遍历 X/Y/Z 三个轴的 8 种翻转组合，
    通过计算采样点的平均最近邻距离，选出无倒置、无错误翻转的最佳粗配准矩阵。
    
    参数:
        source_combined: 合并后的患侧全套碎片点云。
        target: 健侧镜像模板点云。
        verbose: 是否打印详细日志。
    返回:
        best_transform: 4x4 的最佳初始粗配准变换矩阵。
    """
    source_points = _finite_points(source_combined, "combined source point cloud")
    target_points = _finite_points(target, "target point cloud")
    src_centroid, src_axes, _ = compute_pca(source_combined, "combined source point cloud")
    tgt_centroid, tgt_axes, _ = compute_pca(target, "target point cloud")
    
    sign_combinations = [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]]
    best_transform, best_distance = np.eye(4), np.inf
    target_tree = KDTree(target_points)
    
    for signs in sign_combinations:
        src_axes_flipped = src_axes.copy() * signs
        R = tgt_axes @ src_axes_flipped.T
        if np.linalg.det(R) < 0: continue  # 剔除行列式为负的矩阵（包含镜像反射，会导致患侧被错误镜像）
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tgt_centroid - R @ src_centroid
        
        transformed_points = (R @ source_points.T).T + T[:3, 3]
        sampled_points = transformed_points[::max(1, len(transformed_points) // 2000)]
        mean_dist = np.mean(target_tree.query(sampled_points, k=1)[0])
        
        if mean_dist < best_distance:
            best_distance = mean_dist
            best_transform = T
    return best_transform


def apply_transform_to_all_fragments(source_pcds, transform):
    """
    辅助函数：将一个 4x4 变换矩阵批量应用到所有独立的碎片点云上。
    """
    return [copy.deepcopy(pcd).transform(transform) for pcd in source_pcds]


def merge_point_clouds(pcd_list):
    """
    辅助函数：将多个独立的骨折碎片点云合并为一个完整的大点云，主要用于整体形态的 PCA 计算。
    """
    all_points = []
    for pcd in pcd_list:
        if pcd.is_empty():
            continue
        points = np.asarray(pcd.points, dtype=np.float64)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) > 0:
            all_points.append(points)
    if not all_points: return o3d.geometry.PointCloud()
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    return combined_pcd

# ================= 股骨头配准 (Hough Transform) =================

def find_sphere_centers(accumulator, min_bound, voxel_size, grid_dims, threshold):
    """
    从霍夫变换的三维累加器中提取得票最高的候选球心。
    
    原理：
    应用局部最大值滤波器（maximum_filter）找到累加器矩阵中的峰值，
    并使用 DBSCAN 算法对空间中靠得很近的峰值进行聚类，从而输出最稳健的球心坐标。
    
    参数:
        accumulator: 3D 投票累加器矩阵。
        min_bound: 累加器网格的物理起始坐标。
        voxel_size: 单个网格的物理尺寸。
        grid_dims: 累加器网格的维度。
        threshold: 投票阈值（相对于最高票数的比例）。
    返回:
        排好序的候选球心坐标数组（票数最高排在最前）。
    """
    threshold_value = threshold * np.max(accumulator)
    acc_thresh = np.where(accumulator >= threshold_value, accumulator, 0)
    local_max = maximum_filter(acc_thresh, size=10) == acc_thresh
    peak_coords = np.argwhere(local_max & (acc_thresh > 0))
    
    centers, scores = [], []
    for coord in peak_coords:
        centers.append(min_bound + coord * voxel_size)
        scores.append(accumulator[tuple(coord)])
    
    if not centers: return []
    centers, scores = np.array(centers), np.array(scores)
    if len(centers) > 1:
        clustering = DBSCAN(eps=10, min_samples=1).fit(centers)
        unique_centers, unique_scores = [], []
        for label in sorted(set(clustering.labels_)):
            if label == -1: continue
            mask = clustering.labels_ == label
            best_idx = np.argmax(scores[mask])
            unique_centers.append(centers[mask][best_idx])
            unique_scores.append(scores[mask][best_idx])
        sorted_idx = np.argsort(unique_scores)[::-1]
        return np.array(unique_centers)[sorted_idx]
    return centers


def hough_transform_sphere_detection(pcd, voxel_size=1.0, fixed_radius=None, radius_tolerance=2.0, config=None):
    """
    使用 3D 霍夫变换（Hough Transform）在点云中检测球面结构（寻找股骨头）。
    
    原理：
    让股骨头表面点云上的每一个点，沿着其法线方向向内/向外按照特定半径进行“投票”。
    所有法线交汇处（得票最高的空间网格）即为检测到的球心。
    
    参数:
        pcd: 待检测的点云（需具备法向量）。
        fixed_radius: 约束半径。如果已知健侧半径，可传入以极大缩小搜索空间和计算量。
        radius_tolerance: 半径搜索的容差范围。
    返回:
        centers (检测到的球心坐标列表), best_radius (得票最高的半径)
    """
    cfg = config if config is not None else HOUGH_CONFIG
    if not pcd.has_normals(): force_estimate_normals(pcd)
    points, normals = np.asarray(pcd.points), np.asarray(pcd.normals)
    if len(points) < 50: return [], None
    
    if fixed_radius is not None:
        min_radius, max_radius, radius_step = max(10.0, fixed_radius - radius_tolerance), fixed_radius + radius_tolerance, 0.5
    else:
        min_radius, max_radius, radius_step = cfg.get('min_radius', 18.0), cfg.get('max_radius', 30.0), cfg.get('radius_step', 1.0)
        
    grid_voxel = max(voxel_size * cfg.get('grid_voxel_factor', 1.5), 2.0)
    min_bound, max_bound = np.min(points, axis=0) - max_radius - 5, np.max(points, axis=0) + max_radius + 5
    grid_dims = np.ceil((max_bound - min_bound) / grid_voxel).astype(int)
    
    radii = np.arange(min_radius, max_radius + radius_step, radius_step)
    accumulator = np.zeros(grid_dims)
    
    step = max(1, len(points) // cfg.get('sample_ratio', 3000))
    sampled_pts, sampled_nms = points[::step], normals[::step]

    # 正向投票
    centers_pos = sampled_pts[:, None, :] - radii[None, :, None] * sampled_nms[:, None, :]
    idx_pos = np.floor((centers_pos - min_bound) / grid_voxel).astype(int)
    valid_pos = (idx_pos >= 0).all(axis=2) & (idx_pos < grid_dims).all(axis=2)
    vi = idx_pos[valid_pos]
    np.add.at(accumulator, (vi[:, 0], vi[:, 1], vi[:, 2]), 1)
    r_counts = valid_pos.sum(axis=0).astype(int)

    # 反向投票
    if cfg.get('bidirectional', True):
        centers_neg = sampled_pts[:, None, :] + radii[None, :, None] * sampled_nms[:, None, :]
        idx_neg = np.floor((centers_neg - min_bound) / grid_voxel).astype(int)
        valid_neg = (idx_neg >= 0).all(axis=2) & (idx_neg < grid_dims).all(axis=2)
        vi2 = idx_neg[valid_neg]
        np.add.at(accumulator, (vi2[:, 0], vi2[:, 1], vi2[:, 2]), 1)
        r_counts += valid_neg.sum(axis=0).astype(int)

    radius_votes = {r: int(c) for r, c in zip(radii, r_counts) if c > 0}
    best_radius = max(radius_votes, key=radius_votes.get) if radius_votes else None
    centers = find_sphere_centers(accumulator, min_bound, grid_voxel, grid_dims, cfg.get('vote_threshold', 0.3))
    return centers, best_radius


def align_femur_head_pure_geometric(source_pcd, target_pcd, voxel_size):
    """
    股骨头专用的纯几何特征配准算法。
    
    原理：
    1. 提取两者的光滑皮质骨面，过滤掉骨折断面的干扰。
    2. 使用 3D 霍夫变换分别计算患侧股骨头和健侧模板的球心。
    3. 球心对齐（平移）：计算平移向量使两个球心重合。
    4. 质心对齐（旋转）：以球心为支点，利用两者质心形成的向量叉乘计算旋转矩阵，使断裂面的朝向一致。
    
    参数:
        source_pcd: 患侧股骨头点云。
        target_pcd: 健侧模板点云。
    返回:
        T: 组合了平移与旋转的 4x4 初始配准矩阵。
    """
    source_clean = extract_smooth_surface(source_pcd, voxel_size, top_k_percent=CURVATURE_PERCENTILE)
    target_clean = extract_smooth_surface(target_pcd, voxel_size, top_k_percent=CURVATURE_PERCENTILE)
    source_use = source_clean if len(source_clean.points) > 50 else source_pcd
    target_use = target_clean if len(target_clean.points) > 50 else target_pcd
    
    force_estimate_normals(source_use)
    force_estimate_normals(target_use)

    tgt_centers, tgt_radius = hough_transform_sphere_detection(target_use, voxel_size)
    if tgt_radius is not None:
        src_centers, src_radius = hough_transform_sphere_detection(source_use, voxel_size, fixed_radius=tgt_radius, radius_tolerance=3.0)
    else:
        src_centers, src_radius = hough_transform_sphere_detection(source_use, voxel_size)
        
    c_src = np.array(src_centers[0]) if len(src_centers) > 0 else np.array(source_use.get_center())
    c_tgt = np.array(tgt_centers[0]) if len(tgt_centers) > 0 else np.array(target_use.get_center())
    
    vec_src = np.mean(np.asarray(source_use.points) + (c_tgt - c_src), axis=0) - c_tgt
    vec_tgt = np.mean(np.asarray(target_use.points), axis=0) - c_tgt
    vec_src /= (np.linalg.norm(vec_src) + 1e-6)
    vec_tgt /= (np.linalg.norm(vec_tgt) + 1e-6)
    
    v = np.cross(vec_src, vec_tgt)
    s = np.linalg.norm(v)
    c = np.dot(vec_src, vec_tgt)
    if s < 1e-6: R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
        
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = c_tgt - np.dot(R, c_src)
    return T

# ================= 股骨干与碎片配准 =================

def get_target_roi_clusters(current_target, voxel_size):
    """
    将目标模板点云划分为多个局部的感兴趣区域 (ROI)。
    
    原理：
    对于游离的小碎骨，在全局骨骼上跑 RANSAC 非常容易匹配到错误的相似结构（局部最优陷阱）。
    使用 DBSCAN 算法按空间距离将目标点云切分为若干个聚类块（ROI），
    后续配准时强制碎片去分别匹配各个 ROI，能极大提高小碎片的匹配成功率。
    """
    points = np.asarray(current_target.points)
    if len(points) < 200: return [current_target]
    db = DBSCAN(eps=voxel_size*4.0, min_samples=30).fit(points)
    clusters = [current_target.select_by_index(np.where(db.labels_==k)[0]) for k in sorted(set(db.labels_)) if k != -1]
    clusters = [c for c in clusters if len(c.points) > 200]
    clusters.sort(key=lambda x: len(x.points), reverse=True)
    return clusters if clusters else [current_target]


def execute_robust_ransac(source, target, source_fpfh, target_fpfh, voxel_size, ransac_n=3, label=""):
    """
    鲁棒的多重 RANSAC 全局粗配准。
    
    原理：
    RANSAC 是一种基于随机采样的匹配算法，结果具有一定随机性。
    该函数执行指定次数的 RANSAC 尝试，利用 FPFH 特征寻找对应点，并保留 Fitness 最高的一次作为最终结果。
    注意：为了避免因骨表面特征相似度过高导致匹配点骤降，已禁用 mutual_filter。
    """
    best_result, best_fitness = None, -1.0
    for attempt in range(ROBUST_RANSAC_CONFIG['max_attempts']):
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh, False, voxel_size * 1.5,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), ransac_n,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.0)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 0.999))
        if result.fitness > best_fitness:
            best_fitness, best_result = result.fitness, result
        if result.fitness >= ROBUST_RANSAC_CONFIG['good_fitness']: break
    return best_result


def align_partial_femur_shaft(source_shaft, target_shaft, voxel_size, ref_head_center=None):
    """
    股骨干碎片专用的防错配准策略。
    
    原理：
    股骨干是规则的圆柱体，传统配准极易发生“上下倒置”、“前后翻转”以及“沿圆柱面平移滑动”。
    此算法的防御策略流：
    1. 截取 Z 轴特征：仅截取患侧的近端（含断裂面）和健侧的远端作为特征 ROI，避免圆柱体干扰。
    2. 局部 PCA 对齐：对齐主轴，并施加强制姿态约束 (R[2,2]>=0 防止倒置, R[1,1]>=0 防止翻转)。
    3. RANSAC 微调：在姿态正确的约束下利用 FPFH 特征进一步贴合。
    4. 严苛 ICP：使用 Point-to-Point（禁用允许滑动的 Point-to-Plane），依靠断面的参差特征锁死 Z 轴。
    """
    source_clean = extract_smooth_surface(source_shaft, voxel_size, top_k_percent=CURVATURE_PERCENTILE)
    target_clean = extract_smooth_surface(target_shaft, voxel_size, top_k_percent=CURVATURE_PERCENTILE)
    source_use = source_clean if len(source_clean.points) > 100 else source_shaft
    target_use = target_clean if len(target_clean.points) > 100 else target_shaft
    
    target_points = np.asarray(target_use.points)
    target_roi = extract_outer_surface(target_use.select_by_index(np.argsort(target_points[:, 2])[:max(int(len(target_points) * TARGET_SHAFT_ROI_RATIO), 100)]))
    source_points = np.asarray(source_use.points)
    source_roi = source_use.select_by_index(np.argsort(source_points[:, 2])[-max(int(len(source_points) * SHAFT_FEATURE_ROI_RATIO), 100):])
    
    force_estimate_normals(source_roi); force_estimate_normals(target_roi)
    
    _, src_eigenvectors, _ = compute_pca(source_roi)
    _, tgt_eigenvectors, _ = compute_pca(target_roi)
    if np.dot(np.cross(tgt_eigenvectors[:, 0], tgt_eigenvectors[:, 1]), tgt_eigenvectors[:, 2]) < 0:
        tgt_eigenvectors[:, 2] *= -1
        
    best_pca_rmse, T_pca = float('inf'), np.eye(4)
    tree_tgt_roi = KDTree(np.asarray(target_roi.points))
    
    for flip_pc1, flip_pc2 in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        V_src = src_eigenvectors.copy()
        V_src[:, 0] *= flip_pc1; V_src[:, 1] *= flip_pc2; V_src[:, 2] = np.cross(V_src[:, 0], V_src[:, 1])
        R = tgt_eigenvectors @ V_src.T
        if R[2, 2] < 0 or R[1, 1] < 0: continue
        
        t = np.mean(np.asarray(target_roi.points), axis=0) - R @ np.mean(np.asarray(source_roi.points), axis=0)
        src_transformed = (R @ np.asarray(source_roi.points).T).T + t
        rmse = np.sqrt(np.mean(tree_tgt_roi.query(src_transformed[::max(1, len(src_transformed)//500)], k=1)[0] ** 2))
        if rmse < best_pca_rmse:
            best_pca_rmse, T_pca[:3, :3], T_pca[:3, 3] = rmse, R, t
            
    source_roi_pca = copy.deepcopy(source_roi).transform(T_pca)
    force_estimate_normals(source_roi_pca)
    
    source_fpfh = compute_fpfh_feature(source_roi_pca, voxel_size)
    target_fpfh = compute_fpfh_feature(target_roi, voxel_size)
    
    result = execute_robust_ransac(source_roi_pca, target_roi, source_fpfh, target_fpfh, voxel_size, ransac_n=3)
    best_ransac_result = result if (result and result.transformation[2, 2] >= 0 and result.transformation[1, 1] >= 0) else None

    T_ransac = best_ransac_result.transformation if best_ransac_result else np.eye(4)
    T_combined = T_ransac @ T_pca
    
    icp_rough = o3d.pipelines.registration.registration_icp(
        source_roi, target_roi, voxel_size * 3.0, T_combined,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
    
    icp_fine = o3d.pipelines.registration.registration_icp(
        source_roi, target_roi, voxel_size * 1.5, icp_rough.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
        
    best_ransac_fitness = best_ransac_result.fitness if best_ransac_result else -1.0
    final_trans = T_combined if icp_fine.fitness < best_ransac_fitness or icp_fine.fitness < 0.1 else icp_fine.transformation
    return final_trans, source_roi

# ================= 顶层配准策略 =================

def register_fragments_with_strategy(source_pcds, source_names, target_down, registration_order, classified, voxel_size, mask_radius, verbose=False):
    """
    骨折配准系统的核心大脑：分层分类策略调度器。
    
    原理：
    接收所有已排好优先级的碎片，按类型（头、干、碎片）分配相应的专业配准算法，最后统一利用 ICP 精细扣合。
    防碰撞保护：每次成功配准一个碎片后，利用 mask_target_region 在模板上“挖掉”其对应的物理空间，
    确保下一个碎片不会抢占已被复位的区域。
    
    参数:
        source_pcds: 源碎片点云列表。
        source_names: 碎片名称列表（用于日志追踪）。
        target_down: 健侧镜像目标模板（预处理后）。
        registration_order: 从 utils 算出的最佳配准先后顺序。
        classified: 分类字典（标记索引属于哪个大类）。
        voxel_size: 体素大小。
        mask_radius: 动态遮罩的剔除半径（股骨头等核心部件会赋予更大半径）。
        verbose: 是否打印详细日志。
    返回:
        返回所有的结果集合，包括变换矩阵、评估指标、已变换点云以及有效配准表面点云。
    """
    current_target = copy.deepcopy(target_down)
    all_results, all_metrics, transformed_fragments, processed_clouds = [], [], [], []
    target_normals_dirty = not current_target.has_normals()
    ref_head_center = None
    total = len(registration_order)

    for step, idx in enumerate(registration_order):
        source, name = source_pcds[idx], source_names[idx]
        frag_type = 'head' if idx in classified['head'] else ('shaft' if idx in classified['shaft'] else 'other')
        type_label = {'head': '股骨头', 'shaft': '股骨干', 'other': '碎片'}[frag_type]
        if verbose:
            print(f"   [{step+1}/{total}] 配准 {type_label}: {name}")
        init_trans, source_effective = np.eye(4), None
        
        if frag_type == 'head':
            source_effective = extract_smooth_surface(source, voxel_size, top_k_percent=CURVATURE_PERCENTILE)
            init_trans = align_femur_head_pure_geometric(source, current_target, voxel_size)
            tgt_centers, _ = hough_transform_sphere_detection(extract_smooth_surface(current_target, voxel_size, top_k_percent=CURVATURE_PERCENTILE), voxel_size)
            ref_head_center = np.array(tgt_centers[0]) if len(tgt_centers) > 0 else np.array(current_target.get_center())
        elif frag_type == 'shaft':
            init_trans, source_effective = align_partial_femur_shaft(source, current_target, voxel_size, ref_head_center)
        else:
            source_smooth = extract_smooth_surface(source, voxel_size, top_k_percent=CURVATURE_PERCENTILE)
            source_effective = source_smooth
            s_fpfh = compute_fpfh_feature(source_smooth, voxel_size)
            best_roi_res, best_roi_fit = None, -1
            
            for target_roi in get_target_roi_clusters(current_target, voxel_size):
                t_fpfh = compute_fpfh_feature(target_roi, voxel_size)
                roi_res = execute_robust_ransac(source_smooth, target_roi, s_fpfh, t_fpfh, voxel_size, label="ROI")
                if roi_res:
                    icp_check = o3d.pipelines.registration.registration_icp(
                        source_smooth, target_roi, voxel_size, roi_res.transformation,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10))
                    if icp_check.fitness > best_roi_fit:
                        best_roi_fit, best_roi_res = icp_check.fitness, roi_res
            if best_roi_res: init_trans = best_roi_res.transformation

        if not source.has_normals(): force_estimate_normals(source)
        if target_normals_dirty or not current_target.has_normals():
            force_estimate_normals(current_target)
            target_normals_dirty = False

        icp_rough = o3d.pipelines.registration.registration_icp(
            source, current_target, voxel_size * 2.0, init_trans,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
        icp_final = o3d.pipelines.registration.registration_icp(
            source, current_target, voxel_size * 0.5, icp_rough.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
        
        source_transformed = copy.deepcopy(source).transform(icp_final.transformation)
        metrics = evaluate_registration_detailed(source_transformed, current_target, voxel_size * 2.0)
        if verbose:
            print(f"        Fitness: {metrics['fitness']:.4f} | RMSE: {metrics['rmse']:.4f} | Hausdorff: {metrics['hausdorff']:.2f} mm")

        effective_transformed = copy.deepcopy(source_effective).transform(icp_final.transformation) if source_effective else source_transformed
        processed_clouds.append({'name': name, 'type': frag_type, 'cloud': effective_transformed})
        all_results.append(icp_final); all_metrics.append(metrics); transformed_fragments.append(source_transformed)
        
        if step < len(registration_order) - 1:
            current_target = mask_target_region(current_target, source_transformed, mask_radius * (1.2 if frag_type == 'head' else 1.0))
            target_normals_dirty = True

    return all_results, all_metrics, transformed_fragments, processed_clouds


def run_registration_pipeline(source_pcds_raw, target_raw, source_paths, voxel_size, verbose=True):
    """
    全自动点云配准流水线总控制器。
    
    工作流：
    1. 预处理：下采样过滤噪点，计算统一的表面法向。
    2. 健侧处理：对健侧目标进行沿中轴镜像，作为复位基准模板。
    3. 全局落位：合并所有碎片跑一轮 PCA 粗配准，拉近患侧与健侧的距离。
    4. 智能调度：对碎片进行解包、体积计算和分类排序。
    5. 核心配准：送入策略引擎执行精准扣合，收集配准结果。
    6. 统计评估：汇总各项误差评价指标（RMSE，Hausdorff距离等），打包含全部数据的字典返回主程序。
    """
    if verbose: print("   预处理点云 (下采样 + 去噪)...")
    source_pcds = [preprocess_point_cloud(copy.deepcopy(pcd), voxel_size) for pcd in source_pcds_raw]
    target_down = preprocess_point_cloud(copy.deepcopy(target_raw), voxel_size)
    empty_sources = [
        source_paths[i] if i < len(source_paths) else f"source index {i}"
        for i, pcd in enumerate(source_pcds)
        if pcd.is_empty()
    ]
    if empty_sources:
        raise ValueError("No usable points after preprocessing for source file(s):\n" + "\n".join(empty_sources))
    if target_down.is_empty():
        raise ValueError("No usable points after preprocessing for target point cloud")
    if verbose: print("   镜像目标点云 (健侧 -> 患侧)...")
    target_mirrored = mirror_point_cloud(target_down, voxel_size, 'x')
    if target_mirrored.is_empty():
        raise ValueError("No usable points after mirroring target point cloud")

    if verbose: print("   PCA 全局粗配准...")
    combined_source = merge_point_clouds(source_pcds)
    if combined_source.is_empty():
        raise ValueError("No usable source points for PCA alignment")
    pca_transform = align_by_pca(combined_source, target_mirrored, verbose=verbose)
    source_pcds = apply_transform_to_all_fragments(source_pcds, pca_transform)
    
    classified, order = classify_and_order_fragments(source_paths, source_pcds)
    original_names = [os.path.basename(p) for p in source_paths]
    if verbose:
        order_names = [original_names[i] for i in order]
        print(f"   碎片分类完成, 配准顺序: {' -> '.join(order_names)}")
        print(f"   逐碎片精配准 (共 {len(order)} 个)...")

    results, metrics, transformed, processed_clouds = register_fragments_with_strategy(
        source_pcds, original_names, target_mirrored, order, classified, voxel_size, voxel_size*1.5,
        verbose=verbose)
    
    all_dists = np.concatenate([m['distances'] for m in metrics])
    overall_metrics = {
        'mean_rmse': np.mean([m['rmse'] for m in metrics]),
        'mean_fitness': np.mean([m['fitness'] for m in metrics]),
        'mean_distance': np.mean(all_dists),
        'median_distance': np.median(all_dists),
        'max_hausdorff': np.max([m['hausdorff'] for m in metrics]),
        'p95_distance': np.percentile(all_dists, 95),
    }
    return {
        'voxel_size': voxel_size, 'transformed': transformed, 'metrics': metrics,
        'overall': overall_metrics, 'target_mirrored': target_mirrored, 'order': order,
        'original_names': original_names, 'processed_clouds': processed_clouds
    }
