import os
import copy
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neighbors import KDTree
from config import MIN_DISPLAY_FITNESS
from geometry_classifier import GeometryClassifier, LABEL_HEAD, LABEL_SHAFT


_GEOMETRY_CLASSIFIER = GeometryClassifier()

# ================= 基础点云操作 =================

def force_estimate_normals(pcd):
    """
    强制计算点云的法向量并统一朝向。
    
    原理：
    配准算法（如 ICP、FPFH 特征提取、霍夫变换）高度依赖准确的表面法向量。
    该函数使用 KNN（K=30）搜索邻域来计算法向量，并使其朝向一致，防止后续算法因缺乏法向量而崩溃。
    
    参数:
        pcd: 待处理的点云对象。
    返回:
        带有法向量信息的点云对象。
    """
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd.orient_normals_consistent_tangent_plane(100)
    return pcd

def load_mesh_as_pcd(filepath, number_of_points=100000):
    """
    加载 3D 模型文件并统一转换为点云格式。
    
    原理：
    医学图像分割出的通常是 STL 网格模型。为了进行点云配准，需要将网格转换为点云。
    这里使用泊松磁盘采样（Poisson disk sampling），确保在网格表面均匀撒点，避免点云局部聚集。
    
    参数:
        filepath: 文件路径（支持 .stl, .ply, .pcd 等）。
        number_of_points: 当读取网格时，采样的目标点数。
    返回:
        加载或采样后的 Open3D 点云对象。
    """
    if not os.path.exists(filepath):
        print(f"[错误] 文件不存在: {filepath}")
        return o3d.geometry.PointCloud()
    ext = os.path.splitext(filepath)[-1].lower()
    if ext == '.stl':
        mesh = o3d.io.read_triangle_mesh(filepath)
        if not mesh.has_triangles(): return o3d.geometry.PointCloud()
        return mesh.sample_points_poisson_disk(number_of_points=number_of_points)
    else:
        return o3d.io.read_point_cloud(filepath)

def calculate_adaptive_voxel_size(pcd, factor=60.0):
    """
    根据点云的实际物理尺寸自适应计算体素大小（Voxel Size）。
    
    原理：
    不同患者的骨骼尺寸差异较大。通过计算骨骼包围盒对角线长度并除以一个常数因子，
    可以保证无论骨骼大小，下采样后的相对分辨率始终保持一致，提高算法的鲁棒性。
    
    参数:
        pcd: 完整的骨骼点云（通常是健侧模板）。
        factor: 划分因子，值越大，体素越小（分辨率越高）。
    返回:
        计算出的自适应体素大小 (float)。
    """
    aabb = pcd.get_axis_aligned_bounding_box()
    diagonal_length = np.linalg.norm(aabb.get_extent())
    voxel_size = diagonal_length / factor
    return voxel_size

def preprocess_point_cloud(pcd, voxel_size):
    """
    点云预处理流水线：体素下采样 -> 计算法向量 -> 统计滤波去噪。
    
    原理：
    减小点云密度以加速配准计算，同时去除 CT 分割或表面重建时产生的离散噪点。
    
    参数:
        pcd: 原始点云。
        voxel_size: 体素网格大小。
    返回:
        预处理后的干净点云。
    """
    if pcd.is_empty(): return pcd
    pcd_down = pcd.voxel_down_sample(voxel_size)
    force_estimate_normals(pcd_down)
    cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd_down.select_by_index(ind)

def compute_fpfh_feature(pcd, voxel_size):
    """
    提取快速点特征直方图（FPFH）特征。
    
    原理：
    为 RANSAC 全局配准提供特征描述子。FPFH 描述了点及其邻域表面的几何结构（曲率、法向变化等），
    使得算法可以在未知初始位姿的情况下找到匹配对。
    
    参数:
        pcd: 输入点云。
        voxel_size: 体素大小（用于决定特征搜索半径）。
    返回:
        FPFH 特征对象。
    """
    radius_feature = voxel_size * 5
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

def mirror_point_cloud(pcd, voxel_size, mirror_axis='x'):
    """
    沿指定坐标轴镜像点云。
    
    原理：
    在骨科手术规划中，通常将健康的对侧骨骼（健侧）进行镜像，作为粉碎性骨折（患侧）复位的模板。
    镜像后会导致法向量方向翻转，因此必须重新计算法向量。
    
    参数:
        pcd: 健侧骨骼点云。
        voxel_size: 体素大小（保留接口，这里暂未使用，但有助于规范化）。
        mirror_axis: 镜像轴（'x', 'y' 或 'z'）。
    返回:
        镜像后的健侧模板点云。
    """
    points = np.asarray(pcd.points).copy()  
    if len(points) == 0: return o3d.geometry.PointCloud()
    if mirror_axis == 'x': points[:, 0] = -points[:, 0]
    elif mirror_axis == 'y': points[:, 1] = -points[:, 1]
    elif mirror_axis == 'z': points[:, 2] = -points[:, 2]
    mirrored_pcd = o3d.geometry.PointCloud()
    mirrored_pcd.points = o3d.utility.Vector3dVector(points)
    if pcd.has_colors(): mirrored_pcd.colors = pcd.colors
    force_estimate_normals(mirrored_pcd)
    return mirrored_pcd

def extract_smooth_surface(pcd, voxel_size, top_k_percent=0.95, mode='smooth'):
    """
    基于主成分分析（PCA）和曲率分布自适应提取光滑骨表面。
    
    原理：
    骨折断端（断茬）极其粗糙，会干扰配准。此函数计算局部协方差矩阵的特征值来估算曲率，
    过滤掉高曲率的点，仅保留皮质骨的光滑外表面，从而大幅提升 RANSAC 和 ICP 的成功率。
    
    参数:
        pcd: 输入的骨折碎片点云。
        voxel_size: 体素大小（用于决定搜索半径）。
        top_k_percent: 保留点的比例阈值。
        mode: 'smooth' 提取光滑面；'feature' 提取高曲率断裂面。
    返回:
        过滤后的光滑表面点云子集。
    """
    points = np.asarray(pcd.points)
    if len(points) < 10: return pcd
    k_neighbors = min(max(20, int(voxel_size * 10)), len(points) - 1)
    if k_neighbors < 5: return pcd
    kdtree = KDTree(points)
    _, nn_idx = kdtree.query(points, k=k_neighbors + 1)
    neighbor_pts = points[nn_idx[:, 1:]]
    centered = neighbor_pts - np.mean(neighbor_pts, axis=1, keepdims=True)
    covs = np.einsum('nki,nkj->nij', centered, centered) / max(k_neighbors - 1, 1)
    eigvals = np.linalg.eigvalsh(covs)
    eig_sums = np.sum(eigvals, axis=1)
    curvatures = np.where(eig_sums > 1e-12, eigvals[:, 0] / eig_sums, 1.0)
    
    threshold = np.percentile(curvatures, top_k_percent * 100) if mode == 'smooth' else np.percentile(curvatures, (1 - top_k_percent) * 100)
    selected_idx = np.where(curvatures <= threshold)[0] if mode == 'smooth' else np.where(curvatures >= threshold)[0]
    return pcd.select_by_index(selected_idx) if len(selected_idx) >= 50 else pcd

def extract_outer_surface(pcd):
    """
    使用多视角隐点移除算法（HPR）提取骨骼的绝对外表面。
    
    原理：
    医学 CT 分割模型常包含内部的松质骨网格或骨髓腔噪点。
    从 6 个正交视角（上下左右前后）向内看，只保留摄像机能看到的外壳点，
    从而彻底挖空内部噪点，防止碎片配准时错误地卡入骨骼内部。
    
    参数:
        pcd: 可能包含内部结构的输入点云。
    返回:
        只有纯外壳表面的干净点云。
    """
    if pcd.is_empty() or len(pcd.points) < 100: return pcd
    center = pcd.get_center()
    diagonal = np.linalg.norm(pcd.get_axis_aligned_bounding_box().get_extent())
    camera_distance = diagonal * 2.0
    hpr_radius = diagonal * 100
    
    camera_positions = [
        center + np.array([camera_distance, 0, 0]), center + np.array([-camera_distance, 0, 0]),
        center + np.array([0, camera_distance, 0]), center + np.array([0, -camera_distance, 0]),
        center + np.array([0, 0, camera_distance]), center + np.array([0, 0, -camera_distance]),
    ]
    all_visible_indices = set()
    for cam_pos in camera_positions:
        try:
            _, pt_map = pcd.hidden_point_removal(cam_pos, hpr_radius)
            all_visible_indices.update(pt_map)
        except Exception: continue
        
    if len(all_visible_indices) < 100: return pcd
    pcd_surface = pcd.select_by_index(sorted(list(all_visible_indices)))
    _, ind = pcd_surface.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_clean = pcd_surface.select_by_index(ind)
    force_estimate_normals(pcd_clean)
    return pcd_clean

# ================= 评估与数据处理 =================

def mask_target_region(target, transformed_fragment, mask_radius):
    """
    动态遮罩（挖去）健侧模板上已被配准占据的区域。
    
    原理：
    在按顺序拼接多个碎片时，为了防止后面的碎片错误地跑到前面已拼接碎片的位置（抢占地盘），
    在每次配准成功后，将模板上对应位置的点暂时删除。
    
    参数:
        target: 当前的健侧模板点云。
        transformed_fragment: 刚刚完成配准的碎片点云。
        mask_radius: 遮罩影响的剔除半径。
    返回:
        更新后的目标模板点云（被挖去了一块）。
    """
    target_points = np.asarray(target.points)
    fragment_points = np.asarray(transformed_fragment.points)
    tree = KDTree(fragment_points)
    dists, _ = tree.query(target_points, k=1)
    return target.select_by_index(np.where(dists.flatten() > mask_radius)[0])

def evaluate_registration_detailed(source, target, threshold):
    """
    计算配准的详细评估指标。
    
    参数:
        source: 变换后的碎片点云。
        target: 目标参考点云。
        threshold: 距离内点判定阈值。
    返回:
        字典包含: fitness(重合度), rmse(均方根误差), hausdorff(最大偏离距离) 以及 所有点对距离数组。
    """
    eval_result = o3d.pipelines.registration.evaluate_registration(source, target, threshold)
    dists = np.asarray(source.compute_point_cloud_distance(target))
    return {
        "fitness": eval_result.fitness, "rmse": eval_result.inlier_rmse,
        "hausdorff": np.max(dists) if len(dists) > 0 else 0.0, "distances": dists
    }

def classify_and_order_fragments(
    source_paths: List[str],
    source_pcds: List[o3d.geometry.PointCloud],
) -> Tuple[Dict[str, List[int]], List[int]]:
    """
    自动分类骨折碎片并决定配准优先级排序。
    
    原理：
    根据物理逻辑：大块骨骼（如股骨头、主要股骨干）决定了宏观框架，应优先配准；
    小块游离碎骨容易受局部陷阱干扰，应延后配准。
    此函数通过几何特征分类识别特殊结构，其余碎片按包围盒（OBB）体积从大到小排序。
    
    参数:
        source_paths: 碎片文件路径列表（保留接口兼容，不参与分类）。
        source_pcds: 对应的碎片点云列表（用于计算体积）。
    返回:
        classified (分类字典) 和 order (包含索引的最佳配准顺序列表)。
    """
    _ = source_paths
    classified = {'head': [], 'shaft': [], 'other': []}
    for i, pcd in enumerate(source_pcds):
        label = _GEOMETRY_CLASSIFIER.classify_point_cloud(pcd)
        if label == LABEL_HEAD:
            classified['head'].append(i)
        elif label == LABEL_SHAFT:
            classified['shaft'].append(i)
        else:
            classified['other'].append(i)
        
    def get_obb_volume(pcd: o3d.geometry.PointCloud) -> float:
        if pcd.is_empty() or len(pcd.points) < 4: return 0.0
        try: return pcd.get_oriented_bounding_box().volume()
        except: return np.prod(pcd.get_axis_aligned_bounding_box().get_extent())
        
    order = classified['head'] + classified['shaft']
    classified['other'].sort(key=lambda x: get_obb_volume(source_pcds[x]), reverse=True)
    order.extend(classified['other'])
    return classified, order

# ================= 可视化相关 =================

def _visualize_helper(sources, target, names, metrics, output_path, window_name):
    """
    内部渲染辅助函数。将所有组件赋予不同的颜色并进行 3D 渲染显示。
    如果 Fitness 过低（配准失败），则在视图中隐藏该碎片以防干扰视觉。
    """
    geoms = []
    if target:
        t_temp = copy.deepcopy(target)
        t_temp.paint_uniform_color([0.8, 0.8, 0.8])
        geoms.append(t_temp)
        
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]]
    for i, s in enumerate(sources):
        if metrics and i < len(metrics) and metrics[i].get('fitness', 0.0) < MIN_DISPLAY_FITNESS: continue
        s_temp = copy.deepcopy(s) 
        s_temp.paint_uniform_color(colors[i % len(colors)])
        geoms.append(s_temp)
        
    if output_path and geoms:
        merged = o3d.geometry.PointCloud()
        for g in geoms: merged += g
        o3d.io.write_point_cloud(output_path, merged)
    o3d.visualization.draw_geometries(geoms, window_name=window_name)

def visualize_fragments_only(sources, target_center=None, names=None, metrics=None, output_path=None):
    """可视化接口：仅显示拼接完成的碎片阵列（移除灰色背景模板）"""
    _visualize_helper(sources, None, names, metrics, output_path, "仅碎片配准结果 (无模板)")

def visualize_registration(sources, target, transformations=None, names=None, metrics=None, output_path=None):
    """可视化接口：显示碎片贴合在灰色健侧模板上的整体效果"""
    _visualize_helper(sources, target, names, metrics, output_path, "配准结果 (含模板)")

def plot_registration_evaluation(metrics_list, names_list, output_path=None):
    """
    绘制所有碎片配准误差的概率密度分布图 (KDE Plot)。
    
    原理：
    通过高斯核密度估计，直观展示碎片上每个点到模板表面的距离分布情况。
    曲线越靠左越陡峭，说明配准越完美；如果出现明显的右侧长尾，说明可能有翘起或变形。
    
    参数:
        metrics_list: 包含每个碎片评估数据的字典列表。
        names_list: 碎片名称列表（用于图例）。
        output_path: 图表输出保存路径。
    """
    n = len(metrics_list)
    if n == 0: return
    fig, axes = plt.subplots(n + 1, 1, figsize=(10, 4 * (n + 1)), constrained_layout=True)
    if n + 1 == 1: axes = [axes] 
    
    all_dists = np.concatenate([m['distances'] for m in metrics_list])
    for i, (m, name) in enumerate(zip(metrics_list, names_list)):
        ax = axes[i]
        d = m['distances']
        if len(d) > 0:
            lim = np.percentile(d, 99)
            d_plot = d[d <= lim]
            if len(d_plot) > 1:
                kde = stats.gaussian_kde(d_plot)
                x = np.linspace(0, lim, 200)
                ax.plot(x, kde(x), 'b-', lw=2)
                ax.fill_between(x, kde(x), alpha=0.3, color='b')
            mean_val = np.mean(d)
            ax.axvline(mean_val, color='r', linestyle='--')
            info = f"Fitness: {m['fitness']:.3f}\nRMSE: {m['rmse']:.3f}\nHausdorff: {m['hausdorff']:.2f}"
            ax.text(0.95, 0.95, info, transform=ax.transAxes, ha='right', va='top', bbox=dict(boxstyle='round', fc='w', alpha=0.8))
        ax.set_title(f"碎片误差分布: {name}")
        
    ax_all = axes[-1]
    if len(all_dists) > 0:
        lim = np.percentile(all_dists, 99)
        d_plot = all_dists[all_dists <= lim]
        if len(d_plot) > 1:
            kde = stats.gaussian_kde(d_plot)
            x = np.linspace(0, lim, 200)
            ax_all.plot(x, kde(x), 'g-', lw=2)
            ax_all.fill_between(x, kde(x), alpha=0.3, color='g')
        ax_all.axvline(np.mean(all_dists), color='r', linestyle='--')
    ax_all.set_title("整体复位质量分布")
    
    if output_path: plt.savefig(output_path, dpi=300)