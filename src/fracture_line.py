import copy
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree

from config import FRACTURE_LINE_CONFIG
from utils import force_estimate_normals

def detect_boundary_points_centroid_offset(pcd, k=30, ratio=0.4):
    """
    基于局部邻域质心偏移法检测点云的边界点。
    
    原理：
    对于表面内部的点，其局部 K 近邻的质心通常与该点自身非常接近；
    而对于位于边界的点，其 K 近邻的质心会明显偏向内部，导致质心与该点的距离变大。
    
    参数:
        pcd: 待处理的点云（通常是单个骨折碎片）。
        k: K 近邻搜索的邻居数量。较大的 K 值能捕捉更宏观的边缘特征。
        ratio: 偏移量阈值系数。当某点的偏移量大于 (平均邻居距离 * ratio) 时，判定为边界点。
        
    返回:
        boundary_indices: 提取出的边界点在原始点云中的索引数组。
    """
    points = np.asarray(pcd.points)
    n_points = len(points)
    if n_points < k + 1: k = max(3, n_points - 1)

    kdtree = KDTree(points)
    _, nn_idx = kdtree.query(points, k=k + 1)
    nn_idx = nn_idx[:, 1:]

    neighbor_pts = points[nn_idx]
    centroids = np.mean(neighbor_pts, axis=1)
    diffs = neighbor_pts - points[:, None, :]
    avg_dists = np.mean(np.linalg.norm(diffs, axis=2), axis=1)
    offsets = np.linalg.norm(points - centroids, axis=1)

    return np.where(offsets > avg_dists * ratio)[0].astype(int)

def project_boundary_to_target(boundary_pcd, target_pcd, max_distance=100.0):
    """
    将提取出的碎片边缘点投影（吸附）到健侧模板点云的表面上。
    
    原理：
    利用 KDTree 搜索边界点在目标模板上的最近邻点，以此作为投影点。这有助于将各个碎片
    粗糙且参差不齐的断茬边缘映射到一个平滑、连续的参考骨骼表面上，形成更加规整的骨折线。
    
    参数:
        boundary_pcd: 提取出的碎片边界点云。
        target_pcd: 目标点云（即镜像对齐后的健侧模板）。
        max_distance: 允许的最大投影距离阈值。距离模板过远的点将被视为离群噪点被丢弃。
        
    返回:
        snapped_pcd: 投影到模板表面后的点云对象（同时继承了目标模板表面的法向量）。
    """
    if len(boundary_pcd.points) == 0: return o3d.geometry.PointCloud()
    boundary_pts, target_pts = np.asarray(boundary_pcd.points), np.asarray(target_pcd.points)
    target_tree = KDTree(target_pts)
    dists, indices = target_tree.query(boundary_pts, k=1)
    
    valid_mask = dists.flatten() <= max_distance
    snapped_pcd = o3d.geometry.PointCloud()
    snapped_pcd.points = o3d.utility.Vector3dVector(target_pts[indices.flatten()[valid_mask]])
    
    if target_pcd.has_normals():
        snapped_pcd.normals = o3d.utility.Vector3dVector(np.asarray(target_pcd.normals)[indices.flatten()[valid_mask]])
    return snapped_pcd

def denoise_fracture_line(pcd, config=None):
    """
    对投影后的骨折线点云进行去噪处理。
    
    原理：
    边缘提取和投影过程中可能会混入一些孤立的噪点。根据配置参数，使用统计滤波 
    (Statistical Outlier Removal) 或 半径滤波 (Radius Outlier Removal) 来剔除这些游离点。
    
    参数:
        pcd: 待去噪的骨折线点云。
        config: 包含去噪方法及对应超参数的配置字典。
        
    返回:
        清理完成后的骨折线点云。
    """
    if config is None: config = FRACTURE_LINE_CONFIG
    if len(np.asarray(pcd.points)) < 5: return pcd
    method = config.get('denoise_method', 'none')
    
    if method == 'statistical':
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=config.get('statistical_nb', 30), std_ratio=config.get('statistical_std', 1.5))
        return pcd.select_by_index(ind)
    elif method == 'radius':
        _, ind = pcd.remove_radius_outlier(nb_points=config.get('radius_nb', 10), radius=config.get('radius_r', 2.0))
        return pcd.select_by_index(ind)
    return pcd

def extract_fracture_lines(transformed_fragments, target_mirrored, fragment_names=None, config=None):
    """
    骨折线提取的核心主干流程函数。
    
    工作流：
    1. 遍历所有已配准对齐的骨折碎片。
    2. 对每个碎片提取其表面边界点（即骨折断面边缘）。
    3. 将这些边界点投影吸附到完整的健侧模板表面上。
    4. 对吸附后的线条点云进行空间去噪。
    5. 将所有碎片的边缘点合并成一个代表整体骨折线的单一大型点云。
    
    参数:
        transformed_fragments: 已经完成坐标变换（配准到健侧位置）的碎片点云列表。
        target_mirrored: 镜像后的健侧模板点云。
        fragment_names: 碎片对应的名称列表（用于可能的日志输出）。
        config: 控制提取、投影和去噪的配置字典。
        
    返回:
        fracture_lines: 包含所有骨折点及其法向量的合并总点云（红色）。
        per_fragment_lines: 每个单独碎片的骨折线点云构成的列表。
    """
    if config is None: config = FRACTURE_LINE_CONFIG
    k, ratio, snap_dist = config.get('boundary_k', 60), config.get('centroid_ratio', 0.4), config.get('snap_max_distance', 100.0)
    per_fragment_lines, all_line_points = [], []

    for i, fragment in enumerate(transformed_fragments):
        if not fragment.has_normals(): force_estimate_normals(fragment)
        boundary_idx = detect_boundary_points_centroid_offset(fragment, k=k, ratio=ratio)
        boundary_pcd = fragment.select_by_index(boundary_idx)
        
        if len(boundary_pcd.points) == 0:
            per_fragment_lines.append(o3d.geometry.PointCloud()); continue

        snapped_pcd = project_boundary_to_target(boundary_pcd, target_mirrored, max_distance=snap_dist)
        denoised_pcd = denoise_fracture_line(snapped_pcd, config)
        
        per_fragment_lines.append(denoised_pcd)
        if len(denoised_pcd.points) > 0: all_line_points.append(denoised_pcd)

    fracture_lines = o3d.geometry.PointCloud()
    if all_line_points:
        fracture_lines.points = o3d.utility.Vector3dVector(np.vstack([np.asarray(p.points) for p in all_line_points]))
        try: fracture_lines.normals = o3d.utility.Vector3dVector(np.vstack([np.asarray(p.normals) for p in all_line_points]))
        except: pass
        fracture_lines.paint_uniform_color([1, 0, 0])
    return fracture_lines, per_fragment_lines

def create_spheres_from_pcd(pcd, radius, color):
    """
    将单薄的点云数据实体化，转换为由小球体组成的 3D 三角网格。
    
    原理：
    点云在 3D 渲染时没有物理厚度，作为“线”来展示视觉效果通常较差且不易观察。
    此函数以点云中的每个点为球心，利用 numpy 向量化广播技术，一次性快速生成多个
    带有实体厚度的小球网格。内置最大点数限制以防内存溢出。
    
    参数:
        pcd: 需要实体化的输入点云。
        radius: 生成每个实体小球的半径。
        color: 小球网格的统一 RGB 颜色数组。
        
    返回:
        combined_mesh: 拼接合并完成的所有球体组成的单一三角网格对象。
    """
    if pcd.is_empty(): return o3d.geometry.TriangleMesh()
    sphere_proto = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=8)
    base_vertices, base_triangles = np.asarray(sphere_proto.vertices), np.asarray(sphere_proto.triangles)

    points = np.asarray(pcd.points)
    max_spheres = FRACTURE_LINE_CONFIG.get('vis_max_spheres', 8000)
    if len(points) > max_spheres:
        points = points[np.random.choice(len(points), size=max_spheres, replace=False)]

    all_vertices = (base_vertices[None, :, :] + points[:, None, :]).reshape(-1, 3)
    triangle_offsets = (np.arange(len(points)) * base_vertices.shape[0])[:, None, None]
    all_triangles = (base_triangles[None, :, :] + triangle_offsets).reshape(-1, 3)

    combined_mesh = o3d.geometry.TriangleMesh()
    combined_mesh.vertices = o3d.utility.Vector3dVector(all_vertices)
    combined_mesh.triangles = o3d.utility.Vector3iVector(all_triangles.astype(np.int32))
    combined_mesh.paint_uniform_color(color)
    return combined_mesh

def visualize_fracture_lines(target, fracture_line_pcd, per_fragment_lines=None, fragment_names=None, output_path=None):
    """
    可视化最终的骨骼模型与高亮实体骨折线。
    
    原理：
    将健侧模板渲染为平淡的灰色基底，将提取的骨折线渲染为醒目的红色实体圆点覆盖其上。
    为防止带体积的红色骨折线球体与骨骼表面发生深度冲突（穿模，Z-fighting），
    利用法向量将所有骨折点沿法向向外侧轻微偏移（膨胀偏移量由 vis_offset 控制）。
    
    参数:
        target: 作为视觉背景的健侧模板点云。
        fracture_line_pcd: 总体骨折线点云。
        per_fragment_lines: （备用参数）按碎片分类的骨折线列表。
        fragment_names: （备用参数）碎片名称列表。
        output_path: 结果保存路径。如果提供，将把合并了骨骼和染色骨折线的模型保存到本地。
    """
    if fracture_line_pcd.is_empty(): return
    offset_dist = FRACTURE_LINE_CONFIG.get('vis_offset', 0.5)
    line_radius = FRACTURE_LINE_CONFIG.get('vis_line_radius', 0.6) 

    target_vis = copy.deepcopy(target)
    target_vis.paint_uniform_color([0.6, 0.6, 0.6])
    
    line_temp = copy.deepcopy(fracture_line_pcd)
    if line_temp.has_normals():
        line_temp.points = o3d.utility.Vector3dVector(np.asarray(line_temp.points) + np.asarray(line_temp.normals) * offset_dist)
    
    line_mesh = create_spheres_from_pcd(line_temp, radius=line_radius, color=[1.0, 0.0, 0.0])
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="最终结果: 红色大圆点骨折线", width=1280, height=960)
    vis.add_geometry(target_vis)
    vis.add_geometry(line_mesh)
    vis.get_render_option().background_color = np.array([0.0, 0.0, 0.0])
    vis.get_render_option().point_size = FRACTURE_LINE_CONFIG.get('vis_point_size', 2.0)
    
    if output_path:
        merged = target_vis + copy.deepcopy(line_temp).paint_uniform_color([1.0, 0.0, 0.0])
        o3d.io.write_point_cloud(output_path, merged)
        
    vis.run()
    vis.destroy_window()