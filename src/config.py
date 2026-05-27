import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# ================= 绘图与环境配置 =================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei'] 
plt.rcParams['axes.unicode_minus'] = False

# 固定随机种子
np.random.seed(40)
try:
    o3d.utility.random.seed(40)
except Exception:
    pass

# ================= 全局算法参数配置 =================
# 股骨头霍夫变换配置
HOUGH_CONFIG = {
    'min_radius': 18.0,      
    'max_radius': 30.0,      
    'radius_step': 1.0,      
    'grid_voxel_factor': 1.5,
    'vote_threshold': 0.3,   
    'sample_ratio': 3000,    
    'bidirectional': True,   
}

# 保留最光滑的前 N% 点
CURVATURE_PERCENTILE = 0.99

# 权重配置
WEIGHTS = {
    'rmse': 0.30, 'fitness': 0.35, 'mean_dist': 0.20,
    'p95_dist': 0.05, 'hausdorff': 0.10
}

# 股骨干配准参数
SHAFT_FEATURE_ROI_RATIO = 0.7
TARGET_SHAFT_ROI_RATIO = 0.7

# 可视化碎片过滤阈值
MIN_DISPLAY_FITNESS = 1e-6

# 鲁棒 RANSAC 配准参数
ROBUST_RANSAC_CONFIG = {
    'max_attempts': 5,           
    'good_fitness': 0.6,         
}

# 自适应 Voxel Size 划分因子
ADAPTIVE_VOXEL_FACTOR = 150

# 骨折线提取配置
FRACTURE_LINE_CONFIG = {
    'boundary_k': 60,              
    'centroid_ratio': 0.55,        
    'snap_max_distance': 5.0,      
    'denoise_method': 'radius',    
    'radius_nb': 10,               
    'radius_r': 8.0,               
    'vis_point_size': 5.0,         
    'vis_line_radius': 0.6,        
    'vis_offset': 0.5,             
    'vis_max_spheres': 8000,       
}