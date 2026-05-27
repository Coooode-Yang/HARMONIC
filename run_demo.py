import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import open3d as o3d
import matplotlib.pyplot as plt

# 导入拆分的模块
from config import ADAPTIVE_VOXEL_FACTOR, FRACTURE_LINE_CONFIG
from utils import (
    load_mesh_as_pcd, calculate_adaptive_voxel_size, 
    plot_registration_evaluation, visualize_registration, visualize_fragments_only
)
from registration import run_registration_pipeline
from fracture_line import extract_fracture_lines, visualize_fracture_lines

if __name__ == "__main__":
    # ================= 路径配置 =================
    BASE_DIR = Path(__file__).resolve().parent
    SOURCE_DIR = BASE_DIR / "data" / "unhealthy_side"
    TARGET_DIR = BASE_DIR / "data" / "healthy_side"
    OUTPUT_DIR = BASE_DIR / "results"
    
    SOURCE_FILES = [
        "data\\unhealthy_side\\Anonymous_Smoothed_Cyan 1_001.stl",
        "data\\unhealthy_side\\Anonymous_Smoothed_shaft_001.stl",
        "data\\unhealthy_side\\Anonymous_Smoothed_Green 5_001.stl",
        "data\\unhealthy_side\\Anonymous_Smoothed_head_001.stl",
    ] 
    TARGET_FILENAME = "Example.ply"
    
    SOURCE_PATHS = [str(SOURCE_DIR / Path(f).name) for f in SOURCE_FILES]
    TARGET_PATH = str(TARGET_DIR / Path(TARGET_FILENAME).name)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    missing_paths = [p for p in SOURCE_PATHS + [TARGET_PATH] if not os.path.exists(p)]
    if missing_paths:
        raise FileNotFoundError("Missing input file(s):\n" + "\n".join(missing_paths))

    # ================= 预加载所有点云 =================
    print(":: 预加载所有点云文件...")
    source_pcds_raw = [load_mesh_as_pcd(p, 100000) for p in SOURCE_PATHS]
    target_raw = load_mesh_as_pcd(TARGET_PATH, 100000)
    empty_sources = [SOURCE_PATHS[i] for i, pcd in enumerate(source_pcds_raw) if pcd.is_empty()]
    if empty_sources:
        raise ValueError("Failed to load source point cloud(s):\n" + "\n".join(empty_sources))
    if target_raw.is_empty():
        raise ValueError(f"Failed to load target point cloud: {TARGET_PATH}")

    # ================= 计算 Voxel Size =================
    best_voxel_size = calculate_adaptive_voxel_size(target_raw, factor=ADAPTIVE_VOXEL_FACTOR)
    print(f":: 自适应 Voxel Size: {best_voxel_size:.4f} mm")

    # ================= 执行完整配准 =================
    print(":: 开始执行配准流程...")
    result = run_registration_pipeline(
        source_pcds_raw, target_raw, SOURCE_PATHS, best_voxel_size, verbose=True
    )

    transformed = result['transformed']
    metrics = result['metrics']
    target_mirrored = result['target_mirrored']
    order = result['order']
    original_names = result['original_names']
    overall = result['overall']
    print(f":: 配准完成 | Mean RMSE: {overall['mean_rmse']:.4f} | Mean Fitness: {overall['mean_fitness']:.4f}")

    # ================= 导出与保存 =================
    print(":: 保存配准报告与对齐点云...")
    report_path = OUTPUT_DIR / "adaptive_voxel_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"自适应 Voxel Size: {best_voxel_size:.4f} mm\n")
        f.write(f"Mean RMSE: {overall['mean_rmse']:.4f}\n")
        f.write(f"Mean Fitness: {overall['mean_fitness']:.4f}\n")

    ordered_names = [original_names[i] for i in order]
    plot_registration_evaluation(metrics, ordered_names, str(OUTPUT_DIR / "Density_Plot.png"))

    for i, idx in enumerate(order):
        base_name = os.path.splitext(original_names[idx])[0]
        out_ply = OUTPUT_DIR / f"Aligned_{base_name}.ply"
        o3d.io.write_point_cloud(str(out_ply), transformed[i])
        print(f"   已保存: Aligned_{base_name}.ply")

    # ================= 可视化 1 & 2 =================
    print(":: 可视化 [1/2]: 配准结果 (含模板)...")
    visualize_registration(transformed, target_mirrored, names=ordered_names, metrics=metrics,
                           output_path=str(OUTPUT_DIR / "Vis1_WithTemplate.ply"))
    print(":: 可视化 [2/2]: 仅碎片对齐结果...")
    visualize_fragments_only(transformed, names=ordered_names, metrics=metrics,
                             output_path=str(OUTPUT_DIR / "Vis2_FragmentsOnly.ply"))

    # ================= 提取骨折线 =================
    print(":: 提取骨折线...")
    fracture_lines, per_fragment_lines = extract_fracture_lines(
        transformed, target_mirrored, fragment_names=ordered_names, config=FRACTURE_LINE_CONFIG
    )
    n_pts = len(fracture_lines.points) if not fracture_lines.is_empty() else 0
    print(f":: 骨折线提取完成, 共 {n_pts} 个点")

    print(":: 可视化骨折线...")
    visualize_fracture_lines(
        target_mirrored, fracture_lines, per_fragment_lines=per_fragment_lines,
        fragment_names=ordered_names, output_path=str(OUTPUT_DIR / "Vis3_Fracture_Lines.ply")
    )

    print(":: 全部流程完成.")
    plt.show()
