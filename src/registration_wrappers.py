import copy
import csv
import os
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d

from config import ADAPTIVE_VOXEL_FACTOR
from registration import align_by_pca, apply_transform_to_all_fragments, merge_point_clouds, register_fragments_with_strategy
from utils import (
    calculate_adaptive_voxel_size,
    classify_and_order_fragments,
    load_mesh_as_pcd,
    mask_target_region,
    mirror_point_cloud,
    preprocess_point_cloud,
)

try:
    import pandas as pd
except Exception:
    pd = None


def compute_inter_fragment_collision(
    current_fragment: o3d.geometry.PointCloud,
    list_of_registered_fragments: List[o3d.geometry.PointCloud],
    threshold: float = 1.0,
) -> float:
    """
    Compute the collision ratio between the current fragment and registered fragments.
    """
    if current_fragment.is_empty() or not list_of_registered_fragments:
        return 0.0

    registered_points = [
        np.asarray(pcd.points)
        for pcd in list_of_registered_fragments
        if pcd is not None and not pcd.is_empty()
    ]
    if not registered_points:
        return 0.0

    combined = o3d.geometry.PointCloud()
    combined.points = o3d.utility.Vector3dVector(np.vstack(registered_points))
    tree = o3d.geometry.KDTreeFlann(combined)

    points = np.asarray(current_fragment.points)
    if len(points) == 0:
        return 0.0

    threshold_sq = threshold * threshold
    collision_count = 0
    for p in points:
        _, _, dist2 = tree.search_knn_vector_3d(p, 1)
        if len(dist2) > 0 and dist2[0] <= threshold_sq:
            collision_count += 1
    return collision_count / float(len(points))


def _build_source_names(
    source_pcds: List[o3d.geometry.PointCloud],
    source_names: Optional[List[str]],
    source_paths: Optional[List[str]],
) -> List[str]:
    if source_names is not None:
        return source_names
    if source_paths is not None:
        return [os.path.basename(p) for p in source_paths]
    return [f"fragment_{i}" for i in range(len(source_pcds))]


def _write_collision_csv(output_csv: str, rows: List[dict]) -> None:
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "fragment_index", "fragment_name", "collision_ratio", "threshold"],
        )
        writer.writeheader()
        writer.writerows(rows)


def run_registration_with_collision_check(
    source_pcds: List[o3d.geometry.PointCloud],
    target_down: o3d.geometry.PointCloud,
    voxel_size: float,
    mask_radius: float,
    source_paths: Optional[List[str]] = None,
    source_names: Optional[List[str]] = None,
    registration_order: Optional[List[int]] = None,
    classified: Optional[dict] = None,
    collision_threshold: float = 1.0,
    output_csv: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[list, list, list, list, list, List[int]]:
    """
    Wrapper to run registration and report inter-fragment collisions.
    """
    names = _build_source_names(source_pcds, source_names, source_paths)

    if classified is None or registration_order is None:
        path_list = source_paths if source_paths is not None else names
        classified, registration_order = classify_and_order_fragments(path_list, source_pcds)

    results, metrics, transformed, processed = register_fragments_with_strategy(
        source_pcds,
        names,
        target_down,
        registration_order,
        classified,
        voxel_size,
        mask_radius,
        verbose=verbose,
    )

    collision_ratios = []
    collision_rows = []
    registered = []
    for step, (idx, fragment) in enumerate(zip(registration_order, transformed), start=1):
        ratio = compute_inter_fragment_collision(fragment, registered, threshold=collision_threshold)
        collision_ratios.append(ratio)
        if ratio > 0.05:
            name = names[idx] if idx < len(names) else f"fragment_{idx}"
            print(f"[SEVERE] Inter-fragment collision: {name} ratio={ratio:.3f}")
        collision_rows.append({
            "step": step,
            "fragment_index": idx,
            "fragment_name": names[idx] if idx < len(names) else f"fragment_{idx}",
            "collision_ratio": ratio,
            "threshold": collision_threshold,
        })
        registered.append(fragment)

    if output_csv:
        _write_collision_csv(output_csv, collision_rows)

    return results, metrics, transformed, processed, collision_ratios, registration_order


def compute_adaptive_order_greedy(
    source_pcds: List[o3d.geometry.PointCloud],
    target_down: o3d.geometry.PointCloud,
    voxel_size: float,
    mask_radius: float,
    max_iterations: int = 15,
) -> List[int]:
    """
    Greedy ordering using rapid point-to-point ICP fitness against the masked target.
    """
    current_target = copy.deepcopy(target_down)
    remaining = set(range(len(source_pcds)))
    order: List[int] = []

    while remaining:
        best_idx = None
        best_fitness = -1.0
        best_transform = np.eye(4)

        for idx in sorted(remaining):
            source = source_pcds[idx]
            if source.is_empty() or current_target.is_empty():
                continue
            icp_res = o3d.pipelines.registration.registration_icp(
                source,
                current_target,
                voxel_size * 2.0,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations),
            )
            if icp_res.fitness > best_fitness:
                best_fitness = icp_res.fitness
                best_idx = idx
                best_transform = icp_res.transformation

        if best_idx is None:
            order.extend(sorted(list(remaining)))
            break

        order.append(best_idx)
        remaining.remove(best_idx)

        # Mask the target using the coarse alignment result.
        transformed = copy.deepcopy(source_pcds[best_idx]).transform(best_transform)
        current_target = mask_target_region(current_target, transformed, mask_radius)

    return order


def register_fragments_adaptive_order(
    source_pcds: List[o3d.geometry.PointCloud],
    target_down: o3d.geometry.PointCloud,
    voxel_size: float,
    mask_radius: float,
    source_paths: Optional[List[str]] = None,
    source_names: Optional[List[str]] = None,
    classified: Optional[dict] = None,
    verbose: bool = False,
) -> Tuple[list, list, list, list, List[int]]:
    """
    Adaptive-order registration that uses greedy ICP confidence for ordering.
    """
    names = _build_source_names(source_pcds, source_names, source_paths)
    if classified is None:
        path_list = source_paths if source_paths is not None else names
        classified, _ = classify_and_order_fragments(path_list, source_pcds)

    adaptive_order = compute_adaptive_order_greedy(
        source_pcds, target_down, voxel_size, mask_radius
    )

    results, metrics, transformed, processed = register_fragments_with_strategy(
        source_pcds,
        names,
        target_down,
        adaptive_order,
        classified,
        voxel_size,
        mask_radius,
        verbose=verbose,
    )

    return results, metrics, transformed, processed, adaptive_order


def run_experiment_mask_radius_ablation(
    source_pcds: List[o3d.geometry.PointCloud],
    target_down: o3d.geometry.PointCloud,
    voxel_size: float,
    source_paths: Optional[List[str]] = None,
    source_names: Optional[List[str]] = None,
    registration_order: Optional[List[int]] = None,
    classified: Optional[dict] = None,
    verbose: bool = False,
):
    """
    Sweep mask radius and collect average fitness and RMSE for each setting.
    """
    names = _build_source_names(source_pcds, source_names, source_paths)

    if classified is None or registration_order is None:
        path_list = source_paths if source_paths is not None else names
        classified, registration_order = classify_and_order_fragments(path_list, source_pcds)

    results = {"mask_radius": [], "avg_fitness": [], "avg_rmse": []}

    for radius_factor in np.arange(0.0, 4.0 + 1e-9, 0.5):
        test_radius = radius_factor * voxel_size
        _, metrics, _, _ = register_fragments_with_strategy(
            source_pcds,
            names,
            target_down,
            registration_order,
            classified,
            voxel_size,
            test_radius,
            verbose=verbose,
        )
        avg_fitness = float(np.mean([m["fitness"] for m in metrics])) if metrics else 0.0
        avg_rmse = float(np.mean([m["rmse"] for m in metrics])) if metrics else 0.0
        results["mask_radius"].append(test_radius)
        results["avg_fitness"].append(avg_fitness)
        results["avg_rmse"].append(avg_rmse)

    if pd is not None:
        return pd.DataFrame(results)
    return results


def run_collision_test_case(
    source_dir: str,
    target_dir: str,
    output_dir: str,
    source_files: List[str],
    target_filename: str,
    voxel_factor: float = ADAPTIVE_VOXEL_FACTOR,
    mirror_axis: str = "x",
    collision_threshold: float = 1.0,
    mask_radius_factor: float = 1.5,
    verbose: bool = True,
) -> dict:
    """
    End-to-end test case that runs registration with collision reporting and CSV output.
    """
    source_paths = [os.path.join(source_dir, f) for f in source_files]
    target_path = os.path.join(target_dir, target_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if verbose:
        print(":: Loading point clouds...")
    source_pcds_raw = [load_mesh_as_pcd(p, 100000) for p in source_paths]
    target_raw = load_mesh_as_pcd(target_path, 100000)

    voxel_size = calculate_adaptive_voxel_size(target_raw, factor=voxel_factor)
    if verbose:
        print(f":: Adaptive voxel size: {voxel_size:.4f} mm")

    source_pcds = [preprocess_point_cloud(copy.deepcopy(pcd), voxel_size) for pcd in source_pcds_raw]
    target_down = preprocess_point_cloud(copy.deepcopy(target_raw), voxel_size)
    target_mirrored = mirror_point_cloud(target_down, voxel_size, mirror_axis)

    if verbose:
        print(":: PCA global alignment...")
    combined_source = merge_point_clouds(source_pcds)
    pca_transform = align_by_pca(combined_source, target_mirrored, verbose=verbose)
    source_pcds = apply_transform_to_all_fragments(source_pcds, pca_transform)

    classified, order = classify_and_order_fragments(source_paths, source_pcds)
    source_names = [os.path.basename(p) for p in source_paths]
    output_csv = os.path.join(output_dir, "collision_report.csv")

    if verbose:
        print(":: Running registration with collision statistics...")
    results, metrics, transformed, processed, collision_ratios, order = run_registration_with_collision_check(
        source_pcds=source_pcds,
        target_down=target_mirrored,
        voxel_size=voxel_size,
        mask_radius=voxel_size * mask_radius_factor,
        source_names=source_names,
        registration_order=order,
        classified=classified,
        collision_threshold=collision_threshold,
        output_csv=output_csv,
        verbose=verbose,
    )

    mean_collision = float(np.mean(collision_ratios)) if collision_ratios else 0.0
    if verbose:
        print(f":: Collision CSV saved: {output_csv}")
        print(f":: Mean collision ratio: {mean_collision:.4f}")

    return {
        "voxel_size": voxel_size,
        "results": results,
        "metrics": metrics,
        "transformed": transformed,
        "processed": processed,
        "collision_ratios": collision_ratios,
        "order": order,
        "output_csv": output_csv,
    }


if __name__ == "__main__":
    SOURCE_DIR = r"E:\Study\Fracture_Line_Study\Main_Project\No.56-73(without63&69)\patient57"
    TARGET_DIR = r"E:\Study\Fracture_Line_Study\Main_Project\No.56-73(health)"
    OUTPUT_DIR = r"E:\Study\Fracture_Line_Study\Main_Project\OUTPUT\Patient57"

    SOURCE_FILES = [
        "Anonymous_Smoothed_Cyan 1_001.stl",
        "Anonymous_Smoothed_Fuchsia 1_001.stl",
        "Anonymous_Smoothed_Green 5_001.stl",
        "Anonymous_Smoothed_head_001.stl",
        "Anonymous_Smoothed_shaft_001.stl",
    ]
    TARGET_FILENAME = "Patient57.ply"

    run_collision_test_case(
        source_dir=SOURCE_DIR,
        target_dir=TARGET_DIR,
        output_dir=OUTPUT_DIR,
        source_files=SOURCE_FILES,
        target_filename=TARGET_FILENAME,
        verbose=True,
    )
