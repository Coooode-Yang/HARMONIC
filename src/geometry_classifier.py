from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d


LABEL_HEAD = "Head"
LABEL_SHAFT = "Shaft"
LABEL_COMMINUTED = "Comminuted"


@dataclass(frozen=True)
class GeometryThresholds:
    head_vol_thresh: float = 28022.934439
    shaft_vol_thresh: float = 7741.922768
    circ_thresh: float = 0.307017
    sph_thresh: float = 0.284861
    lin_thresh: float = 0.783658
    sav_thresh: float = 0.187248
    sol_thresh: float = 0.663420


@dataclass(frozen=True)
class FragmentFeatures:
    volume: float
    circularity: float
    sphericity: float
    linearity: float
    sav_ratio: float
    solidity: float


class GeometryClassifier:
    def __init__(
        self,
        thresholds: GeometryThresholds = GeometryThresholds(),
        sample_point_count: int = 3000,
    ) -> None:
        self.thresholds = thresholds
        self.sample_point_count = max(3, int(sample_point_count))

    def classify_point_cloud(self, pcd: o3d.geometry.PointCloud) -> str:
        try:
            features = self.extract_features(pcd)
        except (np.linalg.LinAlgError, RuntimeError, ValueError, FloatingPointError):
            return LABEL_COMMINUTED

        if features is None:
            return LABEL_COMMINUTED

        th = self.thresholds
        head_mask = (
            (features.volume >= th.head_vol_thresh)
            and (features.circularity > th.circ_thresh)
            and (features.sphericity > th.sph_thresh)
        )
        if head_mask:
            return LABEL_HEAD

        shaft_mask = (
            (features.volume >= th.shaft_vol_thresh)
            and (features.linearity > th.lin_thresh)
            and (features.solidity > th.sol_thresh)
            and (features.sav_ratio < th.sav_thresh)
        )
        if shaft_mask:
            return LABEL_SHAFT

        return LABEL_COMMINUTED

    def extract_features(self, pcd: o3d.geometry.PointCloud) -> Optional[FragmentFeatures]:
        sampled_pcd = self._uniform_sample_point_cloud(pcd)
        points = np.asarray(sampled_pcd.points, dtype=float)

        if points.shape[0] < 3:
            return None

        cov_matrix = np.cov(points.T)
        if not np.all(np.isfinite(cov_matrix)):
            return None

        eigenvalues, _ = np.linalg.eigh(cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]
        l1, l2, l3 = float(eigenvalues[0]), float(eigenvalues[1]), float(eigenvalues[2])
        if l1 <= 1e-12:
            return None

        linearity = (l1 - l2) / l1
        circularity = l2 / l1
        sphericity = l3 / l1

        obb = sampled_pcd.get_oriented_bounding_box()
        extent = np.asarray(obb.extent, dtype=float)
        volume = float(np.prod(extent))

        hull, _ = sampled_pcd.compute_convex_hull()
        hull_volume = float(hull.get_volume())
        hull_area = float(hull.get_surface_area())

        sav_ratio = hull_area / hull_volume if hull_volume > 0.0 else float("inf")
        solidity = volume / hull_volume if hull_volume > 0.0 else 0.0

        values = np.array(
            [volume, circularity, sphericity, linearity, sav_ratio, solidity],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            return None

        return FragmentFeatures(
            volume=volume,
            circularity=float(circularity),
            sphericity=float(sphericity),
            linearity=float(linearity),
            sav_ratio=float(sav_ratio),
            solidity=float(solidity),
        )

    def _uniform_sample_point_cloud(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        if pcd.is_empty():
            return pcd

        point_count = len(pcd.points)
        if point_count <= self.sample_point_count:
            return pcd

        every_k_points = max(1, point_count // self.sample_point_count)
        sampled_pcd = pcd.uniform_down_sample(every_k_points)
        if len(sampled_pcd.points) <= self.sample_point_count:
            return sampled_pcd

        keep_idx = np.linspace(
            0,
            len(sampled_pcd.points) - 1,
            self.sample_point_count,
            dtype=int,
        )
        return sampled_pcd.select_by_index(keep_idx.tolist())
