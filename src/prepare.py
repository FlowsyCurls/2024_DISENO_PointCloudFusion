import open3d as o3d
from tqdm import tqdm

# Prepare all point clouds with the voxel size received as param.
def prepare_point_clouds(clouds, voxel_size):
    print("\nPreparing ...")
    prepared_clouds = []
    for cloud in tqdm(clouds, desc=""):
        prepared = process_cloud(cloud, voxel_size)
        prepared_clouds.append(prepared)
    print(f"[SUCCESS] Point clouds prepared with '{voxel_size}' voxel size")
    return prepared_clouds


# Preprocesses a point cloud by removing outliers, applying downsampling and calculating FPFH features.
def process_cloud(pcd_raw, voxel_size):
    pcd_filtered, _ = pcd_raw.remove_statistical_outlier(nb_neighbors=12, std_ratio=1.0)
    # pcd_filtered, _ = pcd_raw.remove_radius_outlier(nb_points=16, radius=0.5)

    pcd_downsampled = pcd_filtered.voxel_down_sample(voxel_size)
    pcd_downsampled.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius = voxel_size * 2, max_nn=30))

    # Fast Point Feature Histograms (FPFH)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_downsampled, o3d.geometry.KDTreeSearchParamHybrid(radius = voxel_size * 5, max_nn=100))

    return pcd_downsampled, pcd_fpfh