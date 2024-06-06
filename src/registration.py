import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm

from src.utils import calculate_cumulative_average

# Constantes
PARAMS = {
    "DISTANCE_COARSE_THRESHOLD": 15,
    "DISTANCE_FINE_THRESHOLD": 1.5,
    "EDGE_PRUNE_THRESHOLD": 0.25,
}

""" Global Registration """

# Performs RANSAC global registration based on FPFH features between two reduced point clouds.
def RANSAC(source_down, target_down, source_fpfh, target_fpfh, distance_threshold=0.4):
    # Default thresholds and parameters
    length_threshold = 0.9
    angle_threshold = np.deg2rad(45)
    max_iterations = 100000
    convergence_threshold = 0.999

    # Configure correspondence checkers
    correspondence_checkers = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(length_threshold),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(angle_threshold),
    ]

    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, 
        mutual_filter = True, 
        max_correspondence_distance = distance_threshold,
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 
        ransac_n = 2, 
        checkers = correspondence_checkers,
        criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(max_iterations, convergence_threshold))

# Performs Fast Global Registration based on features matching between two reduced point clouds.
def FastGlobReg(source_down, target_down, source_fpfh, target_fpfh, distance_threshold=0.4):
    # Default thresholds and parameters    
    return o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=distance_threshold))
    
    
""" Local Registration """

    
# Performs registration using the ICP algorithm with point-to-point transformation estimation.
def PointToPointICP(source, target, max_correspondence_distance, init):
    return o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance, init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))

# Performs registration using the ICP algorithm with point-to-plane transformation estimation.
def PointToPlaneICP(source, target, max_correspondence_distance, init):
    return o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance, init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))

""" Registration Methods """ 

def pairwise_registration(source, target, coarse_method:str, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    """
    Perform pairwise registration between two point clouds using specified methods for coarse and fine registration.

    Args:
        source (tuple): Tuple containing downsampled source point cloud and its features.
        target (tuple): Tuple containing downsampled target point cloud and its features.
        coarse_method (str): Method to use for coarse registration ('ransac', 'fast', 'icp').
        max_correspondence_distance_coarse (float): Maximum correspondence distance for coarse registration.
        max_correspondence_distance_fine (float): Maximum correspondence distance for fine registration.

    Returns:
        tuple: Contains transformation matrix, information matrix, RMSE, and fitness score of the registration.

    Raises:
        ValueError: If an unsupported registration method is provided.
    """
    source_down, source_fhfp = source
    target_down, target_fhfp = target
    
    # Coarse/Global registration
    if coarse_method == "ransac":
        coarse = RANSAC(source_down, target_down, source_fhfp, target_fhfp, max_correspondence_distance_coarse)
    elif coarse_method == "fast":
        coarse = FastGlobReg(source_down, target_down, source_fhfp, target_fhfp, max_correspondence_distance_coarse)
    elif coarse_method == "icp":
        coarse = PointToPlaneICP(source_down, target_down, max_correspondence_distance_coarse, np.identity(4))
    else:
        raise ValueError(f"Unsupported coarse registration method: {coarse_method}")

    # Fine/Local registration
    fine = PointToPlaneICP(source_down, target_down, max_correspondence_distance_fine, coarse.transformation)
    transformation_icp = fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source_down, target_down, max_correspondence_distance_fine, fine.transformation)
    
    # Obtain the RMSE error and fitness of the last iteration of ICP
    rmse = fine.inlier_rmse
    fitness = fine.fitness
    
    return transformation_icp, information_icp, rmse, fitness


def _full_registration(clouds, coarse_method:str, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    """
    Perform full registration of multiple point clouds using pairwise registration to build a pose graph.

    Args:
        clouds (list): List of point clouds to be registered.
        coarse_method (str): Method to use for coarse registration ('ransac', 'fast', 'icp').
        max_correspondence_distance_coarse (float): Maximum correspondence distance for coarse registration.
        max_correspondence_distance_fine (float): Maximum correspondence distance for fine registration.

    Returns:
        tuple: Returns a pose graph object containing the registered poses, list of cumulative RMSE errors, and
               fitness values across all registrations.
    """
    
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    
    rmse_errors, fitness_values = [], []
    n =  len(clouds)
    print("\nAligning ...")
    for source_id in tqdm(range(n), desc=""):
        for target_id in range(source_id + 1, n):
            transformation_icp, information_icp, rmse, fitness = pairwise_registration(
                clouds[source_id], clouds[target_id], coarse_method,
                max_correspondence_distance_coarse,
                max_correspondence_distance_fine)
                
            # The registration of point cloud pairs is only performed in the case of odometry.        
            if target_id == source_id + 1:  # odometry case
                
                # Add average and fitness
                rmse_errors = calculate_cumulative_average(rmse_errors, rmse)
                fitness_values = calculate_cumulative_average(fitness_values, fitness)
                
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id, target_id,
                                                   transformation_icp, information_icp,
                                                   uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id, target_id,
                                                   transformation_icp, information_icp,
                                                   uncertain=True))
                
    print(f"[SUCCESS] Point clouds aligned")
    return pose_graph, rmse_errors, fitness_values


def multiway_registration(prepared_clouds, voxel_size, coarse_method):
    max_correspondence_distance_coarse = voxel_size * PARAMS["DISTANCE_COARSE_THRESHOLD"]
    max_correspondence_distance_fine = voxel_size * PARAMS["DISTANCE_FINE_THRESHOLD"]

    # Full registration
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Info) as cm:
        pose_graph, rmse_errors, fitness_values = _full_registration(prepared_clouds, coarse_method,
                                    max_correspondence_distance_coarse,
                                    max_correspondence_distance_fine)
    # Global optimization
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance = max_correspondence_distance_fine,
        edge_prune_threshold = PARAMS["EDGE_PRUNE_THRESHOLD"],
        preference_loop_closure=0.1,
        reference_node = 0)
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Info) as cm:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            # o3d.pipelines.registration.GlobalOptimizationGaussNewton(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)
        
    # Metrics for plotting
    metrics = pd.DataFrame({
        '# of Clouds': range(2, len(rmse_errors) + 2),
        'RMSE': rmse_errors,
        'Fitness': fitness_values
    })

    return pose_graph, metrics


# Applies a transformation to each point cloud according to the provided poses.
def transform_point_clouds(clouds, pose_graph):
    transformed_clouds = []
    for point_id, pcd in enumerate(clouds):
        pcd.transform(pose_graph.nodes[point_id].pose)
        transformed_clouds.append(pcd)
    return transformed_clouds

# Combines multiple point clouds into one and performs downsampling.
def combine_and_downsample_point_clouds(clouds, voxel_size):
    cloud = o3d.geometry.PointCloud()
    for pcd in clouds:
        cloud += pcd

    # Downsampling the combined point cloud
    cloud_down = cloud.voxel_down_sample(voxel_size=voxel_size)
    return cloud_down