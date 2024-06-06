import open3d as o3d
import numpy as np
import pandas as pd
import os
from tqdm import tqdm


def load_data(testing, input_arg, num_point_clouds):
    """
    Loads a dataset based on whether it's for testing or regular use.

    Args:
        testing (bool): Flag to determine whether to load a test dataset.
        input_arg (str): Input argument specifying the dataset or the id of the chosen test dataset.
        num_point_clouds (int): Number of point clouds to load.

    Returns:
        Dataset loaded either from test configuration or from specified directory.
    """
    if testing:
        return load_test_dataset(int(input_arg), num_point_clouds)
    else:
        data_dir = os.path.join(os.getcwd(), "data", input_arg)
        return load_input_dataset(data_dir, num_point_clouds)


# Returns a pandas DataFrame with with some information about the param dataset
def get_dataset_info(clouds):
    num_clouds = len(clouds)
    avg_points_per_cloud = sum(len(c.points) for c in clouds) // num_clouds

    # Create the DataFrame with the data
    dataset_info = pd.DataFrame({
        'Value': [num_clouds, avg_points_per_cloud]
    }, index=['Number of point clouds', 'Average points per cloud'])

    return dataset_info


def load_test_dataset(option, num_pcds):
    """
    Loads a test dataset based on the given option.

    Args:
        option (int): Test dataset number to load.
        num_pcds (int): Number of point clouds to load (-1 to load all).

    Returns:
        list: List of PointCloud objects.
    """
    selected_dataset = None
    if option == 1:
        selected_dataset = o3d.data.LivingRoomPointClouds() # 57 point clouds of binary PLY format from the Redwood RGB-D Dataset.
    elif option == 2:
        selected_dataset = o3d.data.OfficePointClouds()     # 53 point clouds of binary PLY format from Redwood RGB-D Dataset.
    else:
        selected_dataset = o3d.data.DemoICPPointClouds()    # 2 point cloud fragments of binary PCD format
    clouds = [o3d.io.read_point_cloud(path) for path in selected_dataset.paths]
    
    # If num_pcds is -1 or surpassed, then consider all clouds
    if num_pcds == -1 or num_pcds>len(clouds):
        num_pcds = len(clouds)
        
    out = []
    print("\nLoading ...")
    for i, c in tqdm(enumerate(clouds), desc=''):
        if i >= num_pcds:
            break
        out.append(c)
        
    print("[SUCCESS] Point clouds loaded: Open3D")
    return out


def load_input_dataset(directory, num_pcds):
    """
    Loads an input dataset from a given directory.

    Args:
        directory (str): Path to the directory containing point clouds.
        num_pcds (int): Number of point clouds to load (-1 to load all).

    Returns:
        list: List of PointCloud objects.
    """
    extensions = ['ply', 'pcd']
    base_path = os.getcwd()
    relative_filenames = _find_files_by_extension(directory, extensions, base_path)
    clouds = _load_dataset_from_path(relative_filenames)
   
    # If num_pcds is -1 or surpassed, then consider all clouds
    if num_pcds == -1 or num_pcds>len(clouds):
        num_pcds = len(clouds)
     
    out = []
    print("\nLoading ...")
    for i, c in tqdm(enumerate(clouds), desc=""):
        if i >= num_pcds:
            break
        out.append(c)
        
    print(f"[SUCCESS] Point clouds loaded: '{directory}'")
    return out


def _find_files_by_extension(directory, extensions, base_path=None):
    """
    Find files in a directory with given extensions, even within subfolders.
    Args:
        directory_ (str): The directory to search for files in.
        extensions_ (list): List of file extensions to search for.
        base_path (str, optional): Base path to build relative paths. If not provided, absolute paths will be returned.
    Returns:
        list: A list of paths (relative or absolute) to found files matching the given extensions.
    """
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, filename)
                if base_path:
                    file_path = os.path.relpath(file_path, base_path)
                files.append(file_path)
    return files
   

# Loads each source point cloud from its respective file path.
def _load_dataset_from_path(relative_path_filenames):
    """
    Loads point clouds from the given file paths.

    Args:
        file_paths (list): List of file paths to load point clouds from.

    Returns:
        list: List of point clouds.
    """
    clouds = []
    for filename in relative_path_filenames:
        try:
            point_cloud = o3d.io.read_point_cloud(filename)
            clouds.append(point_cloud)
        except Exception as e:
            print(f"[FAILURE] Could not read file '{filename}': {e}")

    # Check if any point cloud is empty
    if not clouds:
        raise RuntimeError("No valid point clouds were found in the directory.")
    return clouds