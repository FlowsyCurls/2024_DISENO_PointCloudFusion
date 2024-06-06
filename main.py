import copy
import sys
import os
import argparse
from matplotlib import pyplot as plt

# Add the project root directory to the PYTHONPATH
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from src import input, prepare, registration, visualization, export
from src.utils import COLORS, get_timestamp, set_metrics_plot

# Set arguments for command line
def options():
    parser = argparse.ArgumentParser(description="Point Cloud Fusion System")
    parser.add_argument('-i', '--input', required=True, help="Path to input point clouds or test dataset number")
    parser.add_argument('-t', '--test', action='store_true', help="Disable testing")
    parser.add_argument('-n', '--num_point_clouds', type=int, default=-1, help="Number of point clouds to consider")
    parser.add_argument('-v', '--voxel_size', type=float, default=0.05, help="Voxel size for downsampling point clouds")
    parser.add_argument('-m', '--method', choices=['ransac', 'fast', 'icp'], default='fast', help="Coarse Registration method")
    parser.add_argument('-f', '--format', choices=['ply', 'pcd'], default='pcd', help="Output file format (ply or pcd)")
    parser.add_argument('--noview', action='store_true', help="Disable visualization")
    
    args = parser.parse_args()
    
    return args

    
# Loading
def load_data(ARGS):
    if ARGS.test:
        return input.load_test_dataset(int(ARGS.input), ARGS.num_point_clouds)
    else:
        data_dir = os.path.join(os.getcwd(), "data", ARGS.input)
        return input.load_input_dataset(data_dir, ARGS.num_point_clouds)


# Exports
def save_results(metrics, model, format):
    fig = set_metrics_plot(metrics)
    timestamp = get_timestamp()
    print("\nExporting...")
    export.save_dataset(metrics, timestamp)
    export.save_plot(fig, timestamp)
    export.save_point_cloud(model, format, timestamp)
    
    
# Show results 
def visualize_results(view, dataset_info, metrics, source_clouds, clouds, model):
    if view:
        print("\n • DATASET INFORMATION\n" + dataset_info.to_markdown())
        print("\n • METRICS SUMMARY\n" + metrics.to_markdown() + "\n")
        plt.show()
        visualization.display(source_clouds, "Raw Clouds")
        visualization.display(clouds, "Model Result", COLORS)
        visualization.display(model, "Aligned Clouds", (0.5,0.5,0.5))
        

if __name__ == "__main__":
    ARGS = options()
    
    # Input
    source_clouds = input.load_data(ARGS.test, ARGS.input, ARGS.num_point_clouds)
    clouds = copy.deepcopy(source_clouds) # Save a copy of the source clouds
    dataset_info = input.get_dataset_info(clouds)
    
    # Processing
    prepared_clouds = prepare.prepare_point_clouds(clouds, ARGS.voxel_size)
    pose_graph, metrics = registration.multiway_registration(prepared_clouds, ARGS.voxel_size, ARGS.method)
    prepared_clouds = registration.transform_point_clouds(clouds, pose_graph)
    model = registration.combine_and_downsample_point_clouds(clouds=prepared_clouds, voxel_size=0.001)
        
    # Output
    save_results(metrics, model, ARGS.format)
    visualize_results(not ARGS.noview, dataset_info, metrics, source_clouds, clouds, model)


    # python src/main.py -i 1 -t -n 8 -o output/result -f ply
    # python src/main.py -i stanford-bunny -n 7 -m icp -o output/result -f ply
