import os
import open3d as o3d
import pandas as pd

from src.utils import ensure_directory

# Configuration
OUTPUT_FOLDER_NAME = "output"

# Save a DataFrame to a CSV file.
def save_dataset(dataset: pd.DataFrame, timestamp):
    try:
        # Ensure the directory for storing the dataset exists
        dir_path = ensure_directory(os.path.join(os.getcwd(), OUTPUT_FOLDER_NAME, f"results_{timestamp}"))
        # Create the full file path for the dataset
        file_path = os.path.join(dir_path, f"metrics_{timestamp}.csv")
        
        # Attempt to save the DataFrame to a CSV file
        dataset.to_csv(file_path, index=False)
        print(f"[SUCCESS] Metrics file saved in: {file_path}")
    
    except Exception as e:
        # Print an error message if any exception occurs during the file saving process
        print(f"[ERROR] Failed to save metrics file: {e}")

# Save a plot to a PNG file within a timestamped subdirectory.
def save_plot(fig, timestamp):
    try:
        dir_path = ensure_directory(os.path.join(os.getcwd(), OUTPUT_FOLDER_NAME, f"results_{timestamp}"))
        file_path = os.path.join(dir_path, f"plot_{timestamp}.png")
        
        # Attempt to save the figure to a PNG file
        fig.savefig(file_path)
        print(f"[SUCCESS] Image saved in: {file_path}")
    
    except Exception as e:
        print(f"[ERROR] An exception occurred while saving the plot: {e}")

# Save a point cloud to a file in specified format.
def save_point_cloud(cloud, format, timestamp):
    try:
        dir_path = ensure_directory(os.path.join(os.getcwd(), OUTPUT_FOLDER_NAME, f"results_{timestamp}"))
        file_path = os.path.join(dir_path, f"model_{timestamp}.{format}")
        
        # Attempt to save the point cloud
        if o3d.io.write_point_cloud(file_path, cloud):
            print(f"[SUCCESS] Model saved in: {file_path}")
        else:
            print(f"[ERROR] Failed to save the model at: {file_path}")
    
    except Exception as e:
        print(f"[ERROR] An exception occurred: {e}")