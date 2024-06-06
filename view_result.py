import open3d as o3d
import tkinter as tk
from tkinter import filedialog

def select_and_display_point_cloud():
    # Configura la raíz de Tkinter
    root = tk.Tk()
    root.withdraw()  # Usamos withdraw para ocultar la ventana principal de Tkinter
    
    # Abre la ventana de diálogo para seleccionar el archivo
    file_path = filedialog.askopenfilename(title="Select a point cloud file", filetypes=[("PCD, PLY files", "*.pcd;*.ply"), ("All files", "*.*")])
    if not file_path:
        print("No file was selected.")
        return
    
    # Lee la nube de puntos
    pcd = o3d.io.read_point_cloud(file_path)
    if pcd.is_empty():
        print("The point cloud is empty. Check the file and format.")
        return
    
    # Visualiza la nube de puntos
    o3d.visualization.draw_geometries([pcd], window_name="Model View")

# Llama a la función para seleccionar y visualizar la nube de puntos
select_and_display_point_cloud()
