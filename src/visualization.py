import copy
import open3d as o3d


def display(pcds, title, color=None):
    
    if not isinstance(pcds, list):
        pcds = [pcds]
    
    colored_pcds = []
    
    if color is None:
        for pcd in pcds:
            colored_pcds.append(pcd)
    elif isinstance(color, tuple):
        for pcd in pcds:
            temp = copy.deepcopy(pcd)
            temp.paint_uniform_color(color)
            colored_pcds.append(temp)
    elif isinstance(color, list):
        for i, pcd in enumerate(pcds):
            temp = copy.deepcopy(pcd)
            temp.paint_uniform_color(color[i % len(color)])
            colored_pcds.append(temp)

    o3d.visualization.draw_geometries(colored_pcds, window_name=title)