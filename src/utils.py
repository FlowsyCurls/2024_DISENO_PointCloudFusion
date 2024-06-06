import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

COLORS = [
    (1, 0, 0),      # Rojo
    (1, 0.5, 0),    # Naranja
    (1, 1, 0),      # Amarillo
    (0, 1, 0),      # Verde
    (0, 1, 1),      # Cian
    (0, 0, 1),      # Azul
    (0.5, 0, 1),    # Púrpura
    (1, 0, 1),      # Magenta
    (0.5, 0.5, 0),  # Amarillo oscuro
    (0, 0.5, 0.5)   # Verde azulado oscuro
]
    
    
# Generate a string representation of the current timestamp
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# Ensure the directory exists, create if it does not.
def ensure_directory(path):
    os.makedirs(path, exist_ok=True)
    return path

    
# Calculate the cumulative average and append it to the list of previous values.
def calculate_cumulative_average(previous_values, new_value):
    if not previous_values:
        return [new_value]
    else:
        cumulative_average = (previous_values[-1] * len(previous_values) + new_value) / (len(previous_values) + 1)
        return previous_values + [cumulative_average]


def set_metrics_plot(metrics):
    """
    Generates a single figure with two subplots: 
    1. RMSE vs Number of Point Clouds (scatter plot)
    2. Fitness vs Number of Point Clouds (scatter plot)

    Parameters:
    metrics (pd.DataFrame): DataFrame containing metrics with columns '# of Clouds', 'RMSE', and 'Fitness'.
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

    # Gráfico de RMSE vs Número de Nubes de Puntos (gráfico de dispersión con línea de tendencia y líneas de error)
    axes[0].scatter(metrics['# of Clouds'], metrics['RMSE'], marker='o', linestyle='', color='b', label='Datos')
    fit_rmse = np.polyfit(metrics['# of Clouds'], metrics['RMSE'], 1)
    fit_fn_rmse = np.poly1d(fit_rmse)
    trendline_rmse = fit_fn_rmse(metrics['# of Clouds'])
    axes[0].plot(metrics['# of Clouds'], trendline_rmse, color='r', linestyle='--',  label='Línea de tendencia')

    for i in range(len(metrics)):
        axes[0].plot([metrics['# of Clouds'][i], metrics['# of Clouds'][i]], [metrics['RMSE'][i], trendline_rmse[i]], color='b', linestyle=':')

    axes[0].set_title('RMSE vs Número de Nubes de Puntos')
    axes[0].set_xlabel('Número de Nubes de Puntos')
    axes[0].set_ylabel('RMSE')
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[0].legend()

    # Gráfico de Fitness vs Número de Nubes de Puntos (gráfico de dispersión con línea de tendencia y líneas de error)
    axes[1].scatter(metrics['# of Clouds'], metrics['Fitness'], marker='o', linestyle='', color='g',  label='Datos')
    fit_fitness = np.polyfit(metrics['# of Clouds'], metrics['Fitness'], 1)
    fit_fn_fitness = np.poly1d(fit_fitness)
    trendline_fitness = fit_fn_fitness(metrics['# of Clouds'])
    axes[1].plot(metrics['# of Clouds'], trendline_fitness, color='r', linestyle='--',  label='Línea de tendencia')

    for i in range(len(metrics)):
        axes[1].plot([metrics['# of Clouds'][i], metrics['# of Clouds'][i]], [metrics['Fitness'][i], trendline_fitness[i]], color='g', linestyle=':')

    axes[1].set_title('Fitness vs Número de Nubes de Puntos')
    axes[1].set_xlabel('Número de Nubes de Puntos')
    axes[1].set_ylabel('Fitness')
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[1].legend()

    plt.tight_layout()
    return fig
