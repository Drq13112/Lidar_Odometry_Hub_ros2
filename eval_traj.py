import numpy as np
import matplotlib.pyplot as plt

# --- Configuración ---
# Nombres de los archivos adjuntos
ground_truth_file = 'traj_gt.txt'
estimated_pose_file = 'estimated_pose.txt'

# --- Carga de Datos ---
try:
    # Cargar los datos de la trayectoria estimada
    # Formato: timestamp tx ty tz qx qy qz qw
    estimated_data = np.loadtxt(estimated_pose_file, delimiter=' ')
    
    # Cargar los datos de la trayectoria ground truth (UTM)
    # Asumimos el mismo formato
    gt_data = np.loadtxt(ground_truth_file, delimiter=' ')
    
    print(f"Se cargaron {len(estimated_data)} puntos de la pose estimada.")
    print(f"Se cargaron {len(gt_data)} puntos de la trayectoria ground truth.")

except FileNotFoundError as e:
    print(f"Error: No se pudo encontrar el archivo {e.filename}.")
    print("Asegúrate de que los archivos 'traj_gt.txt' y 'estimated_pose.txt' estén en el mismo directorio que el script.")
    exit()
except Exception as e:
    print(f"Ocurrió un error al leer los archivos: {e}")
    exit()

# --- Procesamiento de Datos ---

# Extraer las coordenadas X e Y de la pose estimada
# tx es la columna 1 (índice 1), ty es la columna 2 (índice 2)
est_x = -estimated_data[:, 1]
est_y = -estimated_data[:, 2]

# Extraer las coordenadas X e Y de la trayectoria ground truth
gt_x_utm = gt_data[:, 1]
gt_y_utm = gt_data[:, 2]

# Normalizar la trayectoria ground truth para que comience en (0,0)
# Esto la pasa a un sistema de coordenadas local alineado con el de la pose estimada
gt_x_local = gt_x_utm - gt_x_utm[0]
gt_y_local = gt_y_utm - gt_y_utm[0]


# --- Visualización ---
plt.style.use('seaborn-v0_8-whitegrid') # Estilo de gráfico agradable
fig, ax = plt.subplots(figsize=(10, 8))

# Plotea la trayectoria ground truth (normalizada)
ax.plot(gt_x_local, gt_y_local, label='Trayectoria Ground Truth (GT)', color='green', linewidth=3, linestyle='--')

# Plotea la trayectoria estimada
ax.plot(est_x, est_y, label='Trayectoria Estimada', color='dodgerblue', linewidth=2)

# Marcar los puntos de inicio y fin de ambas trayectorias
ax.scatter(gt_x_local[0], gt_y_local[0], marker='o', color='darkgreen', s=100, zorder=5, label='Inicio GT')
ax.scatter(est_x[0], est_y[0], marker='o', color='blue', s=100, zorder=5, label='Inicio Estimado')
ax.scatter(gt_x_local[-1], gt_y_local[-1], marker='x', color='darkgreen', s=150, zorder=5, label='Fin GT')
ax.scatter(est_x[-1], est_y[-1], marker='x', color='blue', s=150, zorder=5, label='Fin Estimado')


# Configuración del gráfico
ax.set_title('Comparación de Trayectorias (Ground Truth vs. Estimada)', fontsize=16)
ax.set_xlabel('Coordenada X (metros)', fontsize=12)
ax.set_ylabel('Coordenada Y (metros)', fontsize=12)
ax.legend(fontsize=10)
ax.axis('equal') # Asegura que la escala en X e Y sea la misma para no distorsionar la forma

# Mostrar el gráfico
plt.show()
