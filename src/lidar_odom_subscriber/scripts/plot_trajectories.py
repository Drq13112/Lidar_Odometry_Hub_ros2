import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import sys
from scipy.spatial.transform import Rotation

def to_transform_matrices(df):
    """Convierte un DataFrame de poses a una lista de matrices de transformación 4x4."""
    matrices = []
    for i, row in df.iterrows():
        T = np.eye(4)
        T[:3, 3] = row[['x', 'y', 'z']].values
        T[:3, :3] = Rotation.from_quat(row[['qx', 'qy', 'qz', 'qw']].values).as_matrix()
        matrices.append(T)
    return np.array(matrices)

def calculate_ape(est_matrices, gt_matrices):
    """Calcula el Absolute Pose Error (APE) para traslación y rotación."""
    trans_errors = []
    rot_errors = []
    for T_est, T_gt in zip(est_matrices, gt_matrices):
        # Error de traslación
        trans_error = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
        trans_errors.append(trans_error)
        
        # Error de rotación
        R_err = np.dot(np.linalg.inv(T_gt[:3, :3]), T_est[:3, :3])
        rot_error_angle = np.arccos((np.trace(R_err) - 1) / 2)
        rot_errors.append(rot_error_angle)
        
    return np.array(trans_errors), np.array(rot_errors)

def calculate_rpe(est_matrices, gt_matrices, delta=1):
    """Calcula el Relative Pose Error (RPE) para traslación y rotación."""
    trans_errors = []
    rot_errors = []
    for i in range(len(est_matrices) - delta):
        # Movimiento relativo estimado
        T_rel_est = np.dot(np.linalg.inv(est_matrices[i]), est_matrices[i+delta])
        # Movimiento relativo real
        T_rel_gt = np.dot(np.linalg.inv(gt_matrices[i]), gt_matrices[i+delta])
        
        # Error de traslación relativo
        trans_error = np.linalg.norm(T_rel_est[:3, 3] - T_rel_gt[:3, 3])
        trans_errors.append(trans_error)
        
        # Error de rotación relativo
        R_err = np.dot(np.linalg.inv(T_rel_gt[:3, :3]), T_rel_est[:3, :3])
        rot_error_angle = np.arccos((np.trace(R_err) - 1) / 2)
        rot_errors.append(rot_error_angle)
        
    return np.array(trans_errors), np.array(rot_errors)

def get_stats(errors, unit='m'):
    """Calcula y formatea estadísticas de un array de errores."""
    if len(errors) == 0:
        return "N/A"
    stats = {
        'mean': np.mean(errors),
        'std': np.std(errors),
        'rmse': np.sqrt(np.mean(errors**2)),
        'min': np.min(errors),
        'max': np.max(errors)
    }
    return (f"Mean: {stats['mean']:.4f} {unit}, RMSE: {stats['rmse']:.4f} {unit}, "
            f"Std: {stats['std']:.4f} {unit}, Min: {stats['min']:.4f} {unit}, Max: {stats['max']:.4f} {unit}")

def plot_trajectories():
    """
    Lee los ficheros CSV de trayectoria, calcula métricas 2D/3D y genera gráficas.
    """
    est_path = '/home/david/ros2_ws/odom_est_traj.csv'
    gt_path = '/home/david/ros2_ws/odom_gt_traj.csv'
    output_path = '/home/david/ros2_ws/trajectory_comparison.png'
    metrics_path = '/home/david/ros2_ws/trajectory_metrics.txt'
    error_plot_path = '/home/david/ros2_ws/error_evolution.png'
    trajectory_3d_path = '/home/david/ros2_ws/trajectory_3d.png'

    try:
        # Leer los datos usando pandas con las nuevas columnas
        col_names = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        est_df = pd.read_csv(est_path, names=col_names, header=0)
        gt_df = pd.read_csv(gt_path, names=col_names, header=0)
        
        print(f"Datos leídos: {len(est_df)} puntos estimados, {len(gt_df)} puntos GPS")
        
    except FileNotFoundError as e:
        print(f"Error: No se pudo encontrar el fichero de datos {e.filename}.", file=sys.stderr)
        return

    # Sincronizar trayectorias al mínimo común
    min_len = min(len(est_df), len(gt_df))
    est_df = est_df.iloc[:min_len]
    gt_df = gt_df.iloc[:min_len]

    # --- CÁLCULOS 3D ---
    est_matrices_3d = to_transform_matrices(est_df)
    gt_matrices_3d = to_transform_matrices(gt_df)
    
    ape_trans_3d, ape_rot_3d = calculate_ape(est_matrices_3d, gt_matrices_3d)
    rpe_trans_3d, rpe_rot_3d = calculate_rpe(est_matrices_3d, gt_matrices_3d)

    # --- CÁLCULOS 2D (ignorando Z) ---
    est_df_2d = est_df.copy()
    gt_df_2d = gt_df.copy()
    est_df_2d['z'] = 0
    gt_df_2d['z'] = 0
    
    est_matrices_2d = to_transform_matrices(est_df_2d)
    gt_matrices_2d = to_transform_matrices(gt_df_2d)
    
    ape_trans_2d, ape_rot_2d = calculate_ape(est_matrices_2d, gt_matrices_2d)
    rpe_trans_2d, rpe_rot_2d = calculate_rpe(est_matrices_2d, gt_matrices_2d)

    # --- GUARDAR MÉTRICAS ---
    with open(metrics_path, "w") as f:
        f.write("=== TRAJECTORY EVALUATION METRICS ===\n\n")
        f.write("--- METRICS 3D ---\n")
        f.write(f"APE (Translation): {get_stats(ape_trans_3d, 'm')}\n")
        f.write(f"APE (Rotation):    {get_stats(np.rad2deg(ape_rot_3d), 'deg')}\n")
        f.write(f"RPE (Translation): {get_stats(rpe_trans_3d, 'm')}\n")
        f.write(f"RPE (Rotation):    {get_stats(np.rad2deg(rpe_rot_3d), 'deg')}\n\n")
        
        f.write("--- METRICS 2D (Z ignored) ---\n")
        f.write(f"APE (Translation): {get_stats(ape_trans_2d, 'm')}\n")
        f.write(f"APE (Rotation):    {get_stats(np.rad2deg(ape_rot_2d), 'deg')}\n")
        f.write(f"RPE (Translation): {get_stats(rpe_trans_2d, 'm')}\n")
        f.write(f"RPE (Rotation):    {get_stats(np.rad2deg(rpe_rot_2d), 'deg')}\n")
    print(f"Métricas guardadas en: {metrics_path}")

    # --- GRÁFICO 1: Evolución de errores por eje ---
    fig_err, axs = plt.subplots(4, 1, figsize=(12, 15), sharex=True)
    time_steps = np.arange(min_len)
    
    # Errores de posición por eje
    pos_err = est_df[['x', 'y', 'z']].values - gt_df[['x', 'y', 'z']].values
    axs[0].plot(time_steps, pos_err[:, 0], label='Error en X')
    axs[0].set_ylabel('Error X (m)')
    axs[0].set_title('Evolución del Error Absoluto por Eje')
    axs[0].grid(True)
    
    axs[1].plot(time_steps, pos_err[:, 1], label='Error en Y', color='orange')
    axs[1].set_ylabel('Error Y (m)')
    axs[1].grid(True)

    axs[2].plot(time_steps, pos_err[:, 2], label='Error en Z', color='green')
    axs[2].set_ylabel('Error Z (m)')
    axs[2].grid(True)

    # Error de rotación (APE rotacional)
    axs[3].plot(time_steps, np.rad2deg(ape_rot_3d), label='Error Angular (Yaw)', color='red')
    axs[3].set_ylabel('Error Angular (°)')
    axs[3].set_xlabel('Paso de tiempo')
    axs[3].grid(True)
    
    fig_err.tight_layout()
    plt.savefig(error_plot_path, dpi=300)
    print(f"Gráfica de evolución de errores guardada en: {error_plot_path}")

    # --- GRÁFICO 2: Comparación de trayectorias 2D ---
    fig_comp, ax1 = plt.subplots(1, 1, figsize=(10, 10))
    ax1.plot(gt_df['x'], gt_df['y'], label='Trayectoria Real (GPS)', color='blue', linewidth=2)
    ax1.plot(est_df['x'], est_df['y'], label='Trayectoria Estimada (LIDAR)', color='red', linestyle='--', linewidth=2)
    ax1.set_title('Comparación de Trayectorias 2D (Plano XY)')
    ax1.set_xlabel('Coordenada X (m)')
    ax1.set_ylabel('Coordenada Y (m)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_aspect('equal', adjustable='box')
    plt.savefig(output_path, dpi=300)
    print(f"Gráfica de comparación 2D guardada en: {output_path}")

    # --- GRÁFICO 3: Visualización de trayectorias en 3D ---
    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    ax_3d.plot(gt_df['x'], gt_df['y'], gt_df['z'], label='Trayectoria Real (GPS)', color='blue', linewidth=2)
    ax_3d.plot(est_df['x'], est_df['y'], est_df['z'], label='Trayectoria Estimada (LIDAR)', color='red', linestyle='--', linewidth=2)
    ax_3d.set_title('Comparación de Trayectorias 3D')
    ax_3d.set_xlabel('X (m)'), ax_3d.set_ylabel('Y (m)'), ax_3d.set_zlabel('Z (m)')
    ax_3d.legend(), ax_3d.grid(True)
    plt.savefig(trajectory_3d_path, dpi=300)
    print(f"Visualización 3D guardada en: {trajectory_3d_path}")

if __name__ == '__main__':
    plot_trajectories()