import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import sys
from scipy.spatial.transform import Rotation
import os

def to_transform_matrices(df):
    """
    Convierte un DataFrame de poses (x, y, z, qx, qy, qz, qw)
    a una lista de matrices de transformación homogénea 4x4.
    """
    matrices = []
    for i, row in df.iterrows():
        T = np.eye(4)
        T[:3, 3] = row[['x', 'y', 'z']].values
        # Asegúrate de que el cuaternión esté normalizado para evitar errores
        quat = row[['qx', 'qy', 'qz', 'qw']].values
        norm = np.linalg.norm(quat)
        if norm > 1e-6:
            quat /= norm
        T[:3, :3] = Rotation.from_quat(quat).as_matrix()
        matrices.append(T)
    return np.array(matrices)

def calculate_ape(est_matrices, gt_matrices):
    """
    Calcula el Absolute Pose Error (APE) para traslación y rotación.
    Comúnmente, el APE de traslación también se conoce como ATE (Absolute Trajectory Error).
    """
    if len(est_matrices) != len(gt_matrices):
        raise ValueError("Las listas de matrices estimadas y reales deben tener la misma longitud.")
    
    trans_errors = []
    rot_errors = []
    for T_est, T_gt in zip(est_matrices, gt_matrices):
        # Error de traslación
        trans_error = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
        trans_errors.append(trans_error)
        
        # Error de rotación
        R_err_matrix = np.dot(T_gt[:3, :3].T, T_est[:3, :3]) # R_gt^T * R_est
        # El ángulo de rotación se puede obtener de la traza de la matriz de error
        # np.clip para evitar errores de dominio en arccos por imprecisiones numéricas
        trace_val = np.clip((np.trace(R_err_matrix) - 1) / 2, -1.0, 1.0)
        rot_error_angle = np.arccos(trace_val)
        rot_errors.append(rot_error_angle)
        
    return np.array(trans_errors), np.array(rot_errors)

def calculate_rpe(est_matrices, gt_matrices, delta=1):
    """Calcula el Relative Pose Error (RPE) para traslación y rotación para un delta dado."""
    if len(est_matrices) != len(gt_matrices):
        raise ValueError("Las listas de matrices estimadas y reales deben tener la misma longitud.")

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
        R_err_matrix = np.dot(T_rel_gt[:3, :3].T, T_rel_est[:3, :3])
        trace_val = np.clip((np.trace(R_err_matrix) - 1) / 2, -1.0, 1.0)
        rot_error_angle = np.arccos(trace_val)
        rot_errors.append(rot_error_angle)
        
    return np.array(trans_errors), np.array(rot_errors)

def read_file(path, col_names):
    """
    Lee un archivo usando delimitador de espacios en blanco si es .txt 
    o usando la configuración por defecto si es .csv.
    """
    ext = os.path.splitext(path)[-1]
    if ext.lower() == '.txt':
        return pd.read_csv(path, delim_whitespace=True, names=col_names, header=None)
    else:
        return pd.read_csv(path, names=col_names, header=None)
    
def get_stats(errors, unit='m'):
    """Calcula y formatea estadísticas (media, RMSE, std, min, max) de un array de errores."""
    if len(errors) == 0:
        return "N/A (no data)"
    stats = {
        'mean': np.mean(errors),
        'rmse': np.sqrt(np.mean(np.square(errors))),
        'std': np.std(errors),
        'min': np.min(errors),
        'max': np.max(errors)
    }
    return (f"Mean: {stats['mean']:.4f}, RMSE: {stats['rmse']:.4f}, "
            f"Std: {stats['std']:.4f}, Min: {stats['min']:.4f}, Max: {stats['max']:.4f} [{unit}]")

def analyze_trajectory(est_path, gt_path, estimator_name):
    """
    Realiza el análisis completo (lectura, sincronización, cálculo de métricas)
    para un par de trayectorias (estimada y ground truth).
    """
    print(f"\n--- Analizando Estimador: {estimator_name} ---")
    try:
        col_names = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        est_df = pd.read_csv(est_path, names=col_names, header=None) # Asumimos que no hay header
        gt_df = pd.read_csv(gt_path, names=col_names, header=None)
        print(f"Datos leídos: {len(est_df)} puntos estimados, {len(gt_df)} puntos de ground truth.")
    except FileNotFoundError as e:
        print(f"Error: No se pudo encontrar el fichero {e.filename}. Saltando este estimador.", file=sys.stderr)
        return None, None

    # Sincronizar trayectorias al mínimo común de muestras.
    # Una estrategia más avanzada podría ser la interpolación por timestamp si estuviera disponible.
    min_len = min(len(est_df), len(gt_df))
    if min_len == 0:
        print("Error: Uno de los ficheros está vacío. No se puede analizar.", file=sys.stderr)
        return None, None
    est_df = est_df.iloc[:min_len]
    gt_df = gt_df.iloc[:min_len]
    print(f"Trayectorias sincronizadas a {min_len} puntos.")

    # --- CÁLCULOS 3D ---
    est_matrices_3d = to_transform_matrices(est_df)
    gt_matrices_3d = to_transform_matrices(gt_df)
    ape_trans_3d, ape_rot_3d = calculate_ape(est_matrices_3d, gt_matrices_3d)
    rpe_trans_3d, rpe_rot_3d = calculate_rpe(est_matrices_3d, gt_matrices_3d)

    # --- CÁLCULOS 2D (ignorando Z y rotaciones pitch/roll) ---
    est_df_2d = est_df.copy()
    gt_df_2d = gt_df.copy()
    # Forzar Z a 0 para el cálculo de APE/RPE de traslación 2D
    est_df_2d['z'] = 0
    gt_df_2d['z'] = 0
    # Para la rotación 2D, solo nos interesa el Yaw. Extraemos el yaw y creamos nuevos cuaterniones.
    yaw_est = Rotation.from_quat(est_df[['qx', 'qy', 'qz', 'qw']].values).as_euler('xyz', degrees=False)[:, 2]
    yaw_gt = Rotation.from_quat(gt_df[['qx', 'qy', 'qz', 'qw']].values).as_euler('xyz', degrees=False)[:, 2]
    est_df_2d[['qx', 'qy', 'qz', 'qw']] = Rotation.from_euler('z', yaw_est).as_quat()
    gt_df_2d[['qx', 'qy', 'qz', 'qw']] = Rotation.from_euler('z', yaw_gt).as_quat()

    est_matrices_2d = to_transform_matrices(est_df_2d)
    gt_matrices_2d = to_transform_matrices(gt_df_2d)
    ape_trans_2d, ape_rot_2d = calculate_ape(est_matrices_2d, gt_matrices_2d)
    rpe_trans_2d, rpe_rot_2d = calculate_rpe(est_matrices_2d, gt_matrices_2d)

    # Recopilar todas las métricas en un diccionario
    metrics = {
        "name": estimator_name,
        "ape_trans_3d": get_stats(ape_trans_3d, 'm'),
        "ape_rot_3d": get_stats(np.rad2deg(ape_rot_3d), 'deg'),
        "rpe_trans_3d": get_stats(rpe_trans_3d, 'm'),
        "rpe_rot_3d": get_stats(np.rad2deg(rpe_rot_3d), 'deg'),
        "ape_trans_2d": get_stats(ape_trans_2d, 'm'),
        "ape_rot_2d": get_stats(np.rad2deg(ape_rot_2d), 'deg'),
        "rpe_trans_2d": get_stats(rpe_trans_2d, 'm'),
        "rpe_rot_2d": get_stats(np.rad2deg(rpe_rot_2d), 'deg'),
    }
    
    # Datos para los gráficos
    plot_data = {
        "est_df": est_df, "gt_df": gt_df,
        "pos_err": est_df[['x', 'y', 'z']].values - gt_df[['x', 'y', 'z']].values,
        "ape_rot_3d": ape_rot_3d,
        "min_len": min_len
    }

    return metrics, plot_data

def generate_individual_plots(plot_data, estimator_name, output_dir):
    """Genera y guarda los gráficos individuales para un estimador."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    est_df = plot_data["est_df"]
    gt_df = plot_data["gt_df"]
    
    # --- GRÁFICO 2D INDIVIDUAL ---
    plt.figure(figsize=(10, 10))
    plt.plot(gt_df['x'], gt_df['y'], label='Ground Truth', color='blue', linewidth=2)
    plt.plot(est_df['x'], est_df['y'], label=f'Estimada ({estimator_name})', color='red', linestyle='--', linewidth=2)
    plt.title(f'Comparación de Trayectorias 2D - {estimator_name}')
    plt.xlabel('Coordenada X (m)'), plt.ylabel('Coordenada Y (m)')
    plt.legend(), plt.grid(True), plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(os.path.join(output_dir, f'trajectory_2d_{estimator_name}.png'), dpi=300)
    plt.close()
    
    print(f"Gráficos individuales para {estimator_name} guardados en el directorio '{output_dir}'.")

def generate_combined_trajectory_plot(all_plot_data, all_metrics, output_dir):
    """Genera un único gráfico comparando las trayectorias de todos los estimadores."""
    if len(all_plot_data) < 1:
        return

    plt.figure(figsize=(12, 12))
    
    # Graficar el Ground Truth (usamos el del primer estimador como referencia)
    gt_df_ref = all_plot_data[0]["gt_df"]
    plt.plot(gt_df_ref['x'], gt_df_ref['y'], label='Ground Truth', color='black', linewidth=3, zorder=1)

    # Actualizamos la lista de colores para tener cinco colores únicos
    colors = ['red', 'green', 'purple', 'orange', 'cyan']
    for i, plot_data in enumerate(all_plot_data):
        est_df = plot_data["est_df"]
        estimator_name = all_metrics[i]["name"]
        color = colors[i % len(colors)]
        plt.plot(est_df['x'], est_df['y'], label=f'Estimada ({estimator_name})', color=color, linestyle='--', linewidth=2, zorder=2+i)

    plt.title('Comparación Combinada de Trayectorias', fontsize=16)
    plt.xlabel('Coordenada X (m)'), plt.ylabel('Coordenada Y (m)')
    plt.legend(), plt.grid(True), plt.gca().set_aspect('equal', adjustable='box')
    
    output_path = os.path.join(output_dir, 'combined_trajectory_comparison.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Gráfico de trayectoria combinada guardado en: {output_path}")
    
def generate_combined_error_plot(all_plot_data, all_metrics, output_dir):
    """Genera un único gráfico comparando los errores de todos los estimadores."""
    if len(all_plot_data) < 1:
        return

    fig, axs = plt.subplots(4, 1, figsize=(15, 18), sharex=True)
    fig.suptitle('Comparación Combinada de Errores Absolutos', fontsize=16)
    
    # Determinar la longitud mínima para el eje de tiempo común
    min_len_common = min(p["min_len"] for p in all_plot_data)
    time_steps = np.arange(min_len_common)
    
    # Actualizamos la lista de colores para tener cinco colores únicos
    colors = ['red', 'green', 'purple', 'orange', 'cyan']
    
    # Títulos y etiquetas para los subplots
    plot_info = [
        {'title': 'Error en Eje X', 'ylabel': 'Error X (m)', 'data_idx': 0},
        {'title': 'Error en Eje Y', 'ylabel': 'Error Y (m)', 'data_idx': 1},
        {'title': 'Error en Eje Z', 'ylabel': 'Error Z (m)', 'data_idx': 2},
        {'title': 'Error Angular 3D', 'ylabel': 'Error Angular (°)', 'data_idx': None} # Caso especial
    ]

    for i, info in enumerate(plot_info):
        ax = axs[i]
        for j, plot_data in enumerate(all_plot_data):
            estimator_name = all_metrics[j]["name"]
            color = colors[j % len(colors)]
            
            if info['data_idx'] is not None: # Errores de posición
                error_data = plot_data["pos_err"][:min_len_common, info['data_idx']]
            else: # Error de rotación
                error_data = np.rad2deg(plot_data["ape_rot_3d"][:min_len_common])
            
            ax.plot(time_steps, error_data, label=f'{estimator_name}', color=color, linewidth=1.5)
        
        ax.set_title(info['title'])
        ax.set_ylabel(info['ylabel'])
        ax.grid(True)
        ax.legend()

    axs[-1].set_xlabel('Paso de tiempo')
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    output_path = os.path.join(output_dir, 'combined_error_comparison.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Gráfico de errores combinados guardado en: {output_path}")


def main():
    # --- CONFIGURACIÓN DE FICHEROS ---
    base_path = '/home/david/ros2_ws/Data/'
    output_dir = os.path.join(base_path, 'comparison_results')

    estimators = {
        "KISS_ICP": {
            "est": os.path.join(base_path, "odom_est_traj_kiss_cyclone.csv"),
            "gt": os.path.join(base_path, "odom_gt_traj_kiss_cyclone.csv")
        },
        "MOLA": {
            "est": os.path.join(base_path, "odom_est_traj_mola.csv"),
            "gt": os.path.join(base_path, "odom_gt_traj_mola.csv")
        },
        "Traj-LO": {
            "est": os.path.join(base_path, "odom_est_traj.csv"),
            "gt": os.path.join(base_path, "odom_gt_traj.csv")
        },
        "SIMPLE": {
            "est": os.path.join(base_path, "odom_est_traj_simple.csv"),
            "gt": os.path.join(base_path, "odom_gt_traj_simple.csv")
        },
        "DLO": {
            "est": os.path.join(base_path, "odom_est_traj_dlo.csv"),
            "gt": os.path.join(base_path, "odom_gt_traj_dlo.csv")
        },
    }
    
    all_metrics = []
    all_plot_data = []
    for name, paths in estimators.items():
        metrics, plot_data = analyze_trajectory(paths["est"], paths["gt"], name)
        if metrics and plot_data:
            all_metrics.append(metrics)
            all_plot_data.append(plot_data)
            # Generar gráficos individuales (opcional)
            generate_individual_plots(plot_data, name, output_dir)

    # --- GENERAR GRÁFICOS COMBINADOS ---
    if len(all_plot_data) > 0:
        generate_combined_trajectory_plot(all_plot_data, all_metrics, output_dir)
        generate_combined_error_plot(all_plot_data, all_metrics, output_dir)
    
    # --- GUARDAR INFORME COMPARATIVO ---
    if all_metrics:
        metrics_path = os.path.join(output_dir, "comparison_metrics.txt")
        with open(metrics_path, "w") as f:
            f.write("======= INFORME COMPARATIVO DE ESTIMADORES DE TRAYECTORIA =======\n")
            for metrics in all_metrics:
                f.write("\n" + "="*50 + "\n")
                f.write(f"  ESTIMADOR: {metrics['name']}\n")
                f.write("="*50 + "\n\n")
                
                f.write("--- MÉTRICAS 3D ---\n")
                f.write(f"APE (Traslación): {metrics['ape_trans_3d']}\n")
                f.write(f"APE (Rotación):   {metrics['ape_rot_3d']}\n")
                f.write(f"RPE (Traslación): {metrics['rpe_trans_3d']}\n")
                f.write(f"RPE (Rotación):   {metrics['rpe_rot_3d']}\n\n")
                
                f.write("--- MÉTRICAS 2D (ignorando Z y rotaciones pitch/roll) ---\n")
                f.write(f"APE (Traslación): {metrics['ape_trans_2d']}\n")
                f.write(f"APE (Rotación):   {metrics['ape_rot_2d']}\n")
                f.write(f"RPE (Traslación): {metrics['rpe_trans_2d']}\n")
                f.write(f"RPE (Rotación):   {metrics['rpe_rot_2d']}\n")
            
            print(f"\nInforme comparativo de métricas guardado en: {metrics_path}")
    else:
        print("\nNo se pudo generar ningún informe ya que no se procesó ningún estimador.")

if __name__ == '__main__':
    main()
