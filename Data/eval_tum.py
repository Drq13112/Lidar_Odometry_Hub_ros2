#!/usr/bin/env python

print("loading required evo modules")
from evo.core import trajectory, sync, metrics
from evo.tools import file_interface
import numpy as np

print("loading trajectories")
traj_ref = file_interface.read_tum_trajectory_file("traj_gt_kiss_cyclone.txt")
traj_est = file_interface.read_tum_trajectory_file("traj_est_kiss_cyclone.txt")

print("registering and aligning trajectories")

# --- INICIO DE LA MODIFICACIÓN: RECORTAR TRAYECTORIAS ---
# Obtener la longitud de cada trayectoria
len_ref = len(traj_ref.timestamps)
len_est = len(traj_est.timestamps)

# Encontrar la longitud mínima
min_len = min(len_ref, len_est)

if min_len == 0:
    print("Error: Una de las trayectorias está vacía. No se puede continuar.")
    exit()

print(f"Longitud original: ref={len_ref}, est={len_est}")
print(f"Recortando ambas trayectorias a la longitud mínima: {min_len}")

# --- INICIO DE LA MODIFICACIÓN: RE-CREAR TRAYECTORIAS CON DATOS RECORTADOS ---
# Crear nuevos objetos de trayectoria con los datos recortados, ya que los
# atributos de los objetos existentes son de solo lectura.
traj_ref = trajectory.PoseTrajectory3D(
    timestamps=traj_ref.timestamps[:min_len],
    positions_xyz=traj_ref.positions_xyz[:min_len],
    orientations_quat_wxyz=traj_ref.orientations_quat_wxyz[:min_len]
)

traj_est = trajectory.PoseTrajectory3D(
    timestamps=traj_est.timestamps[:min_len],
    positions_xyz=traj_est.positions_xyz[:min_len],
    orientations_quat_wxyz=traj_est.orientations_quat_wxyz[:min_len]
)
# --- FIN DE LA MODIFICACIÓN ---

# Associate trajectories by timestamps
# traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff=0.1)
# Align trajectories using umeyama alignment (rotation, translation)
# traj_est.align(traj_ref, correct_scale=False)


print("calculating APE")
data = (traj_ref, traj_est)

# Calculate APE for translation (position error)
ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
ape_metric.process_data(data)
ape_statistics = ape_metric.get_all_statistics()
print("APE translation statistics:")
print(f"  - Mean: {ape_statistics['mean']:.4f} m")
print(f"  - Std: {ape_statistics['std']:.4f} m")
print(f"  - RMSE: {ape_statistics['rmse']:.4f} m")
print(f"  - Min: {ape_statistics['min']:.4f} m")
print(f"  - Max: {ape_statistics['max']:.4f} m")

# Calculate APE for rotation (orientation error)
ape_metric_rotation = metrics.APE(metrics.PoseRelation.rotation_part)
ape_metric_rotation.process_data(data)
ape_statistics_rotation = ape_metric_rotation.get_all_statistics()
print("\nAPE rotation statistics:")
print(f"  - Mean: {ape_statistics_rotation['mean']:.4f} rad ({np.degrees(ape_statistics_rotation['mean']):.4f} deg)")
print(f"  - Std: {ape_statistics_rotation['std']:.4f} rad")
print(f"  - RMSE: {ape_statistics_rotation['rmse']:.4f} rad")

# Calculate RPE for translation
rpe_metric = metrics.RPE(metrics.PoseRelation.translation_part, delta=1.0)
rpe_metric.process_data(data)
rpe_statistics = rpe_metric.get_all_statistics()
print("\nRPE translation statistics:")
print(f"  - Mean: {rpe_statistics['mean']:.4f} m")
print(f"  - Std: {rpe_statistics['std']:.4f} m")
print(f"  - RMSE: {rpe_statistics['rmse']:.4f} m")
print(f"  - Min: {rpe_statistics['min']:.4f} m")
print(f"  - Max: {rpe_statistics['max']:.4f} m")

# Calculate RPE for rotation
rpe_rot_metric = metrics.RPE(metrics.PoseRelation.rotation_part, delta=1.0)
rpe_rot_metric.process_data(data)
rpe_rot_statistics = rpe_rot_metric.get_all_statistics()
print("\nRPE rotation statistics:")
print(f"  - Mean: {rpe_rot_statistics['mean']:.4f} rad ({np.degrees(rpe_rot_statistics['mean']):.4f} deg)")
print(f"  - Std: {rpe_rot_statistics['std']:.4f} rad")
print(f"  - RMSE: {rpe_rot_statistics['rmse']:.4f} rad")

print("\nloading plot modules")
from evo.tools import plot
import matplotlib.pyplot as plt

print("plotting")
plot_collection = plot.PlotCollection("Trajectory Evaluation")

# APE metric values
fig_1 = plt.figure(figsize=(10, 8))
plot.error_array(fig_1.gca(), ape_metric.error, statistics=ape_statistics,
                name="APE (translation)", title=str(ape_metric))
plot_collection.add_figure("APE", fig_1)

# Trajectory colormapped with APE
fig_2 = plt.figure(figsize=(10, 8))
plot_mode = plot.PlotMode.xyz  # Use 3D plot mode
ax = plot.prepare_axis(fig_2, plot_mode)
plot.traj(ax, plot_mode, traj_ref, '--', 'gray', 'reference')
plot.traj_colormap(ax, traj_est, ape_metric.error, plot_mode,
                min_map=ape_statistics["min"],
                max_map=ape_statistics["max"],
                title="APE mapped onto trajectory")
plot_collection.add_figure("Trajectory with APE", fig_2)

# RPE metric values
fig_3 = plt.figure(figsize=(10, 8))
plot.error_array(fig_3.gca(), rpe_metric.error, statistics=rpe_statistics,
                name="RPE (translation)", title=str(rpe_metric))
plot_collection.add_figure("RPE", fig_3)

# Calculate and plot speed
fig_4 = plt.figure(figsize=(10, 8))
plot_mode = plot.PlotMode.xyz
ax = plot.prepare_axis(fig_4, plot_mode)
speeds = [
    trajectory.calc_speed(traj_est.positions_xyz[i],
                        traj_est.positions_xyz[i + 1],
                        traj_est.timestamps[i], traj_est.timestamps[i + 1])
    for i in range(len(traj_est.positions_xyz) - 1)
]
speeds.append(0)
plot.traj(ax, plot_mode, traj_ref, '--', 'gray', 'reference')
plot.traj_colormap(ax, traj_est, speeds, plot_mode, min_map=min(speeds),
                max_map=max(speeds), title="Speed mapped onto trajectory")
plot_collection.add_figure("Trajectory with Speed", fig_4)

# Show all plots
plot_collection.show()

# Save statistics to a file
with open("trajectory_metrics.txt", "w") as f:
    f.write("=== TRAJECTORY EVALUATION METRICS ===\n\n")
    f.write(f"APE (translation):\n")
    f.write(f"  - Mean: {ape_statistics['mean']:.4f} m\n")
    f.write(f"  - Std: {ape_statistics['std']:.4f} m\n")
    f.write(f"  - RMSE: {ape_statistics['rmse']:.4f} m\n")
    f.write(f"  - Min: {ape_statistics['min']:.4f} m\n")
    f.write(f"  - Max: {ape_statistics['max']:.4f} m\n\n")
    
    f.write(f"APE (rotation):\n")
    f.write(f"  - Mean: {ape_statistics_rotation['mean']:.4f} rad ({np.degrees(ape_statistics_rotation['mean']):.4f} deg)\n")
    f.write(f"  - RMSE: {ape_statistics_rotation['rmse']:.4f} rad\n\n")
    
    f.write(f"RPE (translation):\n")
    f.write(f"  - Mean: {rpe_statistics['mean']:.4f} m\n")
    f.write(f"  - Std: {rpe_statistics['std']:.4f} m\n")
    f.write(f"  - RMSE: {rpe_statistics['rmse']:.4f} m\n\n")
    
    f.write(f"RPE (rotation):\n")
    f.write(f"  - Mean: {rpe_rot_statistics['mean']:.4f} rad ({np.degrees(rpe_rot_statistics['mean']):.4f} deg)\n")
    f.write(f"  - RMSE: {rpe_rot_statistics['rmse']:.4f} rad\n")

print("Metrics saved to trajectory_metrics.txt")
