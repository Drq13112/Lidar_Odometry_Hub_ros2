import matplotlib.pyplot as plt
import numpy as np

# --- Datos extraídos del fichero de comparación ---
# Nombres de los algoritmos
algorithms = ['KISS-ICP', 'MOLA', 'DLO', 'SIMPLE', 'Traj-LO']

# Tiempos de cómputo medios en ms
times = [19.7, 46.0, 33.9, 100.0, 77.0]

# Métricas de APE de traslación (Error Absoluto de Pose)
ape_means = np.array([2.4176, 1.2978, 11.4501, 8.0621, 89.6279])
ape_mins = np.array([0.6790, 0.0652, 0.000, 1.1556, 0.0000])
ape_maxs = np.array([7.6660, 3.8758, 22.1543, 18.5306, 24.5445])

# --- INICIO DE LA MODIFICACIÓN: AJUSTAR ERRORES MÍNIMOS ---
# Si el error mínimo es menor de 0.05, se computa como 0.
ape_mins[ape_mins < 0.05] = 0
# --- FIN DE LA MODIFICACIÓN ---

# --- Preparación de los datos para el gráfico ---
# Las barras de error se definen como la diferencia desde la media
# Formato: [error_inferior, error_superior]
lower_error = ape_means - ape_mins
upper_error = ape_maxs - ape_means
asymmetric_error = [lower_error, upper_error]

# Colores para cada algoritmo para una mejor visualización
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# --- Creación del Gráfico ---
fig, ax = plt.subplots(figsize=(12, 8))

# Dibujar cada punto con su barra de error
for i in range(len(algorithms)):
    # Marcar 'SIMPLE' con un asterisco
    label = algorithms[i]
    if algorithms[i] == 'SIMPLE':
        label += '*'
        
    ax.errorbar(times[i], ape_means[i], yerr=np.array([[lower_error[i]], [upper_error[i]]]),
                fmt='o', color=colors[i], ecolor=colors[i], elinewidth=3,
                capsize=5, markersize=8, label=label, alpha=0.8)

# --- Estilo y Etiquetas del Gráfico ---
ax.set_title('Comparación de Algoritmos: Precisión (APE) vs. Tiempo de Cómputo', fontsize=16)
ax.set_xlabel('Tiempo de Cómputo Medio (ms)', fontsize=12)
# --- INICIO DE LA MODIFICACIÓN: CAMBIAR ETIQUETA DEL EJE Y ---
ax.set_ylabel('Error Absoluto de Pose (APE) Medio (m)', fontsize=12)

# --- INICIO DE LA MODIFICACIÓN: ELIMINAR ESCALA LOGARÍTMICA ---
# ax.set_yscale('log') # Se comenta esta línea para usar una escala lineal
# --- FIN DE LA MODIFICACIÓN ---

# Añadir una leyenda para identificar los algoritmos
# y la nota sobre el tiempo de 'SIMPLE'
legend = ax.legend(title="Algoritmos", fontsize=10)
plt.setp(legend.get_title(),fontsize=12)
ax.text(0.98, 0.02, '*Tiempo para SIMPLE no medido, asignado a 100 ms.',
        transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

ax.grid(True, which="both", ls="--", linewidth=0.5)
fig.tight_layout()

# Guardar el gráfico en un archivo
output_filename = 'comparison_ape_vs_time_linear.png'
plt.savefig(output_filename, dpi=300)

print(f"Gráfico guardado como '{output_filename}'")

# Mostrar el gráfico