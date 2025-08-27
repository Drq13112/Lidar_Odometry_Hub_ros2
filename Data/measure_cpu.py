import psutil
import time
import argparse
import sys
import csv

def monitor_cpu_by_pid(pid, interval):
    """
    Monitoriza el uso de CPU (total y por núcleo del sistema) y memoria de un proceso dado su PID.
    Guarda los datos en un archivo CSV.
    """
    print(f"DEBUG: psutil version being used is {psutil.__version__}")
    print(f"Monitorizando el proceso con PID: {pid}...")
    print("Presiona Ctrl+C para detener el monitor.")

    # Inicializa el CSV
    num_cpus = psutil.cpu_count()
    fieldnames = ['timestamp', 'cpu_total', 'memory_MB'] + [f'cpu{n}_percent' for n in range(num_cpus)]
    with open('cpu_usage_log.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        try:
            process = psutil.Process(pid)
            process.cpu_percent(interval=None)  # Inicializa la medición

            while True:
                if not process.is_running():
                    print(f"\nEl proceso con PID {pid} ha terminado.")
                    break

                cpu_usage = process.cpu_percent(interval=interval)
                memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
                cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)

                # Guarda en CSV
                row = {
                    'timestamp': time.time(),
                    'cpu_total': cpu_usage,
                    'memory_MB': memory_usage,
                }
                for i, val in enumerate(cpu_per_core):
                    row[f'cpu{i}_percent'] = val
                writer.writerow(row)

                # Muestra por pantalla
                # core_usage_str = ", ".join([f"{usage:5.1f}%" for usage in cpu_per_core])
                # sys.stdout.write(
                #     f"\rCPU Total: {cpu_usage:6.2f}% | Memoria: {memory_usage:.2f} MB | Por núcleo: [{core_usage_str}]"
                # )
                # sys.stdout.flush()

        except psutil.NoSuchProcess:
            print(f"\nError: No se encontró ningún proceso con el PID {pid}.", file=sys.stderr)
        except KeyboardInterrupt:
            print("\n\nMonitorización detenida por el usuario.")
        finally:
            print("\nSaliendo del script.")

def main():
    parser = argparse.ArgumentParser(
        description="Monitoriza el uso de CPU y memoria de un proceso dado su PID y guarda los datos en un CSV.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--pid",
        type=int,
        required=True,
        help="El PID (Process ID) del proceso a monitorizar."
    )
    parser.add_argument(
        "-i", "--interval",
        type=float,
        default=1.0,
        help="Intervalo de actualización en segundos para la medición. Default: 1.0s."
    )
    parser.epilog = """
    Ejemplo de uso con un nodo de ROS 2:
    pidof <nombre_ejecutable> | xargs -I {} python3 measure_cpu.py --pid {}
    """
    args = parser.parse_args()
    monitor_cpu_by_pid(args.pid, args.interval)

if __name__ == '__main__':
    main()