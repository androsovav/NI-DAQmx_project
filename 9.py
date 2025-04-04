from typing import Tuple
import time

import numpy as np
import matplotlib.pyplot as plt

import nidaqmx
from nidaqmx.constants import AcquisitionType, ProductCategory, Edge
from scipy import signal  # Импортируем SciPy для генерации треугольной волны

def get_terminal_name_with_dev_prefix(task: nidaqmx.Task, terminal_name: str) -> str:
    """Получает имя терминала с префиксом устройства."""
    for device in task.devices:
        if device.product_category not in [
            ProductCategory.C_SERIES_MODULE,
            ProductCategory.SCXI_MODULE,
        ]:
            return f"/{device.name}/{terminal_name}"
    raise RuntimeError("Подходящее устройство не найдено в задаче.")

def generate_triangle_wave_scipy(
    frequency: float,
    amplitude: float,
    sampling_rate: float,
    number_of_samples: int,
) -> np.ndarray:
    """
    Генерирует один период треугольной волны с помощью scipy.signal.sawtooth.
    
    Args:
        frequency: Частота треугольной волны (Гц). Например, для 1/60 Гц.
        amplitude: Амплитуда волны (значения от -amplitude до +amplitude). Здесь 7.
        sampling_rate: Частота дискретизации (С/с).
        number_of_samples: Количество сэмплов для генерации одного периода.
    
    Returns:
        Массив значений треугольной волны.
    """
    # Создаем временной вектор для одного блока сэмплов
    t = np.linspace(0, number_of_samples / sampling_rate, number_of_samples, endpoint=False)
    # scipy.signal.sawtooth с width=0.5 генерирует симметричную треугольную волну от -1 до +1
    triangle_wave = amplitude * signal.sawtooth(2 * np.pi * frequency * t, width=0.5)
    return triangle_wave

def main():
    total_read = 0
    number_of_samples = 1000
    sampling_rate = 1000.0
    acquired_data = []
    timestamps = []
    start_time = time.time()

    with nidaqmx.Task() as ai_task, nidaqmx.Task() as ao_task:

        def callback(task_handle, every_n_samples_event_type, number_of_samples, callback_data):
            nonlocal total_read, acquired_data, timestamps
            read = ai_task.read(number_of_samples_per_channel=number_of_samples)
            current_time = time.time() - start_time
            timestamps.extend(np.linspace(current_time - len(read)/sampling_rate, 
                                          current_time, 
                                          len(read), 
                                          endpoint=False))
            acquired_data.extend(read)
            total_read += len(read)
            print(f"Acquired data: {len(read)} samples. Total {total_read}.", end="\r")
            return 0

        ai_task.ai_channels.add_ai_voltage_chan("Dev1/ai0")
        ai_task.timing.cfg_samp_clk_timing(sampling_rate, sample_mode=AcquisitionType.CONTINUOUS)
        ai_task.register_every_n_samples_acquired_into_buffer_event(number_of_samples, callback)
        terminal_name = get_terminal_name_with_dev_prefix(ai_task, "ai/StartTrigger")

        ao_task.ao_channels.add_ao_voltage_chan("Dev1/ao0")
        ao_task.timing.cfg_samp_clk_timing(sampling_rate, sample_mode=AcquisitionType.CONTINUOUS)
        ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(terminal_name)

        actual_sampling_rate = ao_task.timing.samp_clk_rate
        print(f"Actual sampling rate: {actual_sampling_rate:g} S/s")

        # Генерируем треугольную волну с частотой 1/60 Гц и амплитудой 7,
        # используя scipy.signal.sawtooth. Здесь number_of_samples=1000 для одного блока.
        ao_data = generate_triangle_wave_scipy(
            frequency=1/60,
            amplitude=7.0,
            sampling_rate=actual_sampling_rate,
            number_of_samples=number_of_samples,
        )

        ao_task.write(ao_data)
        ao_task.start()
        ai_task.start()

        input("Acquiring samples continuously. Press Enter to stop.\n")

        ai_task.stop()
        ao_task.stop()

        print(f"\nAcquired {total_read} total samples.")
        
        # Построение графика полученных данных
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, acquired_data, label='Acquired Data')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title('Acquired Signal vs Time')
        plt.grid(True)
        plt.legend()
        
        plot_filename = 'acquired_signal_plot.png'
        plt.savefig(plot_filename)
        print(f"Plot saved as {plot_filename}")
        
        plt.show()

if __name__ == "__main__":
    main()
