from typing import Tuple
import time

import numpy as np
import numpy.typing
import matplotlib.pyplot as plt

import nidaqmx
from nidaqmx.constants import AcquisitionType, ProductCategory, Edge

def get_terminal_name_with_dev_prefix(task: nidaqmx.Task, terminal_name: str) -> str:
    """Получает имя терминала с префиксом устройства.
    """
    for device in task.devices:
        if device.product_category not in [
            ProductCategory.C_SERIES_MODULE,
            ProductCategory.SCXI_MODULE,
        ]:
            return f"/{device.name}/{terminal_name}"
    raise RuntimeError("Подходящее устройство не найдено в задаче.")

def generate_triangle_wave(
    frequency: float,
    amplitude: float,
    sampling_rate: float,
    number_of_samples: int,
    phase_in: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """
    Генерирует один период треугольной волны с требуемыми параметрами.

    Волна генерируется так, что при phase_in = 0 на старте значение равно 0,
    затем линейно растёт до +amplitude в T/4, затем падает до 0 в T/2,
    затем до -amplitude в 3T/4 и возвращается к 0 в T.
    
    Args:
        frequency: Частота волны (Гц). Для 1/60 Гц указываем 1/60.
        amplitude: Амплитуда волны (пик – амплитуда). Здесь 7.
        sampling_rate: Частота дискретизации (С/с).
        number_of_samples: Количество генерируемых отсчетов.
        phase_in: Начальная фаза (в секундах). Обычно 0.
        
    Returns:
        Кортеж, содержащий сгенерированный массив и новую фазу (смещение по времени внутри периода).
    """
    duration_time = number_of_samples / sampling_rate  # длительность в секундах
    t = np.linspace(0, duration_time, number_of_samples, endpoint=False) + phase_in
    T = 1.0 / frequency  # период волны
    # Вычисляем нормализованную фазу, сдвинутую на 0.25 так, чтобы при t=0 значение было 0.
    phi = (t / T + 0.25) % 1.0  
    # Если phi < 0.5, волна растёт, иначе — падает:
    triangle = np.where(phi < 0.5,
                        4 * amplitude * phi - amplitude,   # при phi=0: -amplitude, при phi=0.25: 0, при phi=0.5: +amplitude
                        -4 * amplitude * phi + 3 * amplitude)  # при phi=0.5: +amplitude, при phi=0.75: 0, при phi=0: -amplitude
    phase_out = (phase_in + duration_time) % T
    return triangle, phase_out

def main():
    """Непрерывное приобретение и генерация данных с синхронизацией входа и выхода."""
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

        # Настройка аналогового входа
        ai_task.ai_channels.add_ai_voltage_chan("Dev1/ai0")
        ai_task.timing.cfg_samp_clk_timing(sampling_rate, sample_mode=AcquisitionType.CONTINUOUS)
        ai_task.register_every_n_samples_acquired_into_buffer_event(number_of_samples, callback)
        # Получаем имя терминала для старта
        terminal_name = get_terminal_name_with_dev_prefix(ai_task, "ai/StartTrigger")

        # Настройка аналогового выхода
        ao_task.ao_channels.add_ao_voltage_chan("Dev1/ao0")
        ao_task.timing.cfg_samp_clk_timing(sampling_rate, sample_mode=AcquisitionType.CONTINUOUS)
        ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(terminal_name)

        actual_sampling_rate = ao_task.timing.samp_clk_rate
        print(f"Actual sampling rate: {actual_sampling_rate:g} S/s")

        # Генерируем треугольную волну с частотой 1/60 Гц и амплитудой 7
        ao_data, _ = generate_triangle_wave(
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
