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
        frequency: Частота треугольной волны (Гц). Для 1/60 Гц указываем 1/60.
        amplitude: Амплитуда волны (значения от -amplitude до +amplitude). Здесь 7.
        sampling_rate: Частота дискретизации (С/с).
        number_of_samples: Количество сэмплов для генерации одного блока.
    
    Returns:
        Массив значений треугольной волны.
    """
    t = np.linspace(0, number_of_samples / sampling_rate, number_of_samples, endpoint=False)
    # Функция scipy.signal.sawtooth с параметром width=0.5 генерирует симметричную треугольную волну от -1 до +1.
    triangle_wave = amplitude * signal.sawtooth(2 * np.pi * frequency * t, width=0.5)
    return triangle_wave

def main():
    total_read = 0
    number_of_samples = 1000
    sampling_rate = 1000.0
    acquired_data = []
    timestamps = []
    start_time = time.time()

    # Генерируем данные для аналогового выхода: треугольная волна с частотой 1/60 Гц и амплитудой 7.
    # Эти данные будут подаваться на AO.
    ao_data = generate_triangle_wave_scipy(
        frequency=1,
        amplitude=7.0,
        sampling_rate=sampling_rate,
        number_of_samples=number_of_samples,
    )

    with nidaqmx.Task() as ai_task, nidaqmx.Task() as ao_task:

        def callback(task_handle, event_type, n_samples, callback_data):
            nonlocal total_read, acquired_data, timestamps
            read = ai_task.read(number_of_samples_per_channel=n_samples)
            current_time = time.time() - start_time
            timestamps.extend(np.linspace(current_time - len(read)/sampling_rate, 
                                          current_time, 
                                          len(read), 
                                          endpoint=False))
            acquired_data.extend(read)
            total_read += len(read)
            print(f"Acquired: {len(read)} samples. Total {total_read}.", end="\r")
            return 0

        # Настройка аналогового входа
        ai_task.ai_channels.add_ai_voltage_chan("Dev1/ai0")
        ai_task.timing.cfg_samp_clk_timing(sampling_rate, sample_mode=AcquisitionType.CONTINUOUS)
        ai_task.register_every_n_samples_acquired_into_buffer_event(number_of_samples, callback)
        terminal_name = get_terminal_name_with_dev_prefix(ai_task, "ai/StartTrigger")

        # Настройка аналогового выхода
        ao_task.ao_channels.add_ao_voltage_chan("Dev1/ao0")
        ao_task.timing.cfg_samp_clk_timing(sampling_rate, sample_mode=AcquisitionType.CONTINUOUS)
        ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(terminal_name)

        actual_sampling_rate = ao_task.timing.samp_clk_rate
        print(f"Actual sampling rate: {actual_sampling_rate:g} S/s")

        # Записываем данные треугольной волны в буфер AO
        ao_task.write(ao_data)
        # Запускаем задачи: сначала AO, затем AI (или наоборот, если настроен общий стартовый триггер)
        ao_task.start()
        ai_task.start()

        input("Acquiring samples continuously. Press Enter to stop.\n")

        ai_task.stop()
        ao_task.stop()

        print(f"\nAcquired {total_read} total samples.")

    # Сохраним график выходной треугольной волны (то есть, данные, которые были поданы на AO)
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(ao_data)), ao_data, label='Output Triangular Wave')
    plt.xlabel('Номер отсчёта')
    plt.ylabel('Напряжение (V)')
    plt.title('График треугольной волны, поданной на выход')
    plt.grid(True)
    plt.legend()
    output_plot_filename = 'output_triangle_wave.png'
    plt.savefig(output_plot_filename)
    print(f"Output triangular wave plot saved as {output_plot_filename}")
    plt.show()

if __name__ == "__main__":
    main()
