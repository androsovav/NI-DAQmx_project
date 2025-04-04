from typing import Tuple
import time
import numpy as np
import matplotlib.pyplot as plt
import nidaqmx
from nidaqmx.constants import AcquisitionType, ProductCategory, Edge
from scipy import signal  # Для генерации треугольной волны

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
    Генерирует блок значений треугольной волны с фазовым сдвигом на 1/4 периода,
    используя scipy.signal.sawtooth.
    
    Args:
        frequency: Частота треугольной волны (Гц). Например, для 1/60 Гц.
        amplitude: Амплитуда волны (волна будет в диапазоне -amplitude ... +amplitude).
        sampling_rate: Частота дискретизации (С/с).
        number_of_samples: Количество сэмплов для генерации блока.
    
    Returns:
        Массив значений треугольной волны.
    """
    t = np.linspace(0, number_of_samples / sampling_rate, number_of_samples, endpoint=False)
    phase_shift = np.pi / 2  # сдвиг на 1/4 периода (π/2 радиан)
    triangle_wave = amplitude * signal.sawtooth(2 * np.pi * frequency * t + phase_shift, width=0.5)
    return triangle_wave

def main(
    frequency: float = 1/60,
    amplitude: float = 7.0,
    sampling_rate: float = 10000.0,
    duration: float = 60.0,
) -> str:
    """
    Запускает синхронное приобретение и генерацию данных, сохраняет датасет в CSV.
    
    Параметры:
        frequency: Частота треугольной волны (Гц).
        amplitude: Амплитуда треугольной волны.
        sampling_rate: Частота дискретизации (С/с).
        duration: Время измерения в секундах.
        
    Число сэмплов вычисляется как sampling_rate * duration.
    CSV-файл содержит три столбца: время (начинается с 0), входные данные и выходные данные.
    
    Возвращает:
        Путь к сохранённому CSV-файлу.
    """
    total_read = 0
    number_of_samples = int(sampling_rate * duration)
    acquired_data = []
    timestamps = []
    start_time = time.time()

    # Генерируем данные для AO: треугольная волна
    ao_data = generate_triangle_wave_scipy(
        frequency=frequency,
        amplitude=amplitude,
        sampling_rate=sampling_rate,
        number_of_samples=number_of_samples,
    )

    with nidaqmx.Task() as ai_task, nidaqmx.Task() as ao_task:

        def callback(task_handle, event_type, n_samples, callback_data):
            nonlocal total_read, acquired_data, timestamps
            read = ai_task.read(number_of_samples_per_channel=n_samples)
            current_time = time.time() - start_time
            # Распределяем временные метки равномерно для прочитанных отсчетов
            timestamps.extend(np.linspace(current_time - len(read)/sampling_rate,
                                          current_time,
                                          len(read),
                                          endpoint=False))
            acquired_data.extend(read)
            total_read += len(read)
            print(f"Acquired: {len(read)} samples. Total: {total_read}", end="\r")
            return 0

        # Настройка AI
        ai_task.ai_channels.add_ai_voltage_chan("Dev1/ai0")
        ai_task.timing.cfg_samp_clk_timing(sampling_rate, sample_mode=AcquisitionType.CONTINUOUS)
        ai_task.register_every_n_samples_acquired_into_buffer_event(1000, callback)
        terminal_name = get_terminal_name_with_dev_prefix(ai_task, "ai/StartTrigger")

        # Настройка AO
        ao_task.ao_channels.add_ao_voltage_chan("Dev1/ao0")
        ao_task.timing.cfg_samp_clk_timing(sampling_rate, sample_mode=AcquisitionType.CONTINUOUS)
        ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(terminal_name)

        actual_sampling_rate = ao_task.timing.samp_clk_rate
        print(f"Actual sampling rate: {actual_sampling_rate:g} S/s")

        # Записываем данные для AO
        ao_task.write(ao_data)
        ao_task.start()
        ai_task.start()

        # Ждем, пока не будут приобретены нужное число сэмплов
        while total_read < number_of_samples:
            time.sleep(0.1)

        ai_task.stop()
        ao_task.stop()
        print(f"\nAcquired {total_read} total samples.")

    # Корректируем временные метки: время будет начинаться с 0
    timestamps = np.array(timestamps)
    timestamps = timestamps - timestamps[0]

    # Формируем датасет: три столбца: время, AI, AO
    # Если длина acquired_data и ao_data совпадает, всё хорошо
    dataset = np.column_stack((timestamps[:number_of_samples],
                                np.array(acquired_data[:number_of_samples]),
                                ao_data))
    csv_filename = "dataset.csv"
    header = "Time,AI,AO"
    np.savetxt(csv_filename, dataset, delimiter=",", header=header, comments="")
    print(f"Dataset saved as {csv_filename}")

    # Построение графиков (опционально)
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(ao_data)), ao_data, label='Output Triangular Wave')
    plt.xlabel('Номер отсчёта')
    plt.ylabel('Напряжение (V)')
    plt.title('График треугольной волны, поданной на выход')
    plt.grid(True)
    plt.legend()
    plt.savefig("output_triangle_wave.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps[:number_of_samples], acquired_data[:number_of_samples], label='Acquired AI Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Напряжение (V)')
    plt.title('Acquired Signal vs Time')
    plt.grid(True)
    plt.legend()
    plt.savefig("acquired_signal_plot.png")
    plt.show()

    return csv_filename

if __name__ == "__main__":
    main()
