import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType, Edge
import matplotlib.pyplot as plt
import time

def generate_triangle_wave(num_samples=600000, amplitude=2.0, offset=0.0):
    """
    Генерирует один период треугольной волны.
    
    :param num_samples: Количество точек (сэмплов) в периоде (600000 для 60 сек при 10 кГц)
    :param amplitude: Амплитуда волны (волна от -amplitude до +amplitude)
    :param offset: Смещение по вертикали
    :return: numpy-массив с одним периодом треугольной волны
    """
    half_period = num_samples // 2
    # Восходящая часть: от -amplitude до +amplitude
    up_part = np.linspace(-amplitude, amplitude, half_period, endpoint=False)
    # Нисходящая часть: от +amplitude до -amplitude
    down_part = np.linspace(amplitude, -amplitude, half_period, endpoint=False)
    wave = np.concatenate((up_part, down_part))
    wave += offset
    return wave

def main():
    # Параметры сигнала
    sample_rate = 10000      # 10 кГц
    period_sec = 60          # Период = 60 секунд (1/60 Гц)
    wave_samples = int(sample_rate * period_sec)  # 600000 отсчётов
    amplitude = 2.0          # Треугольная волна от -2 до +2 В
    
    # Генерируем один период треугольной волны
    tri_wave = generate_triangle_wave(num_samples=wave_samples, amplitude=amplitude, offset=0.0)
    
    # Создаем две задачи: одна для АО (вывод) и одна для АИ (ввод)
    with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task:
        # ------------- Настройка задачи АО (вывод) -------------
        ao_task.ao_channels.add_ao_voltage_chan(
            "Dev1/ao0",   # Имя канала, проверьте в NI MAX
            min_val=-5.0,
            max_val=5.0
        )
        ao_task.timing.cfg_samp_clk_timing(
            rate=sample_rate,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=wave_samples
        )
        # Загружаем треугольную волну в буфер, но не запускаем сразу
        ao_task.write(tri_wave, auto_start=False)
        
        # ------------- Настройка задачи АИ (ввод) -------------
        ai_task.ai_channels.add_ai_voltage_chan(
            "Dev1/ai0",   # Имя канала для входа
            min_val=-10.0,
            max_val=10.0
        )
        ai_task.timing.cfg_samp_clk_timing(
            rate=sample_rate,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=wave_samples
        )
        # Настраиваем задачу АИ так, чтобы она ожидала стартовый триггер, поступающий с АО.
        # Это гарантирует, что АИ начнет сбор данных одновременно с запуском АО.
        ai_task.triggers.start_trigger.cfg_dig_edge_start_trig("/Dev1/ao/SampleClock", trigger_edge=Edge.RISING)
        
        # ------------- Синхронный старт задач -------------
        # Для синхронизации сначала запускаем задачу-слейв (АИ), которая ждет триггер
        ai_task.start()
        # Затем запускаем задачу-мастер (АО), которая генерирует стартовый сигнал через свой тактовый выход.
        ao_task.start()
        
        print("Старт синхронизирован: задачи АО и АИ запущены одновременно!")
        print("Считывание одного полного периода (60 секунд) данных...")
        
        # Считываем один период данных (600000 отсчётов)
        data = ai_task.read(number_of_samples_per_channel=wave_samples, timeout=70.0)
        print("Считывание завершено.")
    
    # Построение графика считанного сигнала
    plt.figure(figsize=(12, 6))
    plt.plot(data)
    plt.title("Синхронное считывание: треугольная волна (1/60 Гц, 10 кГц)")
    plt.xlabel("Номер отсчёта")
    plt.ylabel("Напряжение (В)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
