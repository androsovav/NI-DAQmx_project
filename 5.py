import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType
import matplotlib.pyplot as plt

# Параметры сигнала
sample_rate = 10000      # 10 кГц
wave_samples = 600000    # 600000 отсчётов (60 секунд при 10 кГц)
amplitude = 2.0          # ±2 В

def generate_triangle_wave(num_samples, amplitude, offset=0.0):
    half_period = num_samples // 2
    up_part = np.linspace(-amplitude, amplitude, half_period, endpoint=False)
    down_part = np.linspace(amplitude, -amplitude, half_period, endpoint=False)
    wave = np.concatenate((up_part, down_part)) + offset
    return wave

tri_wave = generate_triangle_wave(wave_samples, amplitude)

# Создаем один task, объединяющий AO и AI
with nidaqmx.Task() as task:
    # Добавляем канал аналогового вывода
    task.ao_channels.add_ao_voltage_chan("Dev1/ao0", min_val=-5.0, max_val=5.0)
    # Добавляем канал аналогового ввода
    task.ai_channels.add_ai_voltage_chan("Dev1/ai0", min_val=-10.0, max_val=10.0)
    
    # Настраиваем единый sample clock для всех каналов
    task.timing.cfg_samp_clk_timing(
        rate=sample_rate,
        sample_mode=AcquisitionType.CONTINUOUS,
        samps_per_chan=wave_samples
    )
    
    # Загружаем данные для AO (не запускаем автоматически)
    task.write(tri_wave, auto_start=False)
    
    # Запускаем задачу – одновременно начнут генерироваться данные на AO и собираться данные на AI
    task.start()
    
    print("Задача запущена. AO и AI работают одновременно.")

    # Считываем один период данных с AI
    data = task.read(number_of_samples_per_channel=wave_samples, timeout=70.0)
    print("Считывание завершено.")

# Построим график считанных данных
plt.figure(figsize=(12, 6))
plt.plot(data)
plt.title("Синхронное считывание: треугольная волна (AO) и данные AI")
plt.xlabel("Номер отсчёта")
plt.ylabel("Напряжение (В)")
plt.grid(True)
plt.show()
