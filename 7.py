import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType, Edge
import matplotlib.pyplot as plt
import time

def generate_triangle_wave(num_samples, amplitude, offset=0.0):
    """
    Генерирует один период треугольной волны.
    
    :param num_samples: Общее количество сэмплов в периоде.
                        Например, при 10 кГц и периоде 60 сек – 600000 сэмплов.
    :param amplitude: Амплитуда волны; сигнал изменяется от -amplitude до +amplitude.
    :param offset: Вертикальное смещение (если требуется).
    :return: numpy-массив с одним периодом треугольной волны.
    """
    half_period = num_samples // 2
    # Восходящая часть: от -amplitude до +amplitude
    up_part = np.linspace(-amplitude, amplitude, half_period, endpoint=False)
    # Нисходящая часть: от +amplitude до -amplitude
    down_part = np.linspace(amplitude, -amplitude, half_period, endpoint=False)
    wave = np.concatenate((up_part, down_part)) + offset
    return wave

def main():
    # Параметры сигнала
    sample_rate = 10000      # Частота дискретизации: 10 кГц
    period_sec = 60          # Период треугольной волны: 60 секунд (1/60 Гц)
    wave_samples = int(sample_rate * period_sec)  # 10,000 * 60 = 600,000 сэмплов
    amplitude = 7.0          # Треугольная волна от -7 до +7 В

    # Генерируем один период треугольной волны
    tri_wave = generate_triangle_wave(num_samples=wave_samples, amplitude=amplitude, offset=0.0)

    # Создаем две отдельные задачи: AO и AI
    with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task:
        # ------------- Настройка AO (аналоговый вывод) -------------
        ao_task.ao_channels.add_ao_voltage_chan(
            "Dev1/ao0",   # Проверьте имя канала в NI MAX
            min_val=-5.0,
            max_val=5.0
        )
        ao_task.timing.cfg_samp_clk_timing(
            rate=sample_rate,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=wave_samples
        )
        # Записываем треугольную волну в буфер, но не стартуем сразу
        ao_task.write(tri_wave, auto_start=False)

        # ------------- Настройка AI (аналоговый ввод) -------------
        ai_task.ai_channels.add_ai_voltage_chan(
            "Dev1/ai0",   # Проверьте имя канала в NI MAX
            min_val=-10.0,
            max_val=10.0
        )
        ai_task.timing.cfg_samp_clk_timing(
            rate=sample_rate,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=wave_samples
        )
        # Настраиваем AI, чтобы он ожидал стартовый триггер, приходящий с AO
        ai_task.triggers.start_trigger.cfg_dig_edge_start_trig(
            "/Dev1/ao/SampleClock", trigger_edge=Edge.RISING
        )

        # ------------- Синхронный старт -------------
        # Сначала запускаем AI (слейв), который ожидает триггер,
        # затем запускаем AO (мастер), чье начало генерирует сигнал триггера.
        ai_task.start()
        ao_task.start()

        print("Синхронный старт: AO и AI запущены одновременно!")
        print("Ожидание одного полного периода (60 секунд) данных...")

        # Считываем один период данных с AI (600,000 сэмплов).
        # Увеличенный timeout позволяет дождаться сбора всех отсчетов.
        data = ai_task.read(number_of_samples_per_channel=wave_samples, timeout=70.0)
        print("Считывание завершено.")

    # Построение графика считанных данных
    plt.figure(figsize=(12, 6))
    plt.plot(data)
    plt.title("Синхронное считывание: треугольная волна (1/60 Гц, 10 кГц)")
    plt.xlabel("Номер сэмпла")
    plt.ylabel("Напряжение (В)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
