import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType
import matplotlib.pyplot as plt
import time

def generate_triangle_wave(num_samples=600000, amplitude=2.0, offset=0.0):
    """
    Генерирует один период треугольной волны.

    :param num_samples: Количество точек (сэмплов) в периоде волны.
                        При частоте 10 кГц и периоде 60 сек, их будет 600000.
    :param amplitude: Максимальное отклонение волны. Волна будет меняться от -amplitude до +amplitude.
    :param offset: Смещение волны по вертикали.
    :return: numpy-массив с одним периодом треугольной волны.
    """
    half_period = num_samples // 2

    # Генерируем восходящую часть: от -amplitude до +amplitude
    up_part = np.linspace(-amplitude, amplitude, half_period, endpoint=False)
    # Генерируем нисходящую часть: от +amplitude до -amplitude
    down_part = np.linspace(amplitude, -amplitude, half_period, endpoint=False)

    # Объединяем две части в один период
    wave = np.concatenate((up_part, down_part))
    wave += offset

    return wave

def main():
    # Параметры сигнала и задач
    sample_rate = 10000      # Частота дискретизации: 10 кГц
    period_sec = 60          # Период треугольной волны: 60 секунд (т.е. 1/60 Гц)
    wave_samples = int(sample_rate * period_sec)  # Количество сэмплов за один период: 10_000 * 60 = 600000
    amplitude = 2.0          # Треугольная волна меняется от -2 до +2 В

    # Генерируем один период треугольной волны
    tri_wave = generate_triangle_wave(num_samples=wave_samples, amplitude=amplitude, offset=0.0)

    # Создаем две задачи: одну для AO (вывод) и одну для AI (ввод)
    with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task:
        # ------------------- Настройка AO -------------------
        # Добавляем аналоговый выход (например, Dev1/ao0)
        # Диапазон -5..+5 В выбран с запасом для нашего ±2 В сигнала.
        ao_task.ao_channels.add_ao_voltage_chan(
            "Dev1/ao0",
            min_val=-5.0,
            max_val=5.0
        )
        # Настраиваем тактовый сигнал для непрерывной генерации:
        # - rate=sample_rate: 10 кГц
        # - sample_mode=CONTINUOUS: данные повторяются циклически
        # - samps_per_chan=wave_samples: размер буфера равен количеству сэмплов в одном периоде (600000)
        ao_task.timing.cfg_samp_clk_timing(
            rate=sample_rate,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=wave_samples
        )
        # Загружаем массив треугольной волны в буфер AO, не стартуя сразу задачу
        ao_task.write(tri_wave, auto_start=False)

        # ------------------- Настройка AI -------------------
        # Добавляем аналоговый вход (например, Dev1/ai0)
        # Диапазон -10..+10 В позволяет безопасно захватить сигнал.
        ai_task.ai_channels.add_ai_voltage_chan(
            "Dev1/ai0",
            min_val=-10.0,
            max_val=10.0
        )
        # Настраиваем тактовый сигнал для непрерывного считывания:
        # - rate=sample_rate: 10 кГц
        # - sample_mode=CONTINUOUS
        # - samps_per_chan=wave_samples: размер буфера соответствует одному периоду (600000 отсчётов)
        ai_task.timing.cfg_samp_clk_timing(
            rate=sample_rate,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=wave_samples
        )

        # Запускаем задачу AI сначала, чтобы она успела подготовиться к считыванию, затем AO
        ai_task.start()
        ao_task.start()

        print("Генерация треугольной волны (1/60 Гц) на AO0 и считывание с AI0 начаты!")
        print("Убедитесь, что AO0 физически соединён с AI0 (например, проводом).")
        print("Считывание одного полного периода займет примерно 60 секунд...")

        # Считываем один период данных (600000 отсчётов, около 60 секунд)
        data = ai_task.read(number_of_samples_per_channel=wave_samples, timeout=70.0)

        print("Считан один период данных с AI0.")

    # После выхода из блока with задачи автоматически останавливаются и освобождаются

    # ------------------- Построение графика -------------------
    # Построим полученные данные. Поскольку период длится 60 секунд, график будет очень длинным.
    plt.figure(figsize=(12, 6))
    plt.plot(data)
    plt.title("Сигнал, считанный с AI0 (треугольная волна, 1/60 Гц, 10 кГц)")
    plt.xlabel("Номер отсчёта (10 кГц)")
    plt.ylabel("Напряжение (В)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
