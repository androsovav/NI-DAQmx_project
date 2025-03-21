import numpy as np
import matplotlib.pyplot as plt

def generate_triangle_wave(num_samples=1000, amplitude=2.0, offset=0.0):
    """
    Генерирует один период треугольной волны.
    
    :param num_samples: количество точек (сэмплов) в периоде
    :param amplitude: пиковое значение волны (от -amplitude до +amplitude)
    :param offset: смещение по вертикали (если нужно)
    :return: numpy-массив с одним периодом треугольной волны
    """
    # Половина периода: от -amplitude до +amplitude
    half_period = num_samples // 2

    # Линейный рост: [-amplitude .. +amplitude)
    up_part = np.linspace(-amplitude, amplitude, half_period, endpoint=False)
    # Линейное падение: [+amplitude .. -amplitude]
    down_part = np.linspace(amplitude, -amplitude, half_period, endpoint=False)

    # Склеиваем рост и падение
    wave = np.concatenate((up_part, down_part))
    
    # Добавляем смещение, если нужно
    wave += offset
    
    return wave

# ------------ Основной код ------------
if __name__ == "__main__":
    # Генерируем треугольную волну на 1000 точек в периоде, от -2В до +2В
    tri_wave = generate_triangle_wave(num_samples=1000, amplitude=2.0, offset=0.0)
    
    # Строим график полученной волны
    plt.plot(tri_wave)
    plt.title("Один период треугольной волны (±2 В, 1000 сэмплов)")
    plt.xlabel("Номер отсчёта")
    plt.ylabel("Напряжение (В)")
    plt.show()
