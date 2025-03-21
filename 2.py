import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType
import time

def generate_triangle_wave(num_samples=1000, amplitude=2.0, offset=0.0):
    """
    Генерирует один период треугольной волны.
    """
    half_period = num_samples // 2
    up_part = np.linspace(-amplitude, amplitude, half_period, endpoint=False)
    down_part = np.linspace(amplitude, -amplitude, half_period, endpoint=False)
    wave = np.concatenate((up_part, down_part))
    wave += offset
    return wave

if __name__ == "__main__":
    tri_wave = generate_triangle_wave(num_samples=1000, amplitude=2.0, offset=0.0)
    sample_rate = 1000.0  # 1 кГц => один период 1 сек

    with nidaqmx.Task() as ao_task:
        # Добавляем канал на устройство Dev1, выход ao0
        ao_task.ao_channels.add_ao_voltage_chan(
            "Dev1/ao0",
            min_val=-5.0,
            max_val=5.0
        )

        # Настраиваем непрерывный режим генерации
        ao_task.timing.cfg_samp_clk_timing(
            rate=sample_rate,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=len(tri_wave)
        )

        # Подаём данные в буфер, но auto_start=False (запустим задачу вручную)
        ao_task.write(tri_wave, auto_start=False)

        # Запускаем задачу (генерацию)
        ao_task.start()

        print("Генерация треугольной волны началась!")
        time.sleep(10)
        print("Останавливаем генерацию")

    # Выходим из блока with — задача автоматически завершается и освобождается
