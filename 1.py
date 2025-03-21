import nidaqmx
from nidaqmx.constants import AcquisitionType
import matplotlib.pyplot as plt

# Создаём задачу для чтения одного аналогового канала
with nidaqmx.Task() as task:
    # Добавляем канал аналогового ввода (например, Dev1/ai0)
    task.ai_channels.add_ai_voltage_chan("Dev1/ai0",
                                         min_val=-10.0,
                                         max_val=10.0)
    
    # Настраиваем тактовый сигнал на 1 кГц, непрерывный режим, буфер 1000 отсчётов
    task.timing.cfg_samp_clk_timing(
        rate=1000,
        sample_mode=AcquisitionType.CONTINUOUS
    )
    
    # Считываем 1000 образцов
    data = task.read(number_of_samples_per_channel=10000)

# Теперь строим график полученных данных
plt.plot(data)  # Подаём список 1000 отсчётов на построение
plt.title("AI Data from NI-DAQmx")
plt.xlabel("Sample index")
plt.ylabel("Voltage (V)")
plt.show()
