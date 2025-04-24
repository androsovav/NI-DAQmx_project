import sys
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QMessageBox
)
import ai_ao_sync


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI-AO Sync")

        # Создаем форму для ввода параметров
        layout = QFormLayout()

        # Поля ввода с значениями по умолчанию
        self.freq_edit = QLineEdit(str(1))
        self.amp_edit = QLineEdit(str(7))
        self.sr_edit = QLineEdit(str(10000))
        self.dur_edit = QLineEdit(str(5))

        layout.addRow("Frequency (Hz):", self.freq_edit)
        layout.addRow("Amplitude:", self.amp_edit)
        layout.addRow("Sampling Rate (Hz):", self.sr_edit)
        layout.addRow("Duration (s):", self.dur_edit)

        # Кнопка запуска
        self.run_button = QPushButton("Запустить")
        self.run_button.clicked.connect(self.run)
        layout.addRow(self.run_button)

        self.setLayout(layout)

    def run(self):
        try:
            # Читаем и конвертируем параметры
            frequency = float(self.freq_edit.text())
            amplitude = float(self.amp_edit.text())
            sampling_rate = int(self.sr_edit.text())
            duration = float(self.dur_edit.text())

            # Вызываем основную функцию
            csv_file = ai_ao_sync.main(
                frequency=frequency,
                amplitude=amplitude,
                sampling_rate=sampling_rate,
                duration=duration
            )

            # Сообщаем пользователю о завершении
            QMessageBox.information(
                self,
                "Готово",
                f"Результирующий CSV сохранен в:\n{csv_file}"
            )
        except Exception as e:
            # Показываем ошибку, если что-то пошло не так
            QMessageBox.critical(self, "Ошибка", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())