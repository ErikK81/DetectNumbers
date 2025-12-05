import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout
)
from PySide6.QtGui import QPainter, QPixmap, QImage, QPen
from PySide6.QtCore import Qt
from PIL import Image
from neural_network import MLP
from PySide6.QtCore import QTimer

class DrawArea(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(280, 280)
        self.pixmap = QPixmap(self.size())
        self.pixmap.fill(Qt.white)
        self.last_point = None

    def mousePressEvent(self, event):
        self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            painter = QPainter(self.pixmap)
            pen = QPen(Qt.black, 18, Qt.SolidLine, Qt.RoundCap)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            painter.end()
            self.last_point = event.pos()
            self.update()

    def paintEvent(self, event):
        canvas = QPainter(self)
        canvas.drawPixmap(0, 0, self.pixmap)

    def clear(self):
        self.pixmap.fill(Qt.white)
        self.update()

    def to_mnist_image(self):
        img = self.pixmap.toImage().convertToFormat(QImage.Format_Grayscale8)

        ptr = img.bits()        # agora vira memoryview
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(280, 280)

        pil_img = Image.fromarray(arr).resize((28, 28))
        return pil_img



class DigitApp(QWidget):
    def __init__(self, model):
        super().__init__()

        self.model = model
        self.setWindowTitle("Reconhecedor de Dígitos - PySide6")
        self.setStyleSheet("background-color: #1e1e1e; color: white;")

        self.canvas = DrawArea()

        btn_clear = QPushButton("Limpar")
        btn_clear.clicked.connect(self.canvas.clear)

        self.output = QLabel("Desenhe um número")
        self.output.setStyleSheet("font-size: 28px;")
        self.output.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas, alignment=Qt.AlignCenter)
        layout.addWidget(btn_clear)
        layout.addWidget(self.output)
        self.setLayout(layout)

        # timer salvo no objeto
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.predict_digit)
        self.timer.start(150)

    def predict_digit(self):
        img = self.canvas.to_mnist_image()

        arr = 255 - np.array(img)
        arr = arr / 255.0
        arr = arr.reshape(1, 784)

        pred = self.model.predict(arr)[0]
        self.output.setText(f"Detectado: {pred}")
