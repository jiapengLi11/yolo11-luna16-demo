from __future__ import annotations

import sys
from pathlib import Path

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
UI_DIR = ROOT / "assets" / "ui"
TMP_DIR = ROOT / "runs" / "gui"
DEFAULT_IMAGE = ROOT / "assets" / "samples" / "0004.png"
DEFAULT_ICON = UI_DIR / "lufei.png"
DEFAULT_PLACEHOLDER = UI_DIR / "up.jpeg"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.model: YOLO | None = None
        self.model_path = ""
        self.image_path = str(DEFAULT_IMAGE)
        self.setWindowTitle("LUNA16 YOLO Demo")
        self.setWindowIcon(QIcon(str(DEFAULT_ICON)))
        self.resize(1200, 760)
        self._build_ui()
        self._show_image(self.input_label, self.image_path)
        self._set_status("Select a checkpoint and run inference.")

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)

        header = QLabel("LUNA16 chest nodule detection demo")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("font-size: 24px; font-weight: 600;")
        layout.addWidget(header)

        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        controls = QHBoxLayout()
        self.model_line = QLineEdit()
        self.model_line.setPlaceholderText("Choose a .pt checkpoint")
        self.image_line = QLineEdit(self.image_path)
        controls.addWidget(self.model_line)

        model_button = QPushButton("Model")
        model_button.clicked.connect(self.choose_model)
        controls.addWidget(model_button)

        controls.addWidget(self.image_line)
        image_button = QPushButton("Image")
        image_button.clicked.connect(self.choose_image)
        controls.addWidget(image_button)

        run_button = QPushButton("Run Inference")
        run_button.clicked.connect(self.run_inference)
        controls.addWidget(run_button)
        layout.addLayout(controls)

        image_row = QHBoxLayout()
        self.input_label = QLabel()
        self.output_label = QLabel()
        for label in (self.input_label, self.output_label):
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(520, 520)
            label.setStyleSheet("border: 1px solid #d0d0d0; background: #fafafa;")
        image_row.addWidget(self.input_label)
        image_row.addWidget(self.output_label)
        layout.addLayout(image_row)

        self.setCentralWidget(root)

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _show_image(self, label: QLabel, image_path: str | Path) -> None:
        path = Path(image_path)
        if not path.exists():
            path = DEFAULT_PLACEHOLDER
        pixmap = QPixmap(str(path))
        label.setPixmap(pixmap.scaled(520, 520, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def choose_model(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose checkpoint", "", "Model (*.pt)")
        if not file_path:
            return
        self.model_path = file_path
        self.model_line.setText(file_path)
        try:
            self.model = YOLO(file_path)
        except Exception as exc:
            self.model = None
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        self._set_status(f"Loaded model: {Path(file_path).name}")

    def choose_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose image", "", "Image (*.png *.jpg *.jpeg *.bmp)")
        if not file_path:
            return
        self.image_path = file_path
        self.image_line.setText(file_path)
        self._show_image(self.input_label, file_path)
        self._set_status(f"Loaded image: {Path(file_path).name}")

    def run_inference(self) -> None:
        if self.model is None:
            QMessageBox.warning(self, "Missing model", "Please choose a .pt checkpoint first.")
            return
        source = self.image_line.text().strip()
        if not source:
            QMessageBox.warning(self, "Missing image", "Please choose an input image first.")
            return
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        try:
            result = self.model(source, verbose=False)[0]
        except Exception as exc:
            QMessageBox.critical(self, "Inference failed", str(exc))
            return

        rendered = result.plot()
        output_path = TMP_DIR / "prediction.jpg"
        cv2.imwrite(str(output_path), rendered)
        self._show_image(self.input_label, source)
        self._show_image(self.output_label, output_path)
        boxes = len(result.boxes) if result.boxes is not None else 0
        self._set_status(f"Inference finished. Detections: {boxes}")


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
