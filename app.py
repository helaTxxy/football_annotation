from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from src.main_window import MainWindow


def main() -> int:
    root = Path(__file__).resolve().parent
    app = QApplication(sys.argv)
    win = MainWindow(root=root)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
