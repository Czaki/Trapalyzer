from qtpy.QtCore import Signal
from qtpy.QtWidgets import QAbstractSpinBox, QHBoxLayout, QLabel, QWidget

from PartSeg.common_gui.universal_gui_part import CustomDoubleSpinBox


class TrapezoidWidget:
    @staticmethod
    def get_object():
        return TrapezoidParametersWidget()


class ShortCustomDoubleSpinBox(CustomDoubleSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)


class TrapezoidParametersWidget(QWidget):
    values_changed = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.lower_bound = ShortCustomDoubleSpinBox()
        self.lower_bound.setRange(0, 999999)
        self.lower_bound.setToolTip("lower bound")
        self.lower_bound.valueChanged.connect(self._update_value)
        self.upper_bound = ShortCustomDoubleSpinBox()
        self.upper_bound.setRange(0, 999999)
        self.upper_bound.valueChanged.connect(self._update_value)
        self.upper_bound.setToolTip("upper bound")
        # self.setFixedWidth(250)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        # layout.addWidget(QLabel("l:"))
        layout.addWidget(self.lower_bound)
        layout.addWidget(QLabel("to"))
        layout.addWidget(self.upper_bound)
        self.setLayout(layout)

    def _update_value(self):
        # print("aaa", QApplication.instance().styleSheet())
        self.values_changed.emit(self.lower_bound.value(), self.upper_bound.value())

    def get_value(self):
        return {
            "lower_bound": self.lower_bound.value(),
            "upper_bound": self.upper_bound.value(),
        }

    def set_value(self, value):
        self.lower_bound.setValue(value["lower_bound"])
        self.upper_bound.setValue(value["upper_bound"])


qss_file = """
ShortCustomDoubleSpinBox {
  background-color: {{ foreground }};
  border: none;
  padding: 1px 1px;
  min-width: 50px;
  min-height: 18px;
  border-radius: 2px;
}
"""
