import typing

from magicgui import register_type
from magicgui.widgets import Container, FloatSpinBox, Label
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QAbstractSpinBox, QHBoxLayout, QLabel, QWidget

from PartSeg.common_gui.universal_gui_part import CustomDoubleSpinBox


class TrapezoidRange:
    def __init__(self, lower_bound: float, upper_bound: float):
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    @property
    def lower_bound(self) -> float:
        return self._lower_bound

    @property
    def upper_bound(self) -> float:
        return self._upper_bound

    @classmethod
    def __get_validators__(cls):
        yield cls._validate

    def __str__(self):
        return f"{self._lower_bound} - {self._upper_bound}"

    def __repr__(self):
        return f"TrapezoidRange(lower_bound={self._lower_bound}, upper_bound={self._upper_bound})"

    @classmethod
    def _validate(cls, arg):
        if isinstance(arg, TrapezoidRange):
            return arg
        if not isinstance(arg["lower_bound"], (int, float)) and not isinstance(arg["upper_bound"], (int, float)):
            raise ValueError("values need to be float")
        return TrapezoidRange(**arg)

    def as_dict(self):
        return {"lower_bound": self._lower_bound, "upper_bound": self._upper_bound}


class NewTrapezoidWidget(Container):
    def __init__(self, **kwargs):
        self.lo = FloatSpinBox(min=0, max=1000000)
        self.lo.tooltip = "lower bound"
        self.lo.native.setObjectName("short_spin_box")
        self.lo.native.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.up = FloatSpinBox(min=0, max=1000000)
        self.up.tooltip = "upper bound"
        self.up.native.setObjectName("short_spin_box")
        self.up.native.setButtonSymbols(QAbstractSpinBox.NoButtons)
        super().__init__(widgets=[self.lo, Label(value="to"), self.up], layout="horizontal", labels=False)
        self.margins = (0, 0, 0, 0)

    @property
    def value(self) -> TrapezoidRange:
        return TrapezoidRange(lower_bound=self.lo.value, upper_bound=self.up.value)

    @value.setter
    def value(self, value: typing.Union[TrapezoidRange, typing.Tuple[float, float]]):
        if isinstance(value, TrapezoidRange):
            self.lo.value = value.lower_bound
            self.up.value = value.upper_bound
        else:
            self.lo.value, self.up.value = value


register_type(TrapezoidRange, widget_type=NewTrapezoidWidget)


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
        self.lower_bound.setRange(0, 9999999)
        self.lower_bound.setToolTip("lower bound")
        self.lower_bound.valueChanged.connect(self._update_value)
        self.lower_bound.setDecimals(1)
        self.upper_bound = ShortCustomDoubleSpinBox()
        self.upper_bound.setRange(0, 9999999)
        self.upper_bound.valueChanged.connect(self._update_value)
        self.upper_bound.setToolTip("upper bound")
        self.upper_bound.setDecimals(1)
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
QAbstractSpinBox#short_spin_box {
  background-color: {{ foreground }};
  border: none;
  padding: 1px 1px;
  min-width: 50px;
  min-height: 18px;
  border-radius: 2px;
}
"""
