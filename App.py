from __future__ import annotations

from typing import Optional
from pathlib import Path
from Data.Data import Data
from Model.Model import Model
from PlotDashboard.Dashboard import Dashboard


class App:
    """
    """

    def __init__(
        self,
        data: Optional[Data] = None,
        model: Optional[Model] = None,
        dashboard: Optional[Dashboard] = None,
    ) -> None:
        self.data: Data = data or Data()
        self.model: Model = model or Model()
        self.dashboard: Dashboard = dashboard or Dashboard()

    def run(self) -> None:
        pass
