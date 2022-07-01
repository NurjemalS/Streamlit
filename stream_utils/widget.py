from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union, Type

from streamlit.delta_generator import DeltaGenerator


@dataclass
class InputWidgetOption:
    
    label: str
    type: str 
    value: Optional[Union[List[Any], Any]] = None
    min_value: float = None
    max_value: float = None
    step: float = None

    def render():
        if self.type == "selectbox":
            return st_container.selectbox(
                label=self, 
                options = self.value,
                index = 0
            )
            
        elif self.type == "slider":
            return st_container.slider(
                label=self.label,
                min_value=self.min_value,
                max_value=self.max_value,
                value=self.value,
                step=self.step,
                key=self.label
            )



