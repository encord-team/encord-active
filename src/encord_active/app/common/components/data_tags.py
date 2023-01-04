from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import streamlit as st


@dataclass
class Property:
    key: str
    icon: str
    icon_color: str
    name: str
    is_code: bool = False
    is_url: bool = False

    def get_key(self, metric_name):
        return self.key

    def get_name(self, metric_name):
        return self.name

    def get_icon_color(self, value):
        return self.icon_color


class MetricProperty(Property):
    def get_key(self, metric_name):
        return metric_name

    def get_name(self, metric_name):
        return metric_name


class OutlierProperty(Property):
    COLORS = {
        "Severe": "red",
        "Moderate": "orange",
        "Low": "green",
    }

    def get_icon_color(self, value):
        return self.COLORS.get(value, self.icon_color)


properties: List[Property] = [
    MetricProperty("metric", "tag", "purple", ""),
    Property("class_name", "c", "green", "Predicted class", is_code=True),
    Property("label_class_name", "c", "purple", "Class"),
    Property("iou", "i", "yellow", "IOU"),
    Property("confidence", "microchip", "orange", "Model confidence"),
    Property("url", "pencil", "blue", "Link to Encord editor", is_url=True),
    Property("annotator", "user", "purple", "Annotator"),
    OutlierProperty("outlier", "circle", "blue", "Outlier severity"),
]

TAG_TEMPLATE = """    <div class="tags-item">
        <i class="fa-solid fa-%s text-%s"></i>
        %s
        <span class="tooltiptext">%s</span>
    </div>"""


def build_data_tags(row: pd.Series, metric_name: Optional[str] = None):
    tag_list = []
    for p in properties:
        key = p.get_key(metric_name)
        if not key in row:
            continue
        _value = row[key]
        value = ""
        if isinstance(_value, str):
            value = _value
        elif isinstance(_value, float):
            value = f"{_value:.3f}"
        elif isinstance(_value, int):
            value = f"{_value:d}"

        if len(value) == 0 or value == "nan":
            continue

        if p.is_code:
            value = f"<code>{value}</code>"

        if p.is_url:
            value = f'<a href={value} target="_blank">Editor</a>'

        if value:
            tag_list.append(TAG_TEMPLATE % (p.icon, p.get_icon_color(value), value, p.get_name(metric_name)))

    tags = "\n".join(tag_list)

    html = f"""
<div class="tags-container">
    {tags}
</div>
"""
    st.markdown(html, unsafe_allow_html=True)
