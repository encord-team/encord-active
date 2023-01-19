import streamlit as st
from typing import Union

def summary_item(section: st._DeltaGenerator, title: str, value: Union[str, int],
                 background_color: str, title_html_tag:str='p', value_html_tag:str='h2', value_margin_bottom:str='0'):
        section.markdown(
f'''
<div style="background-color:{background_color}; border-radius:5px; padding: 20px;">
<{title_html_tag}>{title} </{title_html_tag}> 
<{value_html_tag} style="margin-bottom:{value_margin_bottom}%">{value}</{value_html_tag}>
</div>
''',
            unsafe_allow_html=True)

