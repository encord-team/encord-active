from hashlib import md5
from pathlib import Path

import streamlit as st
import streamlit.elements.image as st_image
from PIL import Image

from encord_active.lib.common.colors import Color


def write_page_css():
    # Prepare url for encord logo
    logo_pth = Path(__file__).parents[1] / "assets" / "encord_2_02.png"
    logo = Image.open(logo_pth)
    img_url = st_image.image_to_url(logo, 500, True, "RGB", "PNG", f"logo-{md5(logo.tobytes()).hexdigest()}")

    # Write css
    st.markdown(
        f"""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.0/css/all.min.css">
<style>
    [data-testid="stSidebar"] {{
        background-image: url({img_url});
        background-repeat: no-repeat;
        background-position: 20px 20px;
        background-size: 60%;
    }}

    .tags-container {{
        display: flex;
        flex-flow: row wrap;
        justify-content: space-around;
        align-items: center;
    }}
    div .tags-item {{
        margin: 3px;
    }}
    .text-purple {{
        color: {Color.PURPLE.value}
    }}
    .text-red {{
        color: {Color.RED.value}
    }}
    .text-yellow {{
        color: {Color.YELLOW.value}
    }}
    .text-blue {{
        color: {Color.BLUE.value}
    }}
    .text-orange {{
        color: {Color.ORANGE.value}
    }}
    .text-green {{
        color: {Color.GREEN.value}
    }}

    .tags-item .tooltiptext {{
      visibility: hidden;
      width: 120px;
      background-color: #f0f2f6ee;
      color: #31333f;
      text-align: center;
      border-radius: 4px;
      padding: 5px;
      border: solid 1px #d7d7d7cc;
      box-shadow: 3px 3px 6px #cccccc55;

      /* Position the tooltip */
      position: absolute;
      z-index: 1;
      bottom: 100%;
      left: 50%;
      margin-left: -60px;
    }}

    .tags-item:hover .tooltiptext {{
      visibility: visible;
    }}

    .encord-active-info-box {{
        width: 100%;
        padding: 1em;
        background-color: rgba(28, 131, 225, 0.1);
        color: rgb(0, 66, 128);
        border-radius: 0.25rem;
        margin-top: 1em;
        margin-bottom: 1em;
    }}

    span.data-tag {{
        color: #fff;
        background-color: {Color.PURPLE.value};
        border-radius: 10rem;
        display: inline-block;
        padding: 0.25em 0.4em;
        font-size: 75%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
    }}

    span.data-tag-count {{
        margin-left: 0.2rem;
        background: rgba(255, 255, 255, 0.2);
    }}

    /* Place expand button within image */
    .element-container > div > button[title='View fullscreen'] {{
        right: 0.2rem;
        top: 0.2rem;
    }}

</style>
        """,
        unsafe_allow_html=True,
    )
