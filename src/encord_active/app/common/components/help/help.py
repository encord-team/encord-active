from pathlib import Path
from typing import Optional

import streamlit as st
from streamlit.components.v1 import html

from encord_active.lib.constants import DOCS_URL, JOIN_DISCORD_URL

DISCORD_ICON = """<svg viewBox="0 0 640 512" height="16" width="16" xmlns="http://www.w3.org/2000/svg"><path d="M524.531,69.836a1.5,1.5,0,0,0-.764-.7A485.065,485.065,0,0,0,404.081,32.03a1.816,1.816,0,0,0-1.923.91,337.461,337.461,0,0,0-14.9,30.6,447.848,447.848,0,0,0-134.426,0,309.541,309.541,0,0,0-15.135-30.6,1.89,1.89,0,0,0-1.924-.91A483.689,483.689,0,0,0,116.085,69.137a1.712,1.712,0,0,0-.788.676C39.068,183.651,18.186,294.69,28.43,404.354a2.016,2.016,0,0,0,.765,1.375A487.666,487.666,0,0,0,176.02,479.918a1.9,1.9,0,0,0,2.063-.676A348.2,348.2,0,0,0,208.12,430.4a1.86,1.86,0,0,0-1.019-2.588,321.173,321.173,0,0,1-45.868-21.853,1.885,1.885,0,0,1-.185-3.126c3.082-2.309,6.166-4.711,9.109-7.137a1.819,1.819,0,0,1,1.9-.256c96.229,43.917,200.41,43.917,295.5,0a1.812,1.812,0,0,1,1.924.233c2.944,2.426,6.027,4.851,9.132,7.16a1.884,1.884,0,0,1-.162,3.126,301.407,301.407,0,0,1-45.89,21.83,1.875,1.875,0,0,0-1,2.611,391.055,391.055,0,0,0,30.014,48.815,1.864,1.864,0,0,0,2.063.7A486.048,486.048,0,0,0,610.7,405.729a1.882,1.882,0,0,0,.765-1.352C623.729,277.594,590.933,167.465,524.531,69.836ZM222.491,337.58c-28.972,0-52.844-26.587-52.844-59.239S193.056,219.1,222.491,219.1c29.665,0,53.306,26.82,52.843,59.239C275.334,310.993,251.924,337.58,222.491,337.58Zm195.38,0c-28.971,0-52.843-26.587-52.843-59.239S388.437,219.1,417.871,219.1c29.667,0,53.307,26.82,52.844,59.239C470.715,310.993,447.538,337.58,417.871,337.58Z"></path></svg>"""
DOCS_ICON = """<svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M13 1H3C2.72344 1 2.5 1.22344 2.5 1.5V14.5C2.5 14.7766 2.72344 15 3 15H13C13.2766 15 13.5 14.7766 13.5 14.5V1.5C13.5 1.22344 13.2766 1 13 1ZM12.375 13.875H3.625V10.7344H5.15469C5.33594 11.2469 5.65469 11.7078 6.07812 12.0578C6.61719 12.5031 7.3 12.75 8 12.75C8.7 12.75 9.38281 12.5047 9.92188 12.0578C10.3453 11.7078 10.6641 11.2469 10.8453 10.7344H12.375V9.75H10.0562L9.975 10.1359C9.7875 11.0703 8.95625 11.75 8 11.75C7.04375 11.75 6.2125 11.0703 6.02344 10.1359L5.94219 9.75H3.625V2.125H12.375V13.875ZM5 5.32812H11C11.0688 5.32812 11.125 5.27188 11.125 5.20312V4.45312C11.125 4.38437 11.0688 4.32812 11 4.32812H5C4.93125 4.32812 4.875 4.38437 4.875 4.45312V5.20312C4.875 5.27188 4.93125 5.32812 5 5.32812ZM5 7.82812H11C11.0688 7.82812 11.125 7.77188 11.125 7.70312V6.95312C11.125 6.88437 11.0688 6.82812 11 6.82812H5C4.93125 6.82812 4.875 6.88437 4.875 6.95312V7.70312C4.875 7.77188 4.93125 7.82812 5 7.82812Z" fill="#373737"/>
</svg>"""


def link(text: str, href: str, icon: Optional[str] = None):
    return f'<a style="display: flex; gap: 0.5rem; align-items: center; text-decoration: none; color: unset; font-weight: 600" href={href}>{icon if icon else ""} {text}</a>'


def render_help():
    html((Path(__file__).parent / "help.html").read_text(encoding="utf-8"), height=0, width=0)
    st.markdown(
        f"""
        <div style="display: flex; flex-direction: column; padding: 1rem">
            <span style="font-size: 1.2rem">Need help?</span>
            {link("Join our Discord community", JOIN_DISCORD_URL, DISCORD_ICON)}
            {link("Documentation", DOCS_URL, DOCS_ICON)}
        </div>""",
        unsafe_allow_html=True,
    )