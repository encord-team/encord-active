import sqlite3
from typing import Optional

import streamlit as st


class DBConnection:
    def __init__(self, file: Optional[str] = None):
        if file:
            self.file = file
        else:
            self.file = st.session_state.db_path

    def __enter__(self):
        self.conn = sqlite3.connect(self.file)
        return self.conn

    def __exit__(self, type, value, traceback):
        self.conn.__exit__(type, value, traceback)
