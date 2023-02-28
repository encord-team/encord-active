from streamlit.delta_generator import DeltaGenerator


def top_padding(delta_generator: DeltaGenerator, padding: int = 1):
    for _ in range(padding):
        delta_generator.write("")
