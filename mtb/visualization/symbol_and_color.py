from typing import Dict, Tuple

import plotly.colors as pc

global_color_map = dict()
global_symbol_map = dict()


def add_category(
    category: str,
):
    """Add a category to the global color and symbol map."""
    global global_color_map, global_symbol_map
    palette = pc.qualitative.Plotly

    plotly_symbols = [
        "circle",
        "square",
        "diamond",
        "cross",
        "x",
        "triangle-up",
        "triangle-down",
        "triangle-left",
        "triangle-right",
        "pentagon",
        "hexagon",
        "hexagon2",
        "star",
        "hourglass",
        "bowtie",
    ]

    if category not in global_color_map:
        index = len(global_color_map)
        global_color_map[category] = palette[index % len(palette)]
        global_symbol_map[category] = plotly_symbols[index % len(plotly_symbols)]


def get_symbol_and_color_map() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Get the global color and symbol map."""
    global global_color_map, global_symbol_map
    return global_color_map, global_symbol_map
