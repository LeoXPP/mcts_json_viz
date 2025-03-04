import json 
import math
import glob
import numpy as np

from bokeh.plotting import figure, curdoc
from bokeh.models import (
    ColumnDataSource,
    Slider,
    HoverTool,
    Button,
    Select,
    TextInput,
    Div,
    TapTool,
    MutiSelect,
    RangeSlider,
    CustomJS
)

from bokeh.layouts import column, row
from bokeh.palettes import Category10
from collections import defaultdict

def load_json_files(directory, pattern = "xiepanpan_mcts_tree_*.json"):
    """_summary_

    Args:
        directory (_type_): _description_
        patern (str, optional): _description_. Defaults to "xiepanpan_mcts_tree_*.json".
    """
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    all_data = []
    
    for file in files:
        with open(file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error loading{file}")
                continue
            all_data.append((file, data))
    return all_data

def assign_colors(all_data):
    vehicle_ids = sorted(set(node["vehicle_id"] for _, data in all_data for node in data["nodes"]))
    palete = Category10[10]
    other_colors = list(palete)
    vehicle_colors = ()
    color_idx = 0
    for vid in vehicle_ids:
        if vid == " ego":
            vehicle_colors[vid] = 'red'
        else:
            vehicle_colors[vid] = other_colors[color_idx % len(other_colors)]
            color_idx += 1
            return vehicle_colors


def main():
    global all_data, vehicle_colors, scale, p, slider, source, hover, mode_select, top_n_slider
    global vehicle_id_filter, depth_range_filter, reward_range_filter, max_depth_global
    global node_is_filter_input
    global main_centerline_source, shift_centerline_source
    global boundary_right_source, boundary_left_source, road_fill_source
    
    directory = "."
    scale = 1.0
    all_data = load_json_files(directory, pattern = "xiepanpan_mcts_tree_*.json")
    if not all_data:
        print("No Data Found")
        return
    
    vehicle_colors = assign_colors(all_data)
    
    
    
    
    
    

main() 