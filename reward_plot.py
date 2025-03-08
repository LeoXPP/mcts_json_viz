import glob
import json
from collections import defaultdict

from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Select, MultiSelect, HoverTool
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.palettes import Category10

# 文件路径模式
FILE_PATH = "xiepanpan_mcst_tree_*.json"

# 读取所有符合条件的文件，存入字典：文件名 -> 节点列表
data_dict = {}
file_list = glob.glob(FILE_PATH)
if not file_list:
    raise RuntimeError("没有找到匹配的文件，请检查路径和文件名！")

for f in file_list:
    with open(f, "r", encoding="utf-8") as infile:
        json_data = json.load(infile)
        # 获取节点数据列表，若不存在则设为空列表
        nodes = json_data.get("node_reward_iteration", [])
        data_dict[f] = nodes

# --- Bokeh 控件和图像初始化 ---

# 文件选择器：以文件名作为选项
file_options = sorted(list(data_dict.keys()))
file_select = Select(title="选择文件", value=file_options[0], options=file_options)

# 节点多选控件：初始选项根据首个文件中节点数目生成
def get_node_options(file_name):
    nodes = data_dict[file_name]
    # 以节点的索引作为标识
    return [("{}".format(i), "节点 {}".format(i)) for i in range(len(nodes))]

node_options = get_node_options(file_select.value)
# 默认不选中任何节点
node_multiselect = MultiSelect(title="选择节点（可多选）", value=[], options=node_options, size=6)

# 创建图像：设置学术风格（简洁坐标轴、网格虚线、工具条在上方）
p = figure(title="Reward 迭代图", 
           x_axis_label="Iteration", y_axis_label="Reward",
           tools="pan,wheel_zoom,box_zoom,reset,save", toolbar_location="above")
p.grid.grid_line_dash = [6, 4]

# 添加 Hover 工具（显示迭代和 reward 数值）
hover = HoverTool(tooltips=[("Iteration", "@x"), ("Reward", "@y")])
p.add_tools(hover)

# --- 更新绘图函数 ---
def update_plot():
    # 清除已有的绘图（除工具等）
    p.renderers = []
    
    # 获取当前文件和选中节点
    selected_file = file_select.value
    selected_nodes = node_multiselect.value  # 节点索引的字符串列表
    nodes = data_dict[selected_file]
    
    # 更新图像标题
    p.title.text = f"Reward 迭代图 - 文件: {selected_file}"
    
    # 为区分不同节点，选择一个调色板
    num_lines = len(selected_nodes)
    palette = Category10[10] if num_lines <= 10 else Category10[10] * ((num_lines // 10) + 1)
    
    # 为每个选中的节点绘制折线图
    for idx, node_str in enumerate(selected_nodes):
        node_idx = int(node_str)
        if node_idx < 0 or node_idx >= len(nodes):
            continue
        node = nodes[node_idx]
        history = node.get("history", [])
        # 提取迭代次数和对应的 reward
        iterations = [item["iteration"] for item in history]
        rewards = [item["reward"] for item in history]
        source = ColumnDataSource(data={"x": iterations, "y": rewards})
        
        color = palette[idx]
        p.line("x", "y", source=source, line_width=2, color=color, legend_label=f"节点 {node_idx}")
        p.circle("x", "y", source=source, size=6, color=color)
    
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"

# --- 控件回调 ---
def file_select_callback(attr, old, new):
    # 当文件改变时，更新节点选项，并清空当前选中
    node_multiselect.options = get_node_options(new)
    node_multiselect.value = []
    update_plot()

def node_select_callback(attr, old, new):
    update_plot()

file_select.on_change("value", file_select_callback)
node_multiselect.on_change("value", node_select_callback)

# 初始绘制（未选择节点时可提示用户选择节点）
update_plot()

# --- 布局 ---
layout = column(row(file_select, node_multiselect), p)
curdoc().add_root(layout)
curdoc().title = "节点 Reward 迭代图"
