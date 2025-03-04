import json
import math
import os
import glob
import numpy as np  # 引入 NumPy 进行矩阵运算

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
    MultiSelect,
    RangeSlider,
    CustomJS
)
from bokeh.layouts import column, row
from bokeh.palettes import Category10
from collections import defaultdict

def load_json_files(directory, pattern="xiepanpan_mcst_tree_*.json"):
    """
    加载指定目录下所有匹配模式的 JSON 文件，并按文件名排序。
    """
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    all_data = []
    for file in files:
        with open(file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"无法解码 JSON 文件: {file}")
                continue
            all_data.append((file, data))
    return all_data

def assign_colors(all_data):
    """
    为每个车辆分配颜色。自车（'ego'）使用红色，其他车辆使用 Category10 调色板中的颜色。
    根据新的数据格式，从 bestnodeseq 中提取各车辆的 vehicle_id。
    """
    vehicle_ids = sorted(set(
        state["vehicle_id"]
        for _, data in all_data
        for best_node in data.get("bestnodeseq", [])
        for state in best_node.get("vehicle_states", [])
    ))
    palette = Category10[10]  # 使用 Category10 调色板的前10种颜色
    other_colors = list(palette)
    vehicle_colors = {}
    color_idx = 0
    for vid in vehicle_ids:
        if vid == "ego":
            vehicle_colors[vid] = 'red'
        else:
            vehicle_colors[vid] = other_colors[color_idx % len(other_colors)]
            color_idx += 1
    return vehicle_colors

def get_plot_limits(nodes, scale=1.0):
    """
    根据节点的局部坐标计算绘图的范围，并添加一定的填充。
    """
    all_x = [node["x_local"] / scale for node in nodes]
    all_y = [node["y_local"] / scale for node in nodes]
    if not all_x or not all_y:
        return -10, 10, -10, 10
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    padding_x = 4.0 
    padding_y = 4.0 
    return min_x - padding_x, max_x + padding_x, min_y - padding_y, max_y + padding_y

def calculate_rotated_rectangle(x, y, length, width, angle_deg):
    """
    计算旋转后的矩形四个角的坐标。
    """
    theta = math.radians(angle_deg)
    dx = length / 2
    dy = width / 2
    corners = [
        (-dx, -dy),
        (-dx, dy),
        (dx, dy),
        (dx, -dy)
    ]
    rotated_corners = []
    for cx, cy in corners:
        rx = cx * math.cos(theta) - cy * math.sin(theta) + x
        ry = cx * math.sin(theta) + cy * math.cos(theta) + y
        rotated_corners.append((rx, ry))
    return rotated_corners

def point_in_polygon(x, y, poly_xs, poly_ys):
    """
    判断点 (x, y) 是否在多边形内部。
    """
    num = len(poly_xs)
    j = num - 1
    c = False
    for i in range(num):
        if ((poly_ys[i] > y) != (poly_ys[j] > y)) and \
           (x < (poly_xs[j] - poly_xs[i]) * (y - poly_ys[i]) / (poly_ys[j] - poly_ys[i] + 1e-9) + poly_xs[i]):
            c = not c
        j = i
    return c

def get_transform_matrix_from_initial_pose(initial_pose):
    """
    构建从世界坐标系到初始局部坐标系的变换矩阵。
    初始局部坐标系的x轴为横向，y轴为前进方向。
    """
    x0, y0, theta0 = initial_pose
    phi = math.pi / 2 - theta0  # 旋转角度，确保前进方向为正Y轴
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    T = np.array(
        [
            [cos_phi, -sin_phi, -x0 * cos_phi + y0 * sin_phi],
            [sin_phi, cos_phi, -x0 * sin_phi - y0 * cos_phi],
            [0, 0, 1]
        ]
    )
    return T

def apply_transformation(x, y, T):
    """
    应用变换矩阵 T 到点 (x, y)。
    """
    point = np.array([x, y, 1])
    transformed_point = T @ point
    return transformed_point[0], transformed_point[1]

def prepare_frame_data(data, vehicle_colors, scale=1.0, mode='Top N', top_n=3, node_id_filter=None):
    """
    为当前帧准备绘图数据，将全局坐标转换为局部坐标，并应用各种过滤器。
    数据现在来自 "bestnodeseq" 列表，每个元素包含 "vehicle_states" 列表。
    """
    best_nodes = data.get("bestnodeseq", [])
    if not best_nodes:
        return None, None, None, None, None

    # 提取初始 ego 状态（从第一个 bestnodeseq 中寻找 ego）
    initial_ego_state = None
    for best_node in best_nodes:
        for state in best_node.get("vehicle_states", []):
            if state.get("vehicle_id") == "ego":
                initial_ego_state = state
                break
        if initial_ego_state:
            break

    if initial_ego_state is None:
        print("当前帧的ego车辆未找到。")
        return None, None, None, None, None

    initial_x0 = initial_ego_state["x"]
    initial_y0 = initial_ego_state["y"]
    initial_theta0 = initial_ego_state["theta"]

    T = get_transform_matrix_from_initial_pose((initial_x0, initial_y0, initial_theta0))

    valid_states = []
    # 遍历 bestnodeseq 中每个节点的车辆状态
    for best_node in best_nodes:
        for state in best_node.get("vehicle_states", []):
            if state.get("x") is None or state.get("y") is None:
                continue

            x_global = state["x"]
            y_global = state["y"]
            theta_global = state["theta"]

            x_local, y_local = apply_transformation(x_global, y_global, T)
            theta_local = theta_global - initial_theta0

            # 构造新的节点数据，将当前车辆状态与父节点的属性合并
            new_node = {}
            new_node["x_local"] = x_local
            new_node["y_local"] = y_local
            new_node["theta_local"] = theta_local
            new_node["vehicle_id"] = state.get("vehicle_id", "")
            new_node["reward"] = best_node.get("reward", 0)
            new_node["depth"] = best_node.get("iter", 0)  # 使用 iter 作为深度
            new_node["iter"] = best_node.get("iter", 0)
            new_node["id"] = best_node.get("id", "")
            new_node["acc"] = state.get("acc", "")
            new_node["vel"] = state.get("vel", "")
            # 对于原数据中没有的信息，设置默认值
            new_node["visits"] = 0
            new_node["expanded_num"] = 0
            new_node["size"] = 0
            new_node["max_size"] = 0
            new_node["is_valid"] = False
            new_node["static_reward"] = 0.0
            new_node["relative_time"] = 0.0
            # 新数据中没有 vehicle_rewards，置为空
            new_node["vehicle_rewards"] = "{}"
            # 将当前车辆状态以 JSON 形式存储
            new_node["vehicle_states"] = json.dumps(state, indent=2)

            valid_states.append(new_node)
    
    if not valid_states:
        print("所有状态均无效（x 或 y 为 null）。")
        return None, None, None, None, None

    # 根据变换后的坐标计算绘图范围
    all_new_x = [node["y_local"] / scale for node in valid_states]  # 新x为原y
    all_new_y = [-node["x_local"] / scale for node in valid_states]  # 新y为-原x
    if not all_new_x or not all_new_y:
        min_new_x, max_new_x, min_new_y, max_new_y = -10, 10, -10, 10
    else:
        min_new_x, max_new_x = min(all_new_x), max(all_new_x)
        min_new_y, max_new_y = min(all_new_y), max(all_new_y)
    padding_x = 4.0 
    padding_y = 4.0 
    plot_limits = (min_new_x - padding_x, max_new_x + padding_x, min_new_y - padding_y, max_new_y + padding_y)

    # 更新全局最大深度（这里用 iter 作为深度）
    global max_depth_global
    max_depth = max(node["depth"] for node in valid_states)
    if max_depth > max_depth_global:
        max_depth_global = max_depth
        depth_range_filter.end = max_depth_global
        depth_range_filter.value = (depth_range_filter.value[0], max_depth_global)

    # 应用过滤器
    selected_nodes = [node for node in valid_states if
                      node["vehicle_id"] in vehicle_id_filter.value and
                      depth_range_filter.value[0] <= node["depth"] <= depth_range_filter.value[1] and
                      reward_range_filter.value[0] <= node.get("reward", 0) <= reward_range_filter.value[1]]

    # 应用节点 ID 筛选
    if node_id_filter:
        node_ids = [nid.strip() for nid in node_id_filter.split(',') if nid.strip()]
        if node_ids:
            selected_nodes = [node for node in selected_nodes if node.get('id', '') in node_ids]

    # 如果按照深度分组并选择 Top N，需要调整
    if mode == 'Top N':
        depth_groups = defaultdict(list)
        for node in selected_nodes:
            depth_groups[node["depth"]].append(node)
        filtered_nodes = []
        for depth in sorted(depth_groups.keys()):
            group = depth_groups[depth]
            sorted_group = sorted(group, key=lambda n: n.get("reward", 0), reverse=True)
            top_n_nodes = sorted_group[:top_n]
            filtered_nodes.extend(top_n_nodes)
        selected_nodes = filtered_nodes

    polygons = []
    for node in selected_nodes:
        x = node["y_local"] / scale  # 新x为原y
        y = -node["x_local"] / scale  # 新y为-原x
        theta = node["theta_local"] - math.pi / 2  # 新角度减去90度
        vid = node["vehicle_id"]
        depth = node["depth"]
        reward = node.get("reward", 0)
        color = vehicle_colors.get(vid, 'blue')
        alpha = 1.0 - (depth / (max_depth + 1))

        # 固定车辆尺寸：长1米，宽2米
        length = 1.0
        width = 2.0

        # 计算旋转后的矩形四个角的坐标，使用新角度
        corners = calculate_rotated_rectangle(x, y, length, width, math.degrees(theta))
        xs = [point[0] for point in corners]
        ys = [point[1] for point in corners]

        polygons.append({
            'xs': xs,
            'ys': ys,
            'color': color,
            'alpha': alpha,
            'category': 'ego' if vid == 'ego' else 'other',
            'vid': vid,
            'depth': depth,
            'reward': reward,
            'acc': node.get("acc", ""),
            'vel': node.get("vel", ""),
            'vehicle_rewards': node.get("vehicle_rewards", "{}"),
            'vehicle_states': node.get("vehicle_states", "{}"),
            'visits': node.get("visits", 0),
            'expanded_num': node.get("expanded_num", 0),
            'iter': node.get("iter", 0),
            'id': node.get("id", ""),
            'size': node.get("size", 0),
            'max_size': node.get("max_size", 0),
            'is_valid': node.get("is_valid", False),
            'static_reward': node.get("static_reward", 0.0),
            'relative_time': node.get("relative_time", 0.0),
        })

    data_dict = dict(
        xs=[poly['xs'] for poly in polygons],
        ys=[poly['ys'] for poly in polygons],
        color=[poly['color'] for poly in polygons],
        alpha=[poly['alpha'] for poly in polygons],
        category=[poly['category'] for poly in polygons],
        vid=[poly['vid'] for poly in polygons],
        depth=[poly['depth'] for poly in polygons],
        reward=[poly['reward'] for poly in polygons],
        acc=[poly['acc'] for poly in polygons],
        vel=[poly['vel'] for poly in polygons],
        vehicle_rewards=[poly['vehicle_rewards'] for poly in polygons],
        vehicle_states=[poly['vehicle_states'] for poly in polygons],
        visits=[poly['visits'] for poly in polygons],
        expanded_num=[poly['expanded_num'] for poly in polygons],
        iter=[poly['iter'] for poly in polygons],
        id=[poly['id'] for poly in polygons],
        size=[poly['size'] for poly in polygons],
        max_size=[poly['max_size'] for poly in polygons],
        is_valid=[poly['is_valid'] for poly in polygons],
        static_reward=[poly['static_reward'] for poly in polygons],
        relative_time=[poly['relative_time'] for poly in polygons],
    )

    current_vehicle_ids = sorted(set(node["vehicle_id"] for node in selected_nodes))

    return data_dict, plot_limits, current_vehicle_ids, max_depth, selected_nodes

def update_plot(attr, old, new):
    """
    更新绘图数据和视图范围。
    """
    frame_idx = slider.value
    file, data = all_data[frame_idx]

    current_mode = mode_select.value
    node_id_filter_val = node_id_filter_input.value  # 获取节点 ID 过滤器的值

    new_data, plot_limits, current_vehicle_ids, max_depth, nodes = prepare_frame_data(
        data,
        vehicle_colors,
        scale=scale,
        mode=current_mode,
        top_n=top_n_slider.value,
        node_id_filter=node_id_filter_val
    )

    if new_data is None:
        return

    min_new_x, max_new_x, min_new_y, max_new_y = plot_limits

    p.x_range.start = min_new_x
    p.x_range.end = max_new_x
    p.y_range.start = min_new_y
    p.y_range.end = max_new_y

    p.title.text = f'MCTS Tree Vehicle Visualization\nFile: {os.path.basename(file)}'

    # 更新道路线条
    main_centerline_source.data = dict(x=[min_new_x, max_new_x], y=[0, 0])
    shifted_centerline_source.data = dict(x=[min_new_x, max_new_x], y=[3, 3])
    boundary_right_source.data = dict(x=[min_new_x, max_new_x], y=[-2, -2])
    boundary_left_source.data = dict(x=[min_new_x, max_new_x], y=[6, 6])
    road_fill_source.data = dict(x=[min_new_x, max_new_x, max_new_x, min_new_x], y=[-2, -2, 6, 6])

    source.data = new_data
    p.xaxis.axis_label = 'Forward Direction (meters)'
    p.yaxis.axis_label = 'Lateral Direction (meters)'

def update_mode(attr, old, new):
    """
    更新显示模式（Top N 或 All）。
    """
    if mode_select.value == 'Top N':
        top_n_slider.disabled = False
    else:
        top_n_slider.disabled = True
    update_plot(None, None, None)

def update_top_n(attr, old, new):
    """
    更新 Top N 滑块的值。
    """
    update_plot(None, None, None)

def update_plot_dimensions(attr, old, new):
    """
    动态调整绘图的宽度和高度。
    """
    p.width = width_slider.value
    p.height = height_slider.value

def main():
    global all_data, vehicle_colors, scale, p, slider, source, hover, mode_select, top_n_slider
    global vehicle_id_filter, depth_range_filter, reward_range_filter, max_depth_global
    global node_id_filter_input
    global width_slider, height_slider
    global main_centerline_source, shifted_centerline_source
    global boundary_right_source, boundary_left_source, road_fill_source

    directory = '.'   # JSON 文件目录
    scale = 1.0

    all_data = load_json_files(directory, pattern="xiepanpan_mcst_tree_*.json")
    if not all_data:
        print("OH MO !!! JSON NOT FOUND")
        return

    vehicle_colors = assign_colors(all_data)

    # 初始化全局最大深度（这里用 bestnodeseq 中的 iter 作为深度）
    max_depth_global = max(
        best_node.get("iter", 0)
        for _, data in all_data
        for best_node in data.get("bestnodeseq", [])
    )

    # 车辆 ID 过滤器
    vehicle_ids = sorted(set(
        state["vehicle_id"]
        for _, data in all_data
        for best_node in data.get("bestnodeseq", [])
        for state in best_node.get("vehicle_states", [])
    ))
    vehicle_id_filter = MultiSelect(title="Filter by Vehicle ID", value=vehicle_ids, options=vehicle_ids)
    vehicle_id_filter.on_change('value', lambda attr, old, new: update_plot(None, None, None))

    # 深度范围过滤器
    depth_range_filter = RangeSlider(start=0, end=max_depth_global, value=(0, max_depth_global), step=1, title="Filter by Depth")
    depth_range_filter.on_change('value', lambda attr, old, new: update_plot(None, None, None))

    # 奖励值范围过滤器
    all_rewards = [best_node.get("reward", 0)
                   for _, data in all_data
                   for best_node in data.get("bestnodeseq", [])]
    if all_rewards:
        min_reward = min(all_rewards)
        max_reward = max(all_rewards)
    else:
        min_reward = 0
        max_reward = 0
    reward_range_filter = RangeSlider(start=min_reward, end=max_reward, value=(min_reward, max_reward), step=0.1, title="Filter by Reward")
    reward_range_filter.on_change('value', lambda attr, old, new: update_plot(None, None, None))

    # 节点 ID 过滤器
    node_id_filter_input = TextInput(title="Filter by Node ID (comma-separated)", value="")
    node_id_filter_input.on_change('value', lambda attr, old, new: update_plot(None, None, None))

    initial_frame_idx = 0
    file, data = all_data[initial_frame_idx]
    new_data, plot_limits, current_vehicle_ids, max_depth, nodes = prepare_frame_data(
        data,
        vehicle_colors,
        scale=scale,
        mode='Top N',
        top_n=3,
        node_id_filter=""
    )

    if new_data is None:
        print("初始帧数据准备失败。")
        return

    # 创建 ColumnDataSource
    source = ColumnDataSource(data=new_data)

    # 创建绘图
    p = figure(
        title=f'MCTS Tree Vehicle Visualization\nFile: {os.path.basename(file)}',
        x_range=(plot_limits[0], plot_limits[1]),
        y_range=(plot_limits[2], plot_limits[3]),
        width=800,
        height=600,  # 调整为更宽的显示
        tools="pan,box_zoom,reset,save",
        output_backend="canvas"
    )
    p.background_fill_color = "white"
    p.grid.grid_line_color = None
    p.xaxis.axis_label = 'Forward Direction (meters)'
    p.yaxis.axis_label = 'Lateral Direction (meters)'
    p.title.text_font_size = '20pt'
    p.xaxis.axis_label_text_font_size = '14pt'
    p.yaxis.axis_label_text_font_size = '14pt'

    # 创建道路线条和填充的 ColumnDataSource
    min_new_x, max_new_x, min_new_y, max_new_y = plot_limits
    main_centerline_source = ColumnDataSource(data=dict(x=[min_new_x, max_new_x], y=[0, 0]))
    shifted_centerline_source = ColumnDataSource(data=dict(x=[min_new_x, max_new_x], y=[4, 4]))
    boundary_right_source = ColumnDataSource(data=dict(x=[min_new_x, max_new_x], y=[-4, -4]))
    boundary_left_source = ColumnDataSource(data=dict(x=[min_new_x, max_new_x], y=[6, 6]))
    road_fill_source = ColumnDataSource(data=dict(x=[min_new_x, max_new_x, max_new_x, min_new_x], y=[-4, -4, 6, 6]))

    # 绘制道路填充
    road_fill_renderer = p.patch('x', 'y', source=road_fill_source,
                                 fill_color='lightgrey', fill_alpha=0.5, line_alpha=0)
    main_centerline_renderer = p.line('x', 'y', source=main_centerline_source,
                                      line_width=2, line_dash='dashed', line_color='black')
    shifted_centerline_renderer = p.line('x', 'y', source=shifted_centerline_source,
                                         line_width=2, line_dash='dashed', line_color='black')
    boundary_right_renderer = p.line('x', 'y', source=boundary_right_source,
                                     line_width=2, line_color='black')
    boundary_left_renderer = p.line('x', 'y', source=boundary_left_source,
                                    line_width=2, line_color='black')

    # 创建车辆多边形渲染器
    renderer = p.patches('xs', 'ys', source=source,
                         fill_color='color', fill_alpha='alpha',
                         line_color='black', line_width=2,
                         legend_field='category')

    # 添加 TapTool
    taptool = TapTool()
    p.add_tools(taptool)

    # 奖励详情显示区域
    reward_details_div = Div(text="<h2>Vehicle Rewards Details</h2><p>Click on a vehicle to see details.</p>", width=400)

    def vehicle_tap_callback(event):
        x = event.x
        y = event.y

        indices = []
        for i in range(len(source.data['xs'])):
            xs = source.data['xs'][i]
            ys = source.data['ys'][i]
            if point_in_polygon(x, y, xs, ys):
                indices.append(i)

        if indices:
            index = indices[0]
            vid = source.data['vid'][index]
            vehicle_rewards = source.data['vehicle_rewards'][index]
            vehicle_states = source.data['vehicle_states'][index]
            depth = source.data['depth'][index]
            reward = source.data['reward'][index]
            acc = source.data['acc'][index]
            vel = source.data['vel'][index]
            visits = source.data['visits'][index]
            expanded_num = source.data['expanded_num'][index]
            iter_ = source.data['iter'][index]
            id_ = source.data['id'][index]
            size = source.data['size'][index]
            max_size = source.data['max_size'][index]
            is_valid = source.data['is_valid'][index]
            static_reward = source.data['static_reward'][index]
            relative_time = source.data['relative_time'][index]

            try:
                vehicle_rewards_dict = json.loads(vehicle_rewards)
            except Exception:
                vehicle_rewards_dict = {}
            rewards_html = "<ul>"
            for key, value in vehicle_rewards_dict.items():
                rewards_html += f"<li><b>{key}:</b> {value}</li>"
            rewards_html += "</ul>"

            try:
                vehicle_states_dict = json.loads(vehicle_states)
            except Exception:
                vehicle_states_dict = {}
            vehicle_states_html = "<ul>"
            for key, value in vehicle_states_dict.items():
                vehicle_states_html += f"<li><b>{key}:</b> {value}</li>"
            vehicle_states_html += "</ul>"

            node_info_html = f"""
            <h2>Vehicle ID: {vid}</h2>
            <p><b>ID:</b> {id_}</p>
            <p><b>Depth:</b> {depth}</p>
            <p><b>Reward:</b> {reward}</p>
            <p><b>Acceleration:</b> {acc}</p>
            <p><b>Velocity:</b> {vel}</p>
            <p><b>Visits:</b> {visits}</p>
            <p><b>Expanded Num:</b> {expanded_num}</p>
            <p><b>Iteration:</b> {iter_}</p>
            <p><b>Size:</b> {size}</p>
            <p><b>Max Size:</b> {max_size}</p>
            <p><b>Is Valid:</b> {is_valid}</p>
            <p><b>Static Reward:</b> {static_reward}</p>
            <p><b>Relative Time:</b> {relative_time}</p>
            <h3>Vehicle Rewards:</h3>
            {rewards_html}
            <h3>Vehicle States:</h3>
            {vehicle_states_html}
            """
            reward_details_div.text = node_info_html
        else:
            reward_details_div.text = "<h2>No vehicle selected</h2>"

    p.on_event('tap', vehicle_tap_callback)

    hover = HoverTool(
        renderers=[renderer],
        tooltips=[
            ("Vehicle ID", "@vid"),
            ("Depth", "@depth"),
            ("Reward", "@reward"),
            ("Acceleration", "@acc"),
            ("Velocity", "@vel"),
        ]
    )
    p.add_tools(hover)

    slider = Slider(start=0, end=len(all_data)-1, value=0, step=1, title="Frame", width=300)
    slider.on_change('value', update_plot)

    mode_select = Select(title="Display Mode", value="Top N", options=["Top N", "All"], width=300)
    mode_select.on_change('value', update_mode)

    top_n_slider = Slider(start=1, end=10, value=3, step=1, title="Top N per Depth", disabled=False, width=300)
    top_n_slider.on_change('value', update_top_n)

    play_button = Button(label="► Play", width=100)
    is_playing = False

    def play():
        nonlocal is_playing
        is_playing = True
        play_button.label = "❚❚ Pause"
        curdoc().add_periodic_callback(callback, 1000)

    def pause():
        nonlocal is_playing
        is_playing = False
        play_button.label = "► Play"
        try:
            curdoc().remove_periodic_callback(callback)
        except ValueError:
            pass

    def callback():
        new_value = (slider.value + 1) % len(all_data)
        slider.value = new_value

    def toggle_play():
        nonlocal is_playing
        if is_playing:
            pause()
        else:
            play()

    play_button.on_click(toggle_play)

    width_slider = Slider(start=300, end=3200, value=800, step=50, title="Plot Width (pixels)", width=300)
    width_slider.on_change('value', update_plot_dimensions)

    height_slider = Slider(start=300, end=3200, value=800, step=50, title="Plot Height (pixels)", width=300)
    height_slider.on_change('value', update_plot_dimensions)

    save_button = Button(label="Save Image", button_type="success", width=100)
    save_callback = CustomJS(args=dict(p=p), code="""
        const plot_view = p.view;
        if (!plot_view) {
            console.log("Plot view not found.");
            return;
        }
        const canvas = plot_view.canvas_view.canvas;
        if (!canvas) {
            console.log("Canvas not found.");
            return;
        }
        const dataURL = canvas.toDataURL("image/png");
        const link = document.createElement('a');
        link.href = dataURL;
        link.download = "plot.png";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    """)
    save_button.js_on_click(save_callback)

    controls = column(
        slider,
        mode_select,
        top_n_slider,
        play_button,
        save_button,
        vehicle_id_filter,
        depth_range_filter,
        reward_range_filter,
        node_id_filter_input,
        width_slider,
        height_slider
    )
    left_side = controls

    for control in controls.children:
        control.width = 300

    p.width = width_slider.value
    p.height = height_slider.value
    reward_details_div.width = 400

    layout = row(left_side, p, reward_details_div)

    def set_legend_properties():
        if p.legend:
            for legend in p.legend:
                legend.label_text_font_size = '12pt'
    set_legend_properties()

    curdoc().add_root(layout)
    curdoc().title = "MCTS Tree Vehicle Visualization"
    
main()
