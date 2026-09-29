"""Streamlit demonstration UI for the FIB simulation workflow.

Run from the project virtual environment with:
    streamlit run fib_gui_demo.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st


PACKAGE = Path(__file__).resolve().parent
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from material_library import available_materials  # noqa: E402
from simu_workflow import (  # noqa: E402
    add_material_layer,
    define_etch_region,
    define_simulation_size,
    prepare_simulation,
    run_simulation,
    set_process_parameters,
)


st.set_page_config(page_title="FIB 模拟演示", page_icon="🔬", layout="wide")
st.title("FIB 数值模拟")
st.caption("演示版网页界面：均匀材料层 + 矩形扫描区，调用现有 simu_workflow 求解。")

with st.sidebar:
    st.header("模型与工艺参数")
    st.subheader("计算域")
    col_x, col_y, col_z = st.columns(3)
    size_x = col_x.number_input("X 尺寸 (nm)", min_value=10.0, value=120.0, step=10.0)
    size_y = col_y.number_input("Y 尺寸 (nm)", min_value=10.0, value=120.0, step=10.0)
    size_z = col_z.number_input("Z 尺寸 (nm)", min_value=10.0, value=60.0, step=10.0)
    cell = st.number_input("体素尺寸 (nm)", min_value=0.5, value=2.0, step=0.5)

    st.subheader("初始结构")
    materials = available_materials()
    selected_material = st.selectbox("材料", materials, index=materials.index("Al") if "Al" in materials else 0)
    layer_thickness = st.number_input("材料层厚度 (nm)", min_value=cell, value=45.0, step=1.0)

    st.subheader("扫描区域")
    c1, c2 = st.columns(2)
    x_start = c1.number_input("X 起点 (nm)", min_value=0.0, value=30.0, step=5.0)
    x_end = c2.number_input("X 终点 (nm)", min_value=0.0, value=90.0, step=5.0)
    c3, c4 = st.columns(2)
    y_start = c3.number_input("Y 起点 (nm)", min_value=0.0, value=30.0, step=5.0)
    y_end = c4.number_input("Y 终点 (nm)", min_value=0.0, value=90.0, step=5.0)
    pitch = st.number_input("扫描点间距 (nm)", min_value=1.0, value=10.0, step=1.0)

    st.subheader("束流参数")
    cycles = st.number_input("完整扫描循环数", min_value=1, value=1, step=1)
    voltage = st.number_input("加速电压 (kV)", min_value=1.0, value=30.0, step=1.0)
    current_pa = st.number_input("束流 (pA)", min_value=0.1, value=42.0, step=1.0)
    dwell_us = st.number_input("驻留时间 (µs)", min_value=0.01, value=0.975, step=0.1, format="%.3f")
    sigma = st.number_input("束斑 sigma (nm)", min_value=0.1, value=5.0, step=0.5)
    redeposition = st.checkbox("启用再沉积", value=True)

    output_dir = st.text_input("输出目录", value=str(Path.home() / "fib_gui_demo_output"))
    run_clicked = st.button("校验并运行模拟", type="primary", use_container_width=True)

left, right = st.columns([1, 1])
with left:
    st.subheader("运行前检查")
    st.write(f"网格约为 **{int(size_x / cell)} × {int(size_y / cell)} × {int(size_z / cell)}** 个体素")
    st.write(f"初始材料：**{selected_material}**，层厚 **{layer_thickness:g} nm**")
    st.write(f"扫描范围：X [{x_start:g}, {x_end:g}] nm；Y [{y_start:g}, {y_end:g}] nm")

with right:
    st.subheader("结果")
    st.info("运行完成后会在这里显示表面图和输出文件。")

if run_clicked:
    status = st.status("正在校验模型参数…", expanded=True)
    try:
        simulation = define_simulation_size(
            size_nm=(size_x, size_y, size_z),
            cell_size_nm=cell,
            output_dir=output_dir,
        )
        add_material_layer(simulation, selected_material, thickness_nm=layer_thickness)
        define_etch_region(
            simulation,
            x_range_nm=(x_start, x_end),
            y_range_nm=(y_start, y_end),
            pitch_nm=pitch,
        )
        set_process_parameters(
            simulation,
            scan_cycles=int(cycles),
            voltage_kv=voltage,
            beam_current_a=current_pa * 1e-12,
            dwell_time_s=dwell_us * 1e-6,
            beam_sigma_nm=sigma,
            beam_effect_radius_nm=2.0 * sigma,
            enable_redeposition=redeposition,
            redeposition_rays=16,
            snapshot_count=20,
        )
        state, params = prepare_simulation(simulation)
        status.write(f"网格：{state.shape}；扫描点：{params['N_POINTS_X']} × {params['N_POINTS_Y']}")
        status.write(f"总时间步：{params['N_STEPS']:,}；时间步长：{params['DT_S']:.3g} s")
        status.update(label="正在运行模拟…", state="running", expanded=True)
        result_dir = run_simulation(simulation)
        status.update(label="模拟完成", state="complete", expanded=False)

        st.session_state["fib_result_dir"] = str(result_dir)
        st.success(f"结果已保存到：{result_dir}")
    except Exception as exc:
        status.update(label="校验或计算失败", state="error", expanded=True)
        st.exception(exc)

result_dir_text = st.session_state.get("fib_result_dir")
if result_dir_text:
    result_dir = Path(result_dir_text)
    preview = result_dir / "surface3d_final.png"
    if not preview.exists():
        preview = result_dir / "initial_structure_xy.png"
    if preview.exists():
        st.image(str(preview), caption=preview.name, use_container_width=True)
    existing = sorted(path for path in result_dir.iterdir() if path.is_file())
    if existing:
        with st.expander(f"输出文件（{len(existing)} 个）"):
            for path in existing:
                st.write(f"`{path.name}` — {path.stat().st_size:,} bytes")
