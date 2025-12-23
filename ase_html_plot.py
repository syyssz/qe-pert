#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import plotly.graph_objects as go
from ase import Atoms
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.data.colors import jmol_colors
from ase.data import covalent_radii

# ==========================================
# 1. 参数设置
# ==========================================
qe_input = 'scf.in'
qe_mdtrj = 'total.mdtrj'
output_file = 'movie.html'
step_skip = 10  # 建议根据文件大小调整：文件大就设大一点(如 10 或 50)

# ==========================================
# 2. 修复后的读取逻辑 (更稳健)
# ==========================================
def read_custom_trajectory(input_file, trj_file, skip=1):
    # --- A. 读取原子类型 (nat) ---
    print(f"Reading atom types from {input_file}...")
    nat = 0; symbols = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        if 'nat' in line and nat == 0:
            # 处理 "nat= 10" 或 "nat = 10" 或 "nat, 10" 等情况
            parts = line.replace('=',' ').replace(',',' ').split()
            if 'nat' in parts: 
                try:
                    nat = int(parts[parts.index('nat')+1])
                except: pass
        
        if 'ATOMIC_POSITIONS' in line:
            if nat == 0: raise ValueError("Found ATOMIC_POSITIONS but 'nat' was not found yet.")
            for j in range(nat):
                # 防止空行导致越界
                if i+1+j < len(lines):
                    parts = lines[i+1+j].split()
                    if parts: symbols.append(parts[0])
            break
    
    if nat == 0 or not symbols:
        raise ValueError(f"Failed to parse 'nat' or atomic symbols from {input_file}")
    
    print(f"  -> Found {nat} atoms: {symbols[:5]}...")

    # --- B. 逐行读取轨迹 (自动跳过空行) ---
    print(f"Reading trajectory from {trj_file}...")
    traj = []
    buffer = []
    block_size = nat + 4 # 1行Step信息 + 3行Cell + nat行原子
    frame_count = 0
    
    with open(trj_file, 'r') as f:
        for line in f:
            # 关键修复：遇到空行直接跳过，防止错位
            if not line.strip(): continue
            
            buffer.append(line)
            
            # 当攒够一个 Block (一帧) 的数据时
            if len(buffer) == block_size:
                # 只有符合 skip 步长要求的帧才处理
                if frame_count % skip == 0:
                    try:
                        # 解析 Cell (第2-4行)
                        cell_lines = buffer[1:4]
                        cell = np.array([l.split() for l in cell_lines], dtype=float)
                        
                        # 解析 坐标 (最后 nat 行)
                        atom_lines = buffer[4:4+nat]
                        # 取倒数3列 (x, y, z)，兼容前面可能有原子名的情况
                        pos = np.array([l.split()[-3:] for l in atom_lines], dtype=float)
                        
                        atoms = Atoms(symbols=symbols, positions=pos, cell=cell, pbc=True)
                        traj.append(atoms)
                    except Exception as e:
                        print(f"Warning: Error parsing frame {frame_count}, skipping. Error: {e}")
                
                # 清空缓存，准备读下一帧
                buffer = []
                frame_count += 1
                
    print(f"  -> Successfully loaded {len(traj)} frames (sampled from {frame_count} total steps).")
    return traj, symbols

# ==========================================
# 3. 核心逻辑：利用 ASE 计算 + Plotly 绘图
# ==========================================

# 读取数据
traj, symbols = read_custom_trajectory(qe_input, qe_mdtrj, skip=step_skip)

if not traj:
    raise ValueError("No frames were loaded! Check your file path or format.")

# 预计算样式
unique_syms = sorted(list(set(symbols)))
colors = {s: '#%02x%02x%02x' % tuple(int(c*255) for c in jmol_colors[Z]) 
          for s, Z in zip(unique_syms, [Atoms(s).numbers[0] for s in unique_syms])}
sizes = {s: (covalent_radii[Atoms(s).numbers[0]] + 0.2) * 20 for s in unique_syms}

# 初始化 Plotly 图表
fig = go.Figure()

print("Generating visualization frames...")
frames = []
sliders_dict = {"steps": []}

# 遍历每一帧
for k, atoms in enumerate(traj):
    atoms.wrap() # ASE 自动处理周期性边界条件
    
    # --- A. 利用 ASE 自动计算成键 ---
    cutoffs = natural_cutoffs(atoms, mult=1.15) 
    nl = NeighborList(cutoffs, self_interaction=False, bothways=False)
    nl.update(atoms)
    
    bond_x, bond_y, bond_z = [], [], []
    pos = atoms.get_positions()
    cell = atoms.get_cell()
    
    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        for j, offset in zip(indices, offsets):
            p1 = pos[i]
            p2 = pos[j] + np.dot(offset, cell)
            bond_x.extend([p1[0], p2[0], None])
            bond_y.extend([p1[1], p2[1], None])
            bond_z.extend([p1[2], p2[2], None])

    # --- B. 提取晶胞框 ---
    corners = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1],
                        [1,1,0], [1,0,1], [0,1,1], [1,1,1]])
    cart_corners = np.dot(corners, cell)
    lines = [(0,1), (0,2), (0,3), (1,4), (1,5), (2,4), (2,6), (3,5), (3,6), (4,7), (5,7), (6,7)]
    cell_x, cell_y, cell_z = [], [], []
    for p1, p2 in lines:
        cell_x.extend([cart_corners[p1][0], cart_corners[p2][0], None])
        cell_y.extend([cart_corners[p1][1], cart_corners[p2][1], None])
        cell_z.extend([cart_corners[p1][2], cart_corners[p2][2], None])

    # --- C. 构建当前帧的数据 ---
    frame_data = [
        # 1. 晶胞线
        go.Scatter3d(x=cell_x, y=cell_y, z=cell_z, mode='lines', line=dict(color='black', width=4), hoverinfo='none', showlegend=False),
        # 2. 化学键
        go.Scatter3d(x=bond_x, y=bond_y, z=bond_z, mode='lines', line=dict(color='#444', width=4), hoverinfo='none', showlegend=False),
    ]
    
    # 3. 原子 (按元素分组绘制)
    for sym in unique_syms:
        indices = [i for i, s in enumerate(symbols) if s == sym]
        if not indices: continue
        p_sub = pos[indices]
        frame_data.append(
            go.Scatter3d(
                x=p_sub[:,0], y=p_sub[:,1], z=p_sub[:,2],
                mode='markers',
                marker=dict(size=sizes[sym], color=colors[sym], line=dict(width=1, color='black')),
                name=sym, text=[f"{sym} #{idx}" for idx in indices]
            )
        )

    # 第一帧的数据添加到主图
    if k == 0:
        for trace in frame_data: fig.add_trace(trace)
    
    frames.append(go.Frame(data=frame_data, name=str(k)))
    
    # 进度条
    sliders_dict["steps"].append({
        "args": [[str(k)], {"frame": {"duration": 50, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
        "label": str(k),
        "method": "animate"
    })

# ==========================================
# 4. 设置布局与导出
# ==========================================
fig.update_layout(
    title=f"Trajectory Viewer ({len(traj)} frames)",
    scene=dict(aspectmode='data'),
    updatemenus=[{
        "buttons": [
            {"args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}], "label": "Play", "method": "animate"},
            {"args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}], "label": "Pause", "method": "animate"}
        ],
        "type": "buttons", "showactive": False, "x": 0.1, "y": 0
    }],
    sliders=[sliders_dict]
)

fig.frames = frames

print(f"Writing {output_file} ...")
fig.write_html(output_file)
print("Done! Download and open movie.html")
