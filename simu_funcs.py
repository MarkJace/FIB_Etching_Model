"""Numerical kernel used by the sequential FIB simulation workflow.

``simu_workflow`` validates user inputs, derives the solver variables and
passes one uppercase parameter snapshot to :func:`configure`.  Calculation
scripts therefore do not interact with this module's globals directly.

Data conventions
----------------
frame[x, y, z] stores the Mo atom count in a voxel. surface[x, y] stores
the highest occupied z index in a column, or -1 for an empty column. Length is
in nm except inside the ray caster, where coordinates are expressed in voxel units.
"""

import os
import time
import math

import matplotlib.pyplot as plt
import numpy as np
import vtk
from numba import njit
from vtk.util.numpy_support import numpy_to_vtk

# All figures in one run use the same physical color scale by default.  The
# driver may override this with COLORBAR_MODE=adaptive for diagnostic plots.
COLORBAR_MODE = "global"


# Names checked by configure(). Other uppercase derived values are copied as well.
_REQUIRED_PARAMETER_NAMES = {
    "DX_NM",
    "DY_NM",
    "DZ_NM",
    "NX",
    "NY",
    "NZ",
    "INITIAL_SOLID_LAYERS",
    "ATOM_COUNT_PER_CELL",
    "ATOM_EPSILON",
    "ION_RATE_S",
    "BEAM_EFFECT_RADIUS_NM",
    "BEAM_SIGMA_NM",
    "N_POINTS_X",
    "N_POINTS_PER_PASS",
    "STEPS_PER_DWELL",
    "DT_S",
    "N_STEPS",
    "FULL_N_STEPS",
    "PASS_PERIOD_STEPS",
    "SCAN_STEPS_PER_PASS",
    "SCAN_X_START_NM",
    "SCAN_Y_START_NM",
    "PITCH_X_NM",
    "PITCH_Y_NM",
    "SCAN_INITIAL_X_NM",
    "SCAN_INITIAL_Y_NM",
    "SCAN_FINAL_X_NM",
    "SCAN_FINAL_Y_NM",
    "IB_RETURN_VELOCITY_NM_S",
    "IB_RETURN_DISTANCE_NM",
    "N_PASS",
    "Y0",
    "A_SIGMUND",
    "ALPHA_SIGMUND",
    "BETA_SIGMUND",
    "USE_POISSON_ION_ARRIVAL",
    "ION_RNG",
    "ENABLE_REDEPOSTION",
    "REDEPO_RAYS",
    "REDEPO_COS_POWER",
    "STICKING_COEFFICIENT",
    "STICKING_POWER",
    "REDEPOSITION_RNG",
    "OUTPUT_DIR",
    "SNAPSHOT_INTERVAL",
    "SURFACE_HEIGHT_PLOT_RANGE_NM",
}


def configure(parameters):
    """Load driver parameters into this function module.

    Parameters
    ----------
    parameters : Mapping[str, object]
        Normally the snapshot produced by ``prepare_simulation()``.
        Every uppercase entry is copied, so the computational functions use one
        consistent snapshot of both prescribed and derived variables.

    Returns
    -------
    None
        Module globals are updated in place. A missing required name raises
        KeyError before a long simulation starts.
    """
    missing = sorted(_REQUIRED_PARAMETER_NAMES.difference(parameters))
    if missing:
        raise KeyError("Missing simulation parameters: " + ", ".join(missing))

    module_globals = globals()
    for name, value in parameters.items():
        if name.isupper():
            module_globals[name] = value


# ==================== 1. SURFACE HEIGHT AND NORMAL FUNCTIONS ====================
def surface_state_in_region(
    frame, x0, x1, y0, y1, material_id=None, capacity_by_material=None
):
    """提取指定区域的离散表面索引和连续物理高度。

    输入
    ----
    frame : ndarray, shape (nx, ny, nz)
        每个体素中的原子数；只读，不在本函数内修改。
    x0, x1, y0, y1 : int
        半开区间 ``[x0, x1) x [y0, y1)`` 的全局数组索引。

    输出
    ----
    surface_index : int32 ndarray, shape (x1-x0, y1-y0)
        每列最高的 ``frame > ATOM_EPSILON`` 体素索引；整列为空时为 -1。
    surface_height_nm : float64 ndarray, same shape
        考虑最高体素固体分数后的连续顶面高度，空列为 0 nm。满的 z=79
        体素顶面为 80 nm，半满时顶面为 79.5 nm。

    算法
    ----
    沿 z 轴反转占据掩码后用 argmax 找到第一个 True，再换算回原 z 索引。
    因为全 False 列的 argmax 也会返回 0，所以必须用 ``has_material`` 单独把
    这类列改成 -1。连续高度按
    ``(surface_index + top_atom_fraction) * DZ_NM`` 计算。该表达式假定每列
    材料从底部连续堆叠；这与当前逐列从顶向下刻蚀的表示一致。
    """
    sub = frame[x0:x1, y0:y1, :]
    # 从 +z 一侧反向搜索，得到离真空最近的已占据体素。
    occupied = sub > ATOM_EPSILON
    has_material = np.any(occupied, axis=2)
    reverse_index = np.argmax(occupied[:, :, ::-1], axis=2)
    surface_index = frame.shape[2] - 1 - reverse_index
    surface_index[~has_material] = -1
    surface_index = surface_index.astype(np.int32)

    surface_height_nm = np.zeros(surface_index.shape, dtype=np.float64)
    local_x, local_y = np.indices(surface_index.shape)
    valid = surface_index >= 0
    top_atoms = np.zeros(surface_index.shape, dtype=np.float64)
    top_atoms[valid] = sub[
        local_x[valid], local_y[valid], surface_index[valid]
    ]
    if material_id is None:
        top_capacity = np.full(surface_index.shape, ATOM_COUNT_PER_CELL)
    else:
        if capacity_by_material is None:
            raise ValueError(
                "capacity_by_material is required when material_id is provided"
            )
        material_sub = material_id[x0:x1, y0:y1, :]
        top_material = np.full(surface_index.shape, -1, dtype=np.int32)
        top_material[valid] = material_sub[
            local_x[valid], local_y[valid], surface_index[valid]
        ]
        top_capacity = np.ones(surface_index.shape, dtype=np.float64)
        top_capacity[valid] = capacity_by_material[top_material[valid]]
    top_fraction = np.clip(top_atoms / top_capacity, 0.0, 1.0)
    surface_height_nm[valid] = (
        surface_index[valid] + top_fraction[valid]
    ) * DZ_NM
    return surface_index, surface_height_nm


def surface_in_region(frame, x0, x1, y0, y1):
    """兼容旧接口：只返回指定区域的离散表面体素索引。"""
    surface_index, _ = surface_state_in_region(frame, x0, x1, y0, y1)
    return surface_index


def surface_profile(frame):
    """返回整个计算域的表层体素索引。

    输入 ``frame`` 与 :func:`surface_in_region` 相同；输出形状为
    ``frame.shape[:2]``。这是全域调用的便捷封装，不修改 ``frame``。
    """
    return surface_in_region(frame, 0, frame.shape[0], 0, frame.shape[1])


def surface_height_profile(frame):
    """返回整个计算域的连续表面物理高度，单位为 nm。"""
    _, surface_height_nm = surface_state_in_region(
        frame, 0, frame.shape[0], 0, frame.shape[1]
    )
    return surface_height_nm


def surface_normals(surface_height_nm):
    """由二维高度场计算各列指向真空（+z）侧的单位法线。

    输入 ``surface_height_nm`` 是形状 ``(nx, ny)`` 的连续高度图。输出
    ``(normal_x, normal_y, normal_z)``，三个浮点数组形状均与输入相同。
    本函数不原地修改输入。

    先用 NumPy 有限差分计算 ``dz/dx``、``dz/dy``，再按高度图法线
    ``n = (-dz/dx, -dz/dy, 1)`` 归一化。空列暂取竖直向上的法线
    ``(0, 0, 1)``。依赖全局横向网格尺寸 ``DX_NM/DY_NM``。
    """
    valid = surface_height_nm > 0.0
    z_nm = np.where(valid, surface_height_nm, 0.0)
    gradient_x, gradient_y = np.gradient(
        z_nm, DX_NM, DY_NM, axis=(0, 1)
    )
    normal_x = -gradient_x
    normal_y = -gradient_y
    normal_z = np.ones_like(z_nm)
    normal_x[~valid] = 0.0
    normal_y[~valid] = 0.0
    normal_z[~valid] = 1.0
    norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    return normal_x / norm, normal_y / norm, normal_z / norm


def update_surface_normals(surface_height_nm, normals, changed_x, changed_y):
    """原地更新表面变化列及其一阶邻域的法线。

    ``surface_height_nm`` 是全域连续高度；``normals`` 是
    ``surface_normals`` 返回的三个全域数组；``changed_x`` 和
    ``changed_y`` 是等长的一维全局列坐标。返回 ``None``，结果通过原地写入
    ``normals`` 传出，``surface_height_nm`` 不被修改。

    法线使用中心差分，边界使用单边差分。一个高度变化会影响本列及相邻列的
    差分，因此更新 3x3 邻域；集合 ``affected`` 同时负责去重。空变化集直接
    返回。该增量更新避免每个时间步重算全域法线。
    """
    if changed_x.size == 0:
        return

    normal_x, normal_y, normal_z = normals
    affected = set()
    for x, y in zip(changed_x, changed_y):
        x = int(x)
        y = int(y)
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                xx = x + offset_x
                yy = y + offset_y
                if (
                    0 <= xx < surface_height_nm.shape[0]
                    and 0 <= yy < surface_height_nm.shape[1]
                ):
                    affected.add((xx, yy))

    def height_at(x, y):
        return surface_height_nm[x, y]

    for x, y in affected:
        if surface_height_nm[x, y] <= 0.0:
            normal_x[x, y] = 0.0
            normal_y[x, y] = 0.0
            normal_z[x, y] = 1.0
            continue

        if x == 0:
            gradient_x = (height_at(1, y) - height_at(0, y)) / DX_NM
        elif x == surface_height_nm.shape[0] - 1:
            gradient_x = (
                height_at(x, y) - height_at(x - 1, y)
            ) / DX_NM
        else:
            gradient_x = (
                height_at(x + 1, y) - height_at(x - 1, y)
            ) / (2.0 * DX_NM)

        if y == 0:
            gradient_y = (height_at(x, 1) - height_at(x, 0)) / DY_NM
        elif y == surface_height_nm.shape[1] - 1:
            gradient_y = (
                height_at(x, y) - height_at(x, y - 1)
            ) / DY_NM
        else:
            gradient_y = (
                height_at(x, y + 1) - height_at(x, y - 1)
            ) / (2.0 * DY_NM)

        norm = np.sqrt(gradient_x**2 + gradient_y**2 + 1.0)
        normal_x[x, y] = -gradient_x / norm
        normal_y[x, y] = -gradient_y / norm
        normal_z[x, y] = 1.0 / norm


# ==================== 2. REDEPOSITION AND RAY-CASTING FUNCTIONS ====================
def ray_cast_to_surface(start, direction, surface, ignore_cell=None):
    """用二维高度场上的 DDA 追踪射线，返回首次命中的固体表面。

    输入
    ----
    start : array-like, shape (3,)
        射线起点，单位是体素坐标而不是 nm；整数面是体素边界。
    direction : array-like, shape (3,)
        射线方向。调用方传入单位向量；本函数不会再次归一化。
    surface : int ndarray, shape (nx, ny)
        全域表面高度场，只读。
    ignore_cell : tuple(int, int, int) or None
        可选的源体素坐标；命中该体素时忽略，防止射线立刻撞回发射源。

    输出
    ----
    ``None`` 表示射线逸出计算域或未命中；否则返回
    ``(hit_x, hit_y, hit_z, hit_position)``，前三项是命中体素的整数索引，
    ``hit_position`` 是连续体素坐标中的三维交点。

    算法及限制
    ----------
    Amanatides-Woo 风格的二维 DDA 依次遍历射线穿过的 x-y 列；在每列的进入、
    离开参数区间内，检查射线 z 是否低于该列顶面 ``surface + 1``。这是对二维
    高度场的可见性测试，并未逐个遍历三维体素，所以不能描述悬垂或内部孔洞。
    循环上限 ``nx + ny + 2`` 足以覆盖射线在矩形 x-y 域中的无回头穿越。
    """
    position = np.asarray(start, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    size_x, size_y = surface.shape
    index_x, index_y = np.floor(position[:2]).astype(int)
    if not (0 <= index_x < size_x and 0 <= index_y < size_y):
        return None

    if direction[0] > 0.0:
        step_x = 1
        t_max_x = (index_x + 1.0 - position[0]) / direction[0]
    elif direction[0] < 0.0:
        step_x = -1
        t_max_x = (index_x - position[0]) / direction[0]
    else:
        step_x = 0
        t_max_x = np.inf

    if direction[1] > 0.0:
        step_y = 1
        t_max_y = (index_y + 1.0 - position[1]) / direction[1]
    elif direction[1] < 0.0:
        step_y = -1
        t_max_y = (index_y - position[1]) / direction[1]
    else:
        step_y = 0
        t_max_y = np.inf

    t_delta_x = (
        abs(1.0 / direction[0]) if direction[0] != 0.0 else np.inf
    )
    t_delta_y = (
        abs(1.0 / direction[1]) if direction[1] != 0.0 else np.inf
    )
    t_enter = 0.0

    for _ in range(size_x + size_y + 2):
        if not (0 <= index_x < size_x and 0 <= index_y < size_y):
            return None

        t_exit = min(t_max_x, t_max_y)
        z_enter = position[2] + direction[2] * t_enter
        surface_index = int(surface[index_x, index_y])
        hit_t = None

        if surface_index >= 0:
            z_top = surface_index + 1.0
            if 0.0 <= z_enter <= z_top:
                hit_t = t_enter
            elif z_enter > z_top and direction[2] < 0.0:
                candidate_t = (z_top - position[2]) / direction[2]
                if t_enter <= candidate_t <= t_exit:
                    hit_t = candidate_t

        if hit_t is not None:
            hit_position = position + direction * hit_t
            hit_z = min(
                int(np.floor(hit_position[2] - 1.0e-10)), surface_index
            )
            hit_cell = (index_x, index_y, hit_z)
            if ignore_cell is None or hit_cell != ignore_cell:
                return index_x, index_y, hit_z, hit_position

        if t_max_x < t_max_y:
            index_x += step_x
            t_enter = t_max_x
            t_max_x += t_delta_x
        else:
            index_y += step_y
            t_enter = t_max_y
            t_max_y += t_delta_y

        if position[2] + direction[2] * t_enter < 0.0:
            return None

    return None


def orthogonal_basis(normal):
    """为一个单位法线构造两个单位正交切向量。

    输入是形状 ``(3,)`` 的单位向量 ``normal``；输出 ``(tangent_1,
    tangent_2)``。二者与法线两两正交并归一化。本函数不修改输入。参考轴会
    避开与法线近似平行的 x 轴，以防叉乘长度接近零。
    """
    normal = np.asarray(normal, dtype=np.float64)
    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(normal, reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    tangent_1 = np.cross(normal, reference)
    tangent_1 /= np.linalg.norm(tangent_1)
    tangent_2 = np.cross(normal, tangent_1)
    tangent_2 /= np.linalg.norm(tangent_2)
    return tangent_1, tangent_2


@njit
def _ray_cast_to_surface_jit(start_x, start_y, start_z, direction_x,
                             direction_y, direction_z, surface,
                             ignore_x, ignore_y, ignore_z):
    """Numba DDA kernel; a negative x index means that no hit was found."""
    size_x, size_y = surface.shape
    index_x = int(math.floor(start_x))
    index_y = int(math.floor(start_y))
    if index_x < 0 or index_x >= size_x or index_y < 0 or index_y >= size_y:
        return -1, -1, -1, 0.0, 0.0, 0.0

    if direction_x > 0.0:
        step_x = 1
        t_max_x = (index_x + 1.0 - start_x) / direction_x
        t_delta_x = abs(1.0 / direction_x)
    elif direction_x < 0.0:
        step_x = -1
        t_max_x = (index_x - start_x) / direction_x
        t_delta_x = abs(1.0 / direction_x)
    else:
        step_x = 0
        t_max_x = np.inf
        t_delta_x = np.inf

    if direction_y > 0.0:
        step_y = 1
        t_max_y = (index_y + 1.0 - start_y) / direction_y
        t_delta_y = abs(1.0 / direction_y)
    elif direction_y < 0.0:
        step_y = -1
        t_max_y = (index_y - start_y) / direction_y
        t_delta_y = abs(1.0 / direction_y)
    else:
        step_y = 0
        t_max_y = np.inf
        t_delta_y = np.inf

    t_enter = 0.0
    for _ in range(size_x + size_y + 2):
        if index_x < 0 or index_x >= size_x or index_y < 0 or index_y >= size_y:
            return -1, -1, -1, 0.0, 0.0, 0.0

        t_exit = min(t_max_x, t_max_y)
        z_enter = start_z + direction_z * t_enter
        surface_index = int(surface[index_x, index_y])
        has_hit = False
        hit_t = 0.0
        if surface_index >= 0:
            z_top = surface_index + 1.0
            if 0.0 <= z_enter <= z_top:
                has_hit = True
                hit_t = t_enter
            elif z_enter > z_top and direction_z < 0.0:
                candidate_t = (z_top - start_z) / direction_z
                if t_enter <= candidate_t <= t_exit:
                    has_hit = True
                    hit_t = candidate_t

        if has_hit:
            hit_x = start_x + direction_x * hit_t
            hit_y = start_y + direction_y * hit_t
            hit_z_position = start_z + direction_z * hit_t
            hit_z = min(int(math.floor(hit_z_position - 1.0e-10)), surface_index)
            if not (index_x == ignore_x and index_y == ignore_y and hit_z == ignore_z):
                return index_x, index_y, hit_z, hit_x, hit_y, hit_z_position

        if t_max_x < t_max_y:
            index_x += step_x
            t_enter = t_max_x
            t_max_x += t_delta_x
        else:
            index_y += step_y
            t_enter = t_max_y
            t_max_y += t_delta_y
        if start_z + direction_z * t_enter < 0.0:
            return -1, -1, -1, 0.0, 0.0, 0.0
    return -1, -1, -1, 0.0, 0.0, 0.0


@njit
def _contiguous_deposition_z(proposed_z, surface_z):
    """Clamp a redeposition target to the first voxel above a height field."""
    return min(proposed_z, surface_z + 1)


@njit
def _apply_redeposition_jit(frame, material_id, source_x, source_y, source_z,
                            source_material, source_amount,
                            source_normal_x, source_normal_y,
                            source_normal_z, hit_normal_x, hit_normal_y,
                            hit_normal_z, surface, random_cosine,
                            random_azimuth, redepo_rays,
                            capacity_by_material, emission_power_by_material,
                            sticking_coefficient_by_material,
                            sticking_power_by_material, atom_epsilon):
    """Compiled ray sampling, visibility tracing, sticking, and deposition."""
    deposited_mask = np.zeros(surface.shape, dtype=np.bool_)
    total_emitted = 0.0
    total_deposited = 0.0
    max_distance_squared = float(max(frame.shape)) ** 2
    random_index = 0

    for source_index in range(source_amount.size):
        amount = float(source_amount[source_index])
        if amount <= atom_epsilon:
            random_index += redepo_rays
            continue

        index_x = int(source_x[source_index])
        index_y = int(source_y[source_index])
        index_z = int(source_z[source_index])
        source_id = int(source_material[source_index])
        cos_power = emission_power_by_material[source_id]
        normal_x = source_normal_x[source_index]
        normal_y = source_normal_y[source_index]
        normal_z = source_normal_z[source_index]

        # Construct the same orthonormal tangent basis as orthogonal_basis().
        if abs(normal_x) > 0.9:
            tangent_1_x = -normal_z
            tangent_1_y = 0.0
            tangent_1_z = normal_x
        else:
            tangent_1_x = 0.0
            tangent_1_y = normal_z
            tangent_1_z = -normal_y
        tangent_length = math.sqrt(
            tangent_1_x * tangent_1_x
            + tangent_1_y * tangent_1_y
            + tangent_1_z * tangent_1_z
        )
        tangent_1_x /= tangent_length
        tangent_1_y /= tangent_length
        tangent_1_z /= tangent_length
        tangent_2_x = normal_y * tangent_1_z - normal_z * tangent_1_y
        tangent_2_y = normal_z * tangent_1_x - normal_x * tangent_1_z
        tangent_2_z = normal_x * tangent_1_y - normal_y * tangent_1_x

        packet = amount / redepo_rays
        start_x = index_x + 0.5 + normal_x * 1.05
        start_y = index_y + 0.5 + normal_y * 1.05
        start_z = index_z + 0.5 + normal_z * 1.05

        for _ in range(redepo_rays):
            cosine = random_cosine[random_index] ** (1.0 / (cos_power + 1.0))
            sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
            azimuth = 2.0 * math.pi * random_azimuth[random_index]
            random_index += 1
            direction_x = (normal_x * cosine + tangent_1_x * sine * math.cos(azimuth)
                           + tangent_2_x * sine * math.sin(azimuth))
            direction_y = (normal_y * cosine + tangent_1_y * sine * math.cos(azimuth)
                           + tangent_2_y * sine * math.sin(azimuth))
            direction_z = (normal_z * cosine + tangent_1_z * sine * math.cos(azimuth)
                           + tangent_2_z * sine * math.sin(azimuth))
            direction_length = math.sqrt(direction_x * direction_x + direction_y * direction_y + direction_z * direction_z)
            direction_x /= direction_length
            direction_y /= direction_length
            direction_z /= direction_length
            total_emitted += packet

            hit_x, hit_y, _, hit_position_x, hit_position_y, hit_position_z = _ray_cast_to_surface_jit(
                start_x, start_y, start_z, direction_x, direction_y, direction_z,
                surface, index_x, index_y, index_z
            )
            if hit_x < 0:
                continue

            incidence_cosine = max(
                0.0,
                -(direction_x * hit_normal_x[hit_x, hit_y]
                  + direction_y * hit_normal_y[hit_x, hit_y]
                  + direction_z * hit_normal_z[hit_x, hit_y]),
            )
            sticking = (sticking_coefficient_by_material[source_id]
                        * incidence_cosine ** sticking_power_by_material[source_id])
            sticking = min(max(sticking, 0.0), 1.0)
            remaining_deposit = packet * sticking
            deposited_this_packet = 0.0

            while remaining_deposit > atom_epsilon:
                deposit_x = int(math.floor(hit_position_x - direction_x * 0.25))
                deposit_y = int(math.floor(hit_position_y - direction_y * 0.25))
                deposit_z = int(math.floor(hit_position_z - direction_z * 0.25))
                if (deposit_x < 0 or deposit_x >= frame.shape[0]
                        or deposit_y < 0 or deposit_y >= frame.shape[1]):
                    break

                # The solver represents every column as a continuous height
                # field.  Ray/voxel rounding can otherwise choose z > surface+1
                # on a neighbouring column and create a floating redeposited
                # voxel.  Clamp a new coating to the first voxel directly above
                # the current column surface; partially filled surface voxels
                # remain eligible through their existing z index below.
                deposit_z = _contiguous_deposition_z(
                    deposit_z, int(surface[deposit_x, deposit_y])
                )
                if deposit_z < 0 or deposit_z >= frame.shape[2]:
                    break

                deposit_id = int(material_id[deposit_x, deposit_y, deposit_z])
                if deposit_id < 0:
                    # A new coating voxel inherits the sputtered source material.
                    material_id[deposit_x, deposit_y, deposit_z] = source_id
                    deposit_id = source_id
                if deposit_id == source_id:
                    capacity = (capacity_by_material[source_id]
                                - float(frame[deposit_x, deposit_y, deposit_z]))
                else:
                    # The compact two-array representation cannot mix materials
                    # inside one voxel. Continue outward to the next free voxel.
                    capacity = 0.0
                if capacity > atom_epsilon:
                    added = min(remaining_deposit, capacity)
                    frame[deposit_x, deposit_y, deposit_z] += added
                    if deposit_z > surface[deposit_x, deposit_y]:
                        surface[deposit_x, deposit_y] = deposit_z
                    deposited_mask[deposit_x, deposit_y] = True
                    remaining_deposit -= added
                    deposited_this_packet += added

                hit_position_x -= direction_x * 0.75
                hit_position_y -= direction_y * 0.75
                hit_position_z -= direction_z * 0.75
                distance_squared = ((hit_position_x - start_x) ** 2
                                    + (hit_position_y - start_y) ** 2
                                    + (hit_position_z - start_z) ** 2)
                if distance_squared > max_distance_squared:
                    break
            total_deposited += deposited_this_packet
    return total_emitted, total_deposited, deposited_mask


def apply_redeposition(
    frame, material_id, sources, source_normals, hit_normals, surface,
    material_properties,
):
    """按 Monte Carlo 弹道射线模型把部分溅射原子沉积到可见表面。

    输入
    ----
    frame : float ndarray, shape (NX, NY, NZ)
        原子数场；本函数会把黏附原子原地加回其中。
    sources : tuple(x, y, z, material_id, amount) or None
        五个等长一维数组，记录溅射源位置、源材料及原子数。
    source_normals : tuple(nx, ny, nz)
        与 sources 等长的一维发射法线分量，取自刻蚀前表面。
    hit_normals : tuple(nx, ny, nz)
        三个全域二维法线数组，用于计算命中表面的入射余弦。
    surface : int ndarray, shape (NX, NY)
        全域高度场；沉积成功时会原地提高相应列的表面索引。

    输出
    ----
    ``(total_emitted, total_deposited, deposited_x, deposited_y)``：前两项是本步
    参与射线抽样的原子总数和实际填回的原子总数；后两项是发生沉积的全局列
    坐标一维数组，供调用方增量更新法线。函数同时原地修改 ``frame/surface``。

    算法
    ----
    每个源的原子量等分为 ``REDEPO_RAYS`` 个 packet。正式驱动默认使用 32 条
    射线；方向服从源法线半球上的
    ``p(beta) ∝ cos(beta)^n`` 分布，n 由源材料和电压决定；DDA 返回首次
    命中表面。黏附率参数也由源材料选择。黏附 packet
    从交点真空侧开始填充，满体素后沿射线来向继续寻找可用容量。逸出域、未黏附
    或无法放入的原子不回到 frame。随机数来自独立的 ``REDEPOSITION_RNG``。
    """
    empty_indices = np.empty(0, dtype=int)
    if sources is None or REDEPO_RAYS <= 0:
        return 0.0, 0.0, empty_indices, empty_indices.copy()

    source_normal_x, source_normal_y, source_normal_z = source_normals
    hit_normal_x, hit_normal_y, hit_normal_z = hit_normals
    source_x, source_y, source_z, source_material, source_amount = sources
    # Random draws remain outside the JIT kernel, so the driver owns the RNG
    # stream exactly as before and can reproduce a run from RANDOM_SEED.
    # Stratify the cosine variate within each source: this preserves the exact
    # p(cos(beta)) distribution while reducing packet-to-packet variance versus
    # drawing all packets independently (particularly useful at 32 rays/source).
    n_packets = source_amount.size * REDEPO_RAYS
    random_pairs = REDEPOSITION_RNG.random((n_packets, 2))
    if REDEPO_RAYS > 1:
        packet_in_source = np.arange(n_packets, dtype=np.float64) % REDEPO_RAYS
        random_pairs[:, 0] = (
            packet_in_source + random_pairs[:, 0]
        ) / REDEPO_RAYS
    total_emitted, total_deposited, deposited_mask = _apply_redeposition_jit(
        frame, material_id, source_x, source_y, source_z,
        source_material, source_amount,
        source_normal_x, source_normal_y, source_normal_z,
        hit_normal_x, hit_normal_y, hit_normal_z, surface,
        random_pairs[:, 0], random_pairs[:, 1], REDEPO_RAYS,
        material_properties["capacity"],
        material_properties["emission_power"],
        material_properties["sticking_coefficient"],
        material_properties["sticking_power"], ATOM_EPSILON,
    )
    deposited_x, deposited_y = np.nonzero(deposited_mask)
    return total_emitted, total_deposited, deposited_x, deposited_y


# ==================== 3. SCAN-PATH AND SPUTTERING FUNCTIONS ====================
def scan_point(point_index):
    """把单个 Pass 内的一维点序号映射为 serpentine 作用点。

    输入 ``point_index`` 是 ``[0, N_POINTS_PER_PASS)`` 内的整数。输出
    ``(beam_x_nm, beam_y_nm, row_index)``：前两项是束流中心的 nm 坐标，
    后一项是从 0 开始的扫描行号。偶数行 x 从小到大，奇数行反向。
    越界输入抛出 ``IndexError``；函数只读取全局扫描参数，无副作用。
    """
    if not 0 <= point_index < N_POINTS_PER_PASS:
        raise IndexError("point_index is outside one Pass")

    row_index, point_in_row = divmod(point_index, N_POINTS_X)
    if row_index % 2 == 0:
        x_point_index = point_in_row
    else:
        x_point_index = N_POINTS_X - 1 - point_in_row

    beam_x = SCAN_X_START_NM + x_point_index * PITCH_X_NM
    beam_y = SCAN_Y_START_NM + row_index * PITCH_Y_NM
    return beam_x, beam_y, row_index


def beam_state_for_step(step):
    """返回整数时间步对应的扫描或 beam-off 返回状态。

    每个周期先按固定点序完成一个 Pass，再从末点沿直线匀速返回初始点。
    返回期间 ``beam_on`` 为 False，且 ``point_index`` 为 -1。

    输入 ``step`` 会转换为非负整数。输出六元组
    ``(beam_x_nm, beam_y_nm, row_index, beam_on, pass_index, point_index)``。
    扫描时行号、点号有效；返回时二者均为 -1。超过完整规划步数时返回“最后
    一个 Pass 已结束、束流关闭且位于初始点”的终态。本函数是纯状态映射，
    不修改模拟数据。

    在扫描区用 ``step // STEPS_PER_DWELL`` 保证一个作用点连续驻留固定步数；
    在返回区用速度乘已经历时间得到直线插值比例。``IB_RETURN_STEPS`` 由 ceil
    得到，所以最后一个离散返回步可能不足以精确落在初始点，但下一周期的首个
    扫描状态会直接位于初始点。
    """
    full_simulation_steps = FULL_N_STEPS
    step = max(0, int(step))

    # 模拟结束时束流已经完成最后一次关闭返回并停在初始位置。
    if step >= full_simulation_steps:
        return (
            SCAN_INITIAL_X_NM,
            SCAN_INITIAL_Y_NM,
            -1,
            False,
            N_PASS - 1,
            -1,
        )

    pass_index, step_in_period = divmod(step, PASS_PERIOD_STEPS)
    if step_in_period < SCAN_STEPS_PER_PASS:
        point_index = step_in_period // STEPS_PER_DWELL
        beam_x, beam_y, row_index = scan_point(point_index)
        return beam_x, beam_y, row_index, True, pass_index, point_index

    return_step = step_in_period - SCAN_STEPS_PER_PASS
    return_elapsed_s = return_step * DT_S
    return_distance_nm = min(
        IB_RETURN_VELOCITY_NM_S * return_elapsed_s,
        IB_RETURN_DISTANCE_NM,
    )
    if IB_RETURN_DISTANCE_NM > 0.0:
        return_fraction = return_distance_nm / IB_RETURN_DISTANCE_NM
    else:
        return_fraction = 1.0
    beam_x = SCAN_FINAL_X_NM + return_fraction * (
        SCAN_INITIAL_X_NM - SCAN_FINAL_X_NM
    )
    beam_y = SCAN_FINAL_Y_NM + return_fraction * (
        SCAN_INITIAL_Y_NM - SCAN_FINAL_Y_NM
    )
    return beam_x, beam_y, -1, False, pass_index, -1


def beam_state(t_s):
    """兼容按物理时间查询束流状态的接口。

    每个 Pass 都从左下角初始点开始，并使用相同的逐行往复点序；Pass
    之间包含计时但不分配离子的束流关闭返回阶段。

    输入是物理时间（s），输出简化四元组
    ``(beam_x_nm, beam_y_nm, row_index, beam_on)``。它先向下映射到包含该时刻
    的整数步，再调用 ``beam_state_for_step``；用于兼容旧的按时间查询接口。
    """
    step = int(np.floor(t_s / DT_S + 1.0e-12))
    beam_x, beam_y, row_index, beam_on, _, _ = beam_state_for_step(step)
    return beam_x, beam_y, row_index, beam_on


# ==================== 4. MATERIAL-REMOVAL FUNCTIONS ====================
def remove_from_columns(
    frame, x0, y0, surface, requested, track_sources=False
):
    """从局部区域的各列顶部向下移除请求量，并可记录溅射源。

    输入
    ----
    frame : float ndarray, shape (NX, NY, NZ)
        全域原子数场；被刻蚀体素会被原地扣减。
    x0, y0 : int
        局部 ``requested/surface`` 左下角在 frame 中的全局索引。
    surface : int ndarray, shape (width_x, width_y)
        刻蚀前局部表面索引，只读。
    requested : ndarray, same shape as surface
        每列请求移除的原子数，通常等于局部离子数乘溅射产额。
    track_sources : bool
        True 时逐层记录实际扣除量，供再沉积使用。

    输出
    ----
    ``(removed, remaining, sources)``。``removed`` 和 ``remaining`` 均与请求矩阵
    同形，分别是实际移除量和因列已刻空而未满足的量；``sources`` 为
    ``(global_x, global_y, z, amount)`` 四个一维数组，未跟踪或无移除时为 None。

    算法
    ----
    向量化地同时处理所有活动列，并强制满足逐层刻蚀不变量：每列每轮只访问
    当前最高体素；若请求量小于该层存量，只减少该层且不接触下一层；若请求量
    大于该层存量，先将该层清零，再把未满足量带到 ``z-1``。因此一次时间步可
    跨越多层，但绝不会越过仍有材料的上层。最多循环 ``NZ + 1`` 次。

    小于 ``ATOM_EPSILON`` 的体素残量作为数值舍入量并入本次实际移除量，从而
    在把体素归零时保持原子账本一致。函数不直接更新调用方的 ``surface``，
    主循环须在扣除后重新提取高度。
    """
    width_x, width_y = requested.shape
    local_x, local_y = np.indices((width_x, width_y))
    global_x = local_x + x0
    global_y = local_y + y0

    remaining = np.where(surface >= 0, requested, 0.0).astype(np.float64)
    removed = np.zeros_like(remaining)
    current_z = surface.astype(np.int32).copy()
    source_x_parts = []
    source_y_parts = []
    source_z_parts = []
    source_amount_parts = []

    # 每次循环严格只处理每列的当前最高层。
    for _ in range(frame.shape[2] + 1):
        active = (remaining > ATOM_EPSILON) & (current_z >= 0)
        if not np.any(active):
            break

        rows, cols = np.nonzero(active)
        gx = global_x[rows, cols]
        gy = global_y[rows, cols]
        gz = current_z[rows, cols]
        available = frame[gx, gy, gz].astype(np.float64)
        request_here = remaining[rows, cols]
        taken = np.minimum(request_here, available)
        updated = available - taken

        # 若只剩数值容差内的原子，将这部分也计入 taken 后再归零。这样 frame
        # 中消失的原子全部进入 removed/source 账本，不会凭空丢失。
        numerical_residue = (updated > 0.0) & (updated <= ATOM_EPSILON)
        if np.any(numerical_residue):
            taken[numerical_residue] += updated[numerical_residue]
            updated[numerical_residue] = 0.0

        if track_sources:
            emitted = taken > ATOM_EPSILON
            if np.any(emitted):
                source_x_parts.append(gx[emitted].copy())
                source_y_parts.append(gy[emitted].copy())
                source_z_parts.append(gz[emitted].copy())
                source_amount_parts.append(taken[emitted].copy())
        frame[gx, gy, gz] = updated
        removed[rows, cols] += taken
        remaining_here = np.maximum(request_here - taken, 0.0)
        remaining[rows, cols] = remaining_here

        # 只有“当前层已空”且“请求仍未满足”的列才允许进入下一层。
        layer_is_empty = updated <= ATOM_EPSILON
        carry_to_lower_layer = (
            layer_is_empty & (remaining_here > ATOM_EPSILON)
        )
        if np.any(carry_to_lower_layer):
            current_z[
                rows[carry_to_lower_layer], cols[carry_to_lower_layer]
            ] -= 1

        # 理论上 remaining>0 必然意味着 available 已全部取完；此检查用于防止
        # 后续修改破坏“上层未空不得刻蚀下层”的核心不变量。
        blocked_carry = (
            (remaining_here > ATOM_EPSILON) & ~layer_is_empty
        )
        if np.any(blocked_carry):
            raise RuntimeError(
                "Removal request attempted to bypass a non-empty upper voxel"
            )

    remaining[remaining <= ATOM_EPSILON] = 0.0
    if source_amount_parts:
        sources = (
            np.concatenate(source_x_parts),
            np.concatenate(source_y_parts),
            np.concatenate(source_z_parts),
            np.concatenate(source_amount_parts),
        )
    else:
        sources = None
    return removed, remaining, sources


def sputter_multimaterial_columns(
    frame,
    material_id,
    x0,
    y0,
    surface,
    incident_ions,
    incidence_angle,
    material_properties,
    track_sources=False,
):
    """Remove material while carrying unused *ion dose* across interfaces.

    Unlike :func:`remove_from_columns`, this routine does not carry a requested
    number of atoms into the next layer.  It consumes the incident-ion share
    needed to empty the current material, then recalculates the sputter yield
    for the material exposed below.  This is the required behavior when two
    materials have different atoms/ion yields or voxel atom capacities.

    Returns ``(removed_atoms, remaining_ions, sources, requested_atoms)``.
    A source tuple contains ``x, y, z, material_id, amount`` arrays.
    """
    width_x, width_y = incident_ions.shape
    local_x, local_y = np.indices((width_x, width_y))
    global_x = local_x + x0
    global_y = local_y + y0
    remaining_ions = np.where(surface >= 0, incident_ions, 0.0).astype(np.float64)
    removed = np.zeros_like(remaining_ions)
    requested_atoms = np.zeros_like(remaining_ions)
    request_initialized = np.zeros(remaining_ions.shape, dtype=bool)
    current_z = surface.astype(np.int32).copy()
    source_parts = [[], [], [], [], []]

    # These dense arrays are indexed by material_id.  Keeping them outside the
    # loop avoids dictionary lookups for every surface column and is also what
    # makes the function compatible with the Numba redeposition kernel.
    yield0 = material_properties["yield0"]
    sigmund_a = material_properties["sigmund_a"]
    sigmund_alpha = material_properties["sigmund_alpha"]
    sigmund_beta = material_properties["sigmund_beta"]

    for _ in range(frame.shape[2] + 1):
        active = (remaining_ions > 1.0e-15) & (current_z >= 0)
        if not np.any(active):
            break
        rows, cols = np.nonzero(active)
        gx = global_x[rows, cols]
        gy = global_y[rows, cols]
        gz = current_z[rows, cols]
        ids = material_id[gx, gy, gz].astype(np.int32)
        if np.any(ids < 0):
            bad = int(np.flatnonzero(ids < 0)[0])
            bx, by, bz = int(gx[bad]), int(gy[bad]), int(gz[bad])
            raise RuntimeError(
                "surface cache points to a vacuum voxel at "
                f"({bx}, {by}, {bz}); atoms={float(frame[bx, by, bz]):.9g}, "
                f"cached_surface={int(surface[rows[bad], cols[bad]])}"
            )

        angle = incidence_angle[rows, cols]
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        alpha = sigmund_alpha[ids]
        beta = sigmund_beta[ids]
        effective_width = np.sqrt(
            alpha**2 * cos_angle**2 + beta**2 * sin_angle**2
        )
        layer_yield = (
            yield0[ids]
            * alpha / effective_width
            * cos_angle
            * np.exp(
                sigmund_a[ids] ** 2 / (2.0 * alpha**2)
                * (1.0 - alpha**2 / effective_width**2 * cos_angle**2)
            )
        )
        layer_yield = np.maximum(layer_yield, 0.0)
        if np.any(layer_yield <= 0.0):
            raise RuntimeError("an occupied material has zero sputter yield")

        available = frame[gx, gy, gz].astype(np.float64)
        # ``possible`` is expressed in target atoms, while remaining_ions is
        # deliberately kept in incident ions so the next material gets a new Y.
        possible = remaining_ions[rows, cols] * layer_yield
        first_request = ~request_initialized[rows, cols]
        if np.any(first_request):
            requested_atoms[rows[first_request], cols[first_request]] = possible[
                first_request
            ]
            request_initialized[rows[first_request], cols[first_request]] = True
        taken = np.minimum(possible, available)
        ions_consumed = taken / layer_yield
        updated = available - taken
        numerical_residue = (updated > 0.0) & (updated <= ATOM_EPSILON)
        if np.any(numerical_residue):
            taken[numerical_residue] += updated[numerical_residue]
            ions_consumed[numerical_residue] = (
                taken[numerical_residue] / layer_yield[numerical_residue]
            )
            updated[numerical_residue] = 0.0

        if track_sources:
            emitted = taken > ATOM_EPSILON
            if np.any(emitted):
                source_parts[0].append(gx[emitted].copy())
                source_parts[1].append(gy[emitted].copy())
                source_parts[2].append(gz[emitted].copy())
                source_parts[3].append(ids[emitted].copy())
                source_parts[4].append(taken[emitted].copy())

        frame[gx, gy, gz] = updated
        removed[rows, cols] += taken
        remaining_here = np.maximum(
            remaining_ions[rows, cols] - ions_consumed, 0.0
        )
        remaining_ions[rows, cols] = remaining_here
        layer_empty = updated <= ATOM_EPSILON
        if np.any(layer_empty):
            # An emptied voxel becomes vacuum immediately; the next loop pass
            # can expose the material below it without skipping a layer.
            material_id[gx[layer_empty], gy[layer_empty], gz[layer_empty]] = -1
        descend = layer_empty & (remaining_here > 1.0e-15)
        if np.any(descend):
            current_z[rows[descend], cols[descend]] -= 1

        # A partially occupied voxel must have consumed all ion dose.  If not,
        # carrying it below would violate the top-down removal invariant.
        blocked = (~layer_empty) & (remaining_here > 1.0e-12)
        if np.any(blocked):
            raise RuntimeError("ion dose attempted to bypass a non-empty voxel")

    remaining_ions[remaining_ions <= 1.0e-15] = 0.0
    if source_parts[4]:
        sources = tuple(np.concatenate(parts) for parts in source_parts)
    else:
        sources = None
    return removed, remaining_ions, sources, requested_atoms


# ==================== 5. OUTPUT FUNCTIONS ====================
def _data_color_limits(values):
    """Return finite data limits for an image colorbar.

    The original plots used the initial 80 nm surface as a fixed upper limit.
    Redeposition can build material above that level, so fixed limits silently
    clipped the late-time snapshots.  Keep the limits data-driven and protect
    the degenerate constant-field case for Matplotlib.
    """
    if str(COLORBAR_MODE).strip().lower() == "global":
        # The computational domain is the guaranteed global physical range;
        # this includes material redeposited above the initial surface.
        return 0.0, float(DOMAIN_HEIGHT_Z_NM)

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    lower = float(finite.min())
    upper = float(finite.max())
    if upper <= lower:
        pad = max(1.0e-12, abs(lower) * 1.0e-6)
        lower -= pad
        upper += pad
    return lower, upper


def save_snapshot(
    surface_height_nm,
    t_s,
    beam_x_nm,
    beam_y_nm,
    beam_on,
    pass_index,
    point_index,
    filename,
):
    """把连续表面高度和当前束流状态保存为 PNG。

    ``surface_height_nm`` 是全域二维连续顶面高度；``t_s`` 单位为秒；
    束流 x/y 坐标单位为 nm；
    ``beam_on``、``pass_index``、``point_index`` 控制标记颜色和标题；
    ``filename`` 是相对 ``OUTPUT_DIR`` 的文件名。函数返回 ``None``，副作用是
    创建并关闭 Matplotlib figure、写入文件。颜色显示连续表面高度，色标范围
    根据当前有限数据的实际最小值和最大值自动调整，因此再沉积超过初始表面
    后仍不会被截断。
    """
    plot_height_min_nm, plot_height_max_nm = _data_color_limits(
        surface_height_nm
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(
        surface_height_nm.T,
        origin="lower",
        cmap="viridis",
        vmin=plot_height_min_nm,
        vmax=plot_height_max_nm,
        aspect="equal",
        extent=[0, NX * DX_NM, 0, NY * DY_NM],
    )
    fig.colorbar(image, ax=ax, label="surface height (nm)")
    marker_color = "red" if beam_on else "gray"
    beam_label = "ON / point dwell" if beam_on else "OFF / return"
    if beam_on:
        phase_label = f"point {point_index + 1}/{N_POINTS_PER_PASS}"
    else:
        phase_label = "beam-off return"
    ax.plot(
        [beam_x_nm],
        [beam_y_nm],
        marker="x",
        color=marker_color,
        markersize=6,
    )
    ax.set_title(
        f"Pass {pass_index + 1}/{N_PASS}; "
        f"{phase_label}; "
        f"t = {t_s * 1e3:.3f} ms\n"
        f"beam {beam_label}; ({beam_x_nm:.1f}, {beam_y_nm:.1f}) nm"
    )
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_frame_vtk(
    filename, frame, material_id=None, capacity_by_material=None
):
    """把三维原子数场写为 VTK ImageData 单元数据。

    输入 ``filename`` 是目标路径，``frame`` 是 ``[x,y,z]`` 排列的原子数数组；
    返回 ``None``。输出包含 ``AtomCount`` 和归一化并截断到 [0,1] 的
    ``SolidFraction`` 两个 float cell arrays，后者被设为默认 scalars。
    ``vtkImageData`` 的点维数比体素维数各多 1。写出前转置为 ``[z,y,x]``，
    使 C-order 展平结果符合 VTK 的 x-fastest 顺序，文件使用 legacy binary VTK。
    """
    frame_nx, frame_ny, frame_nz = frame.shape
    image = vtk.vtkImageData()
    image.SetDimensions(frame_nx + 1, frame_ny + 1, frame_nz + 1)
    image.SetSpacing(DX_NM, DY_NM, DZ_NM)
    image.SetOrigin(0.0, 0.0, 0.0)

    # VTK 单元数据要求 x 索引变化最快，因此先转为 [z, y, x]。
    atoms_flat = np.ascontiguousarray(frame.transpose(2, 1, 0)).ravel()
    atom_array = numpy_to_vtk(atoms_flat, deep=True, array_type=vtk.VTK_FLOAT)
    atom_array.SetName("AtomCount")

    if material_id is None:
        solid_fraction = np.clip(frame / ATOM_COUNT_PER_CELL, 0.0, 1.0)
    else:
        if capacity_by_material is None:
            raise ValueError("capacity_by_material is required with material_id")
        capacity_field = np.ones(frame.shape, dtype=np.float32)
        occupied = material_id >= 0
        capacity_field[occupied] = capacity_by_material[material_id[occupied]]
        solid_fraction = np.zeros(frame.shape, dtype=np.float32)
        solid_fraction[occupied] = np.clip(
            frame[occupied] / capacity_field[occupied], 0.0, 1.0
        )
    fraction_flat = np.ascontiguousarray(
        solid_fraction.transpose(2, 1, 0)
    ).ravel()
    fraction_array = numpy_to_vtk(
        fraction_flat, deep=True, array_type=vtk.VTK_FLOAT
    )
    fraction_array.SetName("SolidFraction")

    image.GetCellData().SetScalars(fraction_array)
    image.GetCellData().AddArray(atom_array)
    if material_id is not None:
        material_flat = np.ascontiguousarray(
            material_id.transpose(2, 1, 0)
        ).ravel()
        material_array = numpy_to_vtk(
            material_flat, deep=True, array_type=vtk.VTK_SHORT
        )
        material_array.SetName("MaterialId")
        image.GetCellData().AddArray(material_array)

    writer = vtk.vtkDataSetWriter()
    writer.SetFileName(filename)
    writer.SetInputData(image)
    writer.SetFileTypeToBinary()
    writer.Write()
    print(
        f"VTK written: {filename} "
        f"({frame_nx}x{frame_ny}x{frame_nz} cells)"
    )


# ==================== 6. MAIN SIMULATION FUNCTION ====================
def main(initial_state=None):
    """初始化样品、执行完整扫描，并保存结果与守恒统计。

    本函数无输入、无显式返回值；所有配置来自模块级常量和运行环境变量。
    主要副作用是分配并原地演化 ``frame``，在 ``OUTPUT_DIR`` 写 PNG、VTK、NPY
    文件，并在标准输出打印进度和汇总。其计算流程可拆成：

    1. 建立均匀 Mo 平板、表面高度场及可选法线场；
    2. 对每个离散时间步查询束流状态，beam-off 时仅推进时间；
    3. beam-on 时完成局部表面/坡度、高斯剂量、Poisson 离子数、角度产额、
       逐列刻蚀，以及可选再沉积；
    4. 周期性输出快照，结束后保存体数据、表面数组和三维图；
    5. 计算刻蚀深度及原子账本。原子账本中 ``total_sputtered`` 是离开 frame
       前的总量，再沉积量作为回流项加入，因此理论关系为
       ``initial + redeposited - final - sputtered = 0``（允许浮点误差）。
    """
    # ---- 11.1 初始化体数据及几何缓存 ----
    # The legacy no-state branch remains available for compatibility; normal
    # structure-aware runs pass a validated StructureState from simu_run.py.
    if initial_state is None:
        frame = np.zeros((NX, NY, NZ), dtype=np.float32)
        frame[:, :, :INITIAL_SOLID_LAYERS] = ATOM_COUNT_PER_CELL
        material_id = np.full((NX, NY, NZ), -1, dtype=np.int16)
        material_id[:, :, :INITIAL_SOLID_LAYERS] = 0
        material_properties = {
            "capacity": np.asarray([ATOM_COUNT_PER_CELL], dtype=np.float64),
            "yield0": np.asarray([Y0], dtype=np.float64),
            "emission_power": np.asarray([REDEPO_COS_POWER], dtype=np.float64),
            "sticking_coefficient": np.asarray(
                [STICKING_COEFFICIENT], dtype=np.float64
            ),
            "sticking_power": np.asarray([STICKING_POWER], dtype=np.float64),
            "sigmund_a": np.asarray([A_SIGMUND], dtype=np.float64),
            "sigmund_alpha": np.asarray([ALPHA_SIGMUND], dtype=np.float64),
            "sigmund_beta": np.asarray([BETA_SIGMUND], dtype=np.float64),
        }
    else:
        # Copy the state because the solver mutates both arrays in place.
        initial_state.validate(require_height_field=True)
        if initial_state.shape != (NX, NY, NZ):
            raise ValueError(
                f"Initial structure shape {initial_state.shape} does not match "
                f"configured grid {(NX, NY, NZ)}"
            )
        if not np.allclose(
            initial_state.spacing_nm, (DX_NM, DY_NM, DZ_NM)
        ):
            raise ValueError("Initial structure spacing does not match DX/DY/DZ")
        frame = initial_state.atom_count.copy()
        material_id = initial_state.material_id.copy()
        material_properties = initial_state.materials.property_arrays(
            VOLTAGE_KV, DX_NM * DY_NM * DZ_NM
        )
    initial_atoms = float(frame.sum(dtype=np.float64))
    initial_capacity_field = material_properties["capacity"][
        np.maximum(material_id, 0)
    ]
    initial_solid_volume_nm3 = float(
        np.where(
            material_id >= 0,
            frame / initial_capacity_field * (DX_NM * DY_NM * DZ_NM),
            0.0,
        ).sum(dtype=np.float64)
    )
    surface, surface_height_nm = surface_state_in_region(
        frame, 0, NX, 0, NY, material_id,
        material_properties["capacity"],
    )
    initial_surface_height_nm = surface_height_nm.copy()
    nonempty_initial_height = initial_surface_height_nm[
        initial_surface_height_nm > 0.0
    ]
    initial_height_min_nm = (
        float(nonempty_initial_height.min())
        if nonempty_initial_height.size else 0.0
    )
    initial_height_max_nm = (
        float(nonempty_initial_height.max())
        if nonempty_initial_height.size else 0.0
    )
    normals = (
        surface_normals(surface_height_nm)
        if ENABLE_REDEPOSTION
        else None
    )

    # ---- 11.2 累积统计量 ----
    # expected/sample 描述离子剂量；requested/actual/unfulfilled 描述溅射请求账本；
    # emitted/deposited 描述再沉积射线账本。
    total_expected_ions = 0.0
    total_sampled_ions = 0.0
    total_requested_sputter = 0.0
    total_sputtered = 0.0
    total_sputtered_volume_nm3 = 0.0
    total_unfulfilled_sputter = 0.0
    total_emitted_for_redeposition = 0.0
    total_redeposited = 0.0
    beam_on_steps = 0
    snapshot_count = 0

    # t=0 的初始快照与完整三维初态，便于和终态逐项比较。
    (
        initial_x,
        initial_y,
        _,
        initial_on,
        initial_pass,
        initial_point,
    ) = beam_state_for_step(0)
    initial_snapshot_stem = f"snap_{snapshot_count:04d}"
    save_snapshot(
        surface_height_nm,
        0.0,
        initial_x,
        initial_y,
        initial_on,
        initial_pass,
        initial_point,
        initial_snapshot_stem + ".png",
    )
    write_frame_vtk(
        os.path.join(OUTPUT_DIR, initial_snapshot_stem + ".vtk"), frame,
        material_id, material_properties["capacity"],
    )
    snapshot_count += 1
    write_frame_vtk(
        os.path.join(OUTPUT_DIR, "frame_initial.vtk"), frame,
        material_id, material_properties["capacity"],
    )

    start_time = time.time()
    progress_interval = max(1, N_STEPS // 20)

    # ---- 11.3 时间推进主循环 ----
    for step in range(N_STEPS):
        t_s = step * DT_S           # 刻蚀时间
        (
            beam_x_nm,
            beam_y_nm,
            _,
            beam_on,
            _,
            _,
        ) = beam_state_for_step(step)

        # 扫描点驻留时开启束流；Pass 后的匀速返回阶段只推进时间，不产生离子。
        if beam_on:
            beam_on_steps += 1
            # 离子束中心所在格点
            beam_x = beam_x_nm / DX_NM
            beam_y = beam_y_nm / DY_NM
            radius_x = int(
                np.ceil(BEAM_EFFECT_RADIUS_NM / DX_NM)
            )
            radius_y = int(
                np.ceil(BEAM_EFFECT_RADIUS_NM / DY_NM)
            )
            x0 = max(0, int(beam_x) - radius_x)
            x1 = min(NX, int(beam_x) + radius_x + 1)
            y0 = max(0, int(beam_y) - radius_y)
            y1 = min(NY, int(beam_y) + radius_y + 1)

            # 束斑包围盒只覆盖可能有非零高斯权重的列；再向外扩一格，为坡度
            # 中心差分保留邻点。靠计算域边界处由 np.gradient 使用单边差分。
            padded_x0 = max(0, x0 - 1)
            padded_x1 = min(NX, x1 + 1)
            padded_y0 = max(0, y0 - 1)
            padded_y1 = min(NY, y1 + 1)
            # 一格外扩区域为中心差分提供邻点。
            padded_surface, padded_height_nm = surface_state_in_region(
                frame, padded_x0, padded_x1, padded_y0, padded_y1,
                material_id, material_properties["capacity"],
            )
            gradient_x, gradient_y = np.gradient(
                padded_height_nm,
                DX_NM,
                DY_NM,
                axis=(0, 1),
            )
            core_x0 = x0 - padded_x0
            core_x1 = core_x0 + x1 - x0
            core_y0 = y0 - padded_y0
            core_y1 = core_y0 + y1 - y0
            surface_core = padded_surface[
                core_x0:core_x1, core_y0:core_y1
            ]
            slope_x = gradient_x[core_x0:core_x1, core_y0:core_y1]
            slope_y = gradient_y[core_x0:core_x1, core_y0:core_y1]
            # 垂直向下的束流与高度场法线的夹角满足
            # tan(theta)=sqrt((dz/dx)^2+(dz/dy)^2)，故 theta=atan(|grad z|)。
            incidence_angle = np.arctan(np.hypot(slope_x, slope_y))

            cell_x_nm = (np.arange(x0, x1) + 0.5) * DX_NM
            cell_y_nm = (np.arange(y0, y1) + 0.5) * DY_NM
            # 在独立定义的作用半径内分配高斯束流；相邻作用点的束斑
            # 对同一表面列重复贡献，从而自然形成空间重叠。
            delta_x_nm = cell_x_nm[:, None] - beam_x_nm
            delta_y_nm = cell_y_nm[None, :] - beam_y_nm
            radial_distance_squared_nm2 = delta_x_nm**2 + delta_y_nm**2
            gaussian = np.exp(
                -0.5
                * radial_distance_squared_nm2
                / BEAM_SIGMA_NM**2
            )
            gaussian[
                radial_distance_squared_nm2 > BEAM_EFFECT_RADIUS_NM**2
            ] = 0.0
            gaussian_sum = gaussian.sum()
            if gaussian_sum <= 0.0:
                raise RuntimeError("No grid cells lie inside the beam effect radius")
            # 截断后的权重在当前包围盒内重新归一化为 1。因此每个 beam-on 步的
            # 总期望离子数恒为 I/q*DT；高斯只负责把这份剂量分配到各表面列。
            gaussian /= gaussian_sum
            expected_ions = ION_RATE_S * gaussian * DT_S
            # 独立 Poisson 抽样使每列离子数为整数且总数有散粒噪声；关闭时直接
            # 使用期望值，可用于无统计涨落的确定性对照。
            if USE_POISSON_ION_ARRIVAL:
                ions = ION_RNG.poisson(expected_ions)
            else:
                ions = expected_ions

            # Carry unused incident-ion dose across material interfaces.  The
            # newly exposed material therefore uses its own yield/capacity.
            removed, unfulfilled_ions, sources, requested_removal = (
                sputter_multimaterial_columns(
                frame,
                material_id,
                x0,
                y0,
                surface_core,
                ions,
                incidence_angle,
                material_properties,
                track_sources=True,
                )
            )

            # Each removed voxel is a separate source.  If one dwell crosses a
            # layer boundary, source material and z are retained independently;
            # source normals still come from the pre-etch surface.
            local_source_x, local_source_y = np.nonzero(
                removed > ATOM_EPSILON
            )
            if sources is not None:
                source_x, source_y, _, _, _ = sources
                source_local_x = source_x - x0
                source_local_y = source_y - y0
                source_normal_norm = np.sqrt(
                    slope_x[source_local_x, source_local_y] ** 2
                    + slope_y[source_local_x, source_local_y] ** 2
                    + 1.0
                )
                source_normals = (
                    -slope_x[source_local_x, source_local_y]
                    / source_normal_norm,
                    -slope_y[source_local_x, source_local_y]
                    / source_normal_norm,
                    1.0 / source_normal_norm,
                )
            else:
                source_normals = None

            total_expected_ions += float(expected_ions.sum())
            total_sampled_ions += float(np.asarray(ions).sum())
            total_requested_sputter += float(requested_removal.sum())
            total_sputtered += float(removed.sum())
            if sources is not None:
                _, _, _, source_material_id, source_amount = sources
                total_sputtered_volume_nm3 += float(
                    np.sum(
                        source_amount
                        / material_properties["capacity"][source_material_id]
                        * (DX_NM * DY_NM * DZ_NM),
                        dtype=np.float64,
                    )
                )
            # This statistic is now the unconsumed ion count when a column has
            # been completely emptied; atom-equivalent conversion is undefined
            # after the last material disappears.
            total_unfulfilled_sputter += float(unfulfilled_ions.sum())
            # remove_from_columns 不修改几何缓存，因此同时刷新离散表面索引和
            # 考虑顶层固体分数的连续物理高度。
            new_surface, new_height_nm = surface_state_in_region(
                frame, x0, x1, y0, y1, material_id,
                material_properties["capacity"],
            )
            surface[x0:x1, y0:y1] = new_surface
            surface_height_nm[x0:x1, y0:y1] = new_height_nm

            if ENABLE_REDEPOSTION:
                # 先以刻蚀后的表面更新命中法线；source_normals 已在上方从刻蚀前
                # 的 slope 缓存得到。沉积又可能改变表面，故完成后再增量更新一次。
                changed_x = x0 + local_source_x
                changed_y = y0 + local_source_y
                update_surface_normals(
                    surface_height_nm, normals, changed_x, changed_y
                )
                (
                    emitted,
                    deposited,
                    deposited_x,
                    deposited_y,
                ) = apply_redeposition(
                    frame, material_id, sources, source_normals, normals,
                    surface, material_properties,
                )
                total_emitted_for_redeposition += emitted
                total_redeposited += deposited

                # 沉积可能发生在束斑之外。用沉积列的最小包围盒重新提取离散索引
                # 和连续高度，再更新这些列及一阶邻域的法线。
                if deposited_x.size > 0:
                    deposit_x0 = int(deposited_x.min())
                    deposit_x1 = int(deposited_x.max()) + 1
                    deposit_y0 = int(deposited_y.min())
                    deposit_y1 = int(deposited_y.max()) + 1
                    deposited_surface, deposited_height_nm = (
                        surface_state_in_region(
                            frame,
                            deposit_x0,
                            deposit_x1,
                            deposit_y0,
                            deposit_y1,
                            material_id,
                            material_properties["capacity"],
                        )
                    )
                    surface[
                        deposit_x0:deposit_x1,
                        deposit_y0:deposit_y1,
                    ] = deposited_surface
                    surface_height_nm[
                        deposit_x0:deposit_x1,
                        deposit_y0:deposit_y1,
                    ] = deposited_height_nm
                update_surface_normals(
                    surface_height_nm, normals, deposited_x, deposited_y
                )

        # ---- 11.4 定期可视化和进度 ----
        # 查询 step+1 的状态，因为图像对应本步更新完成后的 t+DT。
        if (step + 1) % SNAPSHOT_INTERVAL == 0:
            (
                snapshot_x,
                snapshot_y,
                _,
                snapshot_on,
                snapshot_pass,
                snapshot_point,
            ) = beam_state_for_step(step + 1)
            snapshot_stem = f"snap_{snapshot_count:04d}"
            save_snapshot(
                surface_height_nm,
                t_s + DT_S,
                snapshot_x,
                snapshot_y,
                snapshot_on,
                snapshot_pass,
                snapshot_point,
                snapshot_stem + ".png",
            )
            # Keep the complete 3-D material state synchronized with every
            # 2-D height image, using the same snapshot number.
            write_frame_vtk(
                os.path.join(OUTPUT_DIR, snapshot_stem + ".vtk"),
                frame,
                material_id,
                material_properties["capacity"],
            )
            snapshot_count += 1

        if (step + 1) % progress_interval == 0:
            elapsed = time.time() - start_time
            completed_fraction = (step + 1) / N_STEPS
            eta = elapsed / completed_fraction * (1.0 - completed_fraction)
            print(
                f"Progress: {completed_fraction * 100:5.1f}% | "
                f"elapsed: {elapsed:6.1f} s | ETA: {eta:6.1f} s",
                flush=True,
            )

    # ---- 11.5 保存最终三维体数据、离散索引、连续高度和刻蚀深度 ----
    etch_depth = np.maximum(
        initial_surface_height_nm - surface_height_nm,
        0.0,
    )
    write_frame_vtk(
        os.path.join(OUTPUT_DIR, "frame_final.vtk"), frame,
        material_id, material_properties["capacity"],
    )
    np.save(os.path.join(OUTPUT_DIR, "frame_final_atoms.npy"), frame)
    np.save(os.path.join(OUTPUT_DIR, "material_id_final.npy"), material_id)
    np.save(os.path.join(OUTPUT_DIR, "surface_final_index.npy"), surface)
    np.save(
        os.path.join(OUTPUT_DIR, "surface_final_nm.npy"), surface_height_nm
    )
    np.save(
        os.path.join(OUTPUT_DIR, "etch_depth_final_nm.npy"), etch_depth
    )
    np.save(
        os.path.join(OUTPUT_DIR, "solid_fraction_final.npy"),
        np.where(
            material_id >= 0,
            np.clip(
                frame
                / material_properties["capacity"][
                    np.maximum(material_id, 0)
                ],
                0.0,
                1.0,
            ),
            0.0,
        ),
    )

    # 为降低绘图成本，每隔 plot_step 个 x/y 网格采样一次连续表面高度。
    figure = plt.figure(figsize=(8, 6))
    axes = figure.add_subplot(111, projection="3d")
    plot_step = 2
    grid_x, grid_y = np.meshgrid(
        np.arange(0, NX, plot_step) * DX_NM,
        np.arange(0, NY, plot_step) * DY_NM,
        indexing="ij",
    )
    surface_plot = surface_height_nm[::plot_step, ::plot_step]
    plot_height_min_nm, plot_height_max_nm = _data_color_limits(
        surface_plot
    )
    surface_artist = axes.plot_surface(
        grid_x,
        grid_y,
        surface_plot,
        cmap="viridis",
        vmin=plot_height_min_nm,
        vmax=plot_height_max_nm,
    )
    figure.colorbar(
        surface_artist,
        ax=axes,
        shrink=0.65,
        pad=0.1,
        label="surface height (nm)",
    )
    axes.set_xlabel("X (nm)")
    axes.set_ylabel("Y (nm)")
    axes.set_zlabel("surface height (nm)")
    redeposition_label = "with redeposition" if ENABLE_REDEPOSTION else ""
    axes.set_title(
        f"Final continuous surface height with serpentine scan "
        f"{redeposition_label}".strip()
    )
    figure.savefig(
        os.path.join(OUTPUT_DIR, "surface3d_final.png"),
        dpi=120,
        bbox_inches="tight",
    )
    plt.close(figure)

    # ---- 11.6 后处理及守恒检查 ----
    final_atoms = float(frame.sum(dtype=np.float64))

    # ---- 11.6.1 刻蚀体积、电荷剂量及 VPD 统计 ----
    # 这里只累计 beam-on 时间；每遍扫描结束后的束流关闭返回时间不计入剂量。
    # beam_charge_nC 是实验设备通常采用的标称电荷 I*t。对于 Poisson 离子
    # 抽样，sampled_charge_nC 则是本次随机实现实际抽到的离子所对应的电荷，
    # 可用于检查统计涨落，但和设备标称 VPD 比较时优先使用 beam_charge_nC。
    cell_volume_nm3 = DX_NM * DY_NM * DZ_NM
    beam_on_time_s = beam_on_steps * DT_S
    beam_charge_nC = I_BEAM_A * beam_on_time_s * 1.0e9
    sampled_charge_nC = (
        total_sampled_ions * ELEMENTARY_CHARGE_C * 1.0e9
    )

    # Multi-material volume must be calculated with each voxel's own capacity.
    net_removed_atoms = initial_atoms - final_atoms
    final_capacity_field = material_properties["capacity"][
        np.maximum(material_id, 0)
    ]
    final_solid_volume_nm3 = float(
        np.where(
            material_id >= 0,
            frame / final_capacity_field * cell_volume_nm3,
            0.0,
        ).sum(dtype=np.float64)
    )
    net_removed_volume_um3 = (
        initial_solid_volume_nm3 - final_solid_volume_nm3
    ) / 1.0e9
    gross_sputtered_volume_um3 = total_sputtered_volume_nm3 / 1.0e9

    if beam_charge_nC > 0.0:
        net_vpd_um3_per_nC = net_removed_volume_um3 / beam_charge_nC
        gross_vpd_um3_per_nC = (
            gross_sputtered_volume_um3 / beam_charge_nC
        )
    else:
        # 例如调试截断发生在第一个 beam-on 步之前时，VPD 没有定义。
        net_vpd_um3_per_nC = float("nan")
        gross_vpd_um3_per_nC = float("nan")

    print("=" * 60)
    print(f"Simulation completed. Steps: {N_STEPS}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Wall time: {time.time() - start_time:.1f} s")
    print(f"Grid cells X/Y/Z: {NX}/{NY}/{NZ}")
    print(f"Z-domain height: {DOMAIN_HEIGHT_Z_NM:g} nm")
    print(
        "Initial non-empty surface height min/max: "
        f"{initial_height_min_nm:g}/{initial_height_max_nm:g} nm"
    )
    print(
        "Initial headspace above highest surface: "
        f"{DOMAIN_HEIGHT_Z_NM - initial_height_max_nm:g} nm"
    )
    print(
        "Frame + material-id array memory: "
        f"{(frame.nbytes + material_id.nbytes) / 2.0**20:.1f} MiB"
    )
    print(f"Pitch X/Y: {PITCH_X_NM:g}/{PITCH_Y_NM:g} nm")
    print(f"Points X/Y: {N_POINTS_X}/{N_POINTS_Y}")
    print(f"Dwell time per point: {DWELL_TIME_S * 1e6:.3f} us")
    print(f"Expected ions per point: {EXPECTED_IONS_PER_POINT:.3e}")
    print(f"Expected ions per pass: {EXPECTED_IONS_PER_PASS:.3e}")
    print(f"Planned expected ions: {PLANNED_EXPECTED_IONS:.3e}")
    print(f"Steps per dwell: {STEPS_PER_DWELL}")
    print(f"Actual time step: {DT_S * 1e9:.3f} ns")
    print(f"Completed scan cycles: {N_PASS}")
    print(f"Scan time per pass: {SCAN_TIME_PER_PASS_S * 1e3:.3f} ms")
    print(f"IB return velocity: {IB_RETURN_VELOCITY_NM_S:.3e} nm/s")
    print(f"IB return distance: {IB_RETURN_DISTANCE_NM:.3f} nm")
    print(f"IB return time: {IB_RETURN_TIME_S * 1e3:.3f} ms")
    print(f"IB return steps per pass: {IB_RETURN_STEPS}")
    print(f"Total steps: {N_STEPS}")
    print(f"Total simulation time: {TOTAL_TIME_S * 1e3:.3f} ms")
    print(f"Beam effect radius: {BEAM_EFFECT_RADIUS_NM:g} nm")
    print(f"Beam-on steps: {beam_on_steps}/{N_STEPS}")
    print(f"Redeposition enabled: {ENABLE_REDEPOSTION}")
    if ENABLE_REDEPOSTION:
        print(f"Redeposition rays/source: {REDEPO_RAYS}")
    print(f"Expected ions: {total_expected_ions:.3e}")
    print(f"Sampled ions: {total_sampled_ions:.3e}")
    print(f"Beam-on time: {beam_on_time_s:.9e} s")
    print(f"Nominal beam charge: {beam_charge_nC:.9e} nC")
    print(f"Sampled-ion charge: {sampled_charge_nC:.9e} nC")
    print(
        "Initial-surface requested sputter atoms: "
        f"{total_requested_sputter:.3e}"
    )
    print(f"Actually sputtered atoms: {total_sputtered:.3e}")
    print(f"Unconsumed ions after empty columns: {total_unfulfilled_sputter:.3e}")
    print(
        "Emitted for redeposition: "
        f"{total_emitted_for_redeposition:.3e} atoms"
    )
    print(f"Redeposited atoms: {total_redeposited:.3e}")
    print(
        "Redeposition fraction: "
        f"{total_redeposited / max(total_emitted_for_redeposition, 1e-30) * 100:.2f}%"
    )
    print(f"Net removed atoms: {net_removed_atoms:.3e}")
    print(f"Net removed volume: {net_removed_volume_um3:.9e} um^3")
    print(
        "Gross sputtered volume: "
        f"{gross_sputtered_volume_um3:.9e} um^3"
    )
    print(f"Net VPD: {net_vpd_um3_per_nC:.6f} um^3/nC")
    print(f"Gross VPD: {gross_vpd_um3_per_nC:.6f} um^3/nC")
    print(f"Maximum etch depth: {etch_depth.max():.1f} nm")
    print(
        "Atom balance error: "
        f"{initial_atoms + total_redeposited - final_atoms - total_sputtered:.3e} atoms"
    )
