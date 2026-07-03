import numpy as np
import math

from functools import partial
import time
from scipy.spatial import KDTree
import networkx as nx

import freud



def compute_array_statistics(time_array, data_array):
    """
    Given time_array (1D) and data_array (1D), compute:
      - mean
      - variance (sample variance)
      - slope from a linear fit data_array ~ slope * time_array + intercept

    Returns a dict with keys: 'mean', 'variance', 'slope'.
    """
    # Basic stats
    mean_val = np.mean(data_array)
    var_val = np.var(data_array)  # sample variance
    data_array =  (data_array - mean_val) / np.sqrt(var_val)
    time_array = time_array / (time_array[-1] - time_array[0])

    # If data_array is constant or time_array has only 1 point, slope=0.
    if len(time_array) < 2 or np.allclose(data_array, data_array[0]):
        slope_val = 0.0
    else:
        slope_val, intercept = np.polyfit(time_array, data_array, 1)

    return {
        "mean": mean_val,
        "variance": var_val,
        "slope": slope_val
    }


def build_contact_matrix(positions, contact_radius, particle_types, type_list, q=None):
    """
    Builds a matrix of contacts using KDTree, optimized for different particle types.
    
    :param positions: An (N, 3) array of particle coordinates.
    :param contact_radius: A single value or an (N_types, N_types) matrix defining the contact radius for each pair of particle types.
    :param particle_types: An array indicating the type of each particle.
    :param q: list, if != None, then function will also calculate and return quantile=q - q th quantile for distribution of each possible pair (i,j) 
    :return: A matrix where the element at position (i, j) represents the average number of neighbors of type j around a particle of type i.
    """
    # Determine the number of particles and types
    N = positions.shape[0]
    N_types = len(type_list)
    positions  = np.array(positions)

    # Initialize the contact matrix
    contact_matrix = np.zeros((N_types, N_types))

    if q != None:
        q = list(q)
        q_matrix = np.zeros((N_types, N_types, len(q)))
    
    # Check if contact_radius is a single value or a matrix
    if np.isscalar(contact_radius):
        # Single value: use the same radius for all particle types
        radii_matrix = np.full((N_types, N_types), contact_radius)
    else:
        # Matrix of values: each particle type pair has its own radius
        radii_matrix = contact_radius

    # Build separate KDTree for each particle type
    trees = {}
    for t in range(N_types):

        type_positions = positions[particle_types == t]
        trees[t] = KDTree(type_positions)
    
    # Perform cross-queries between trees of different types
    for i in range(N_types):
        for j in range(N_types):
            tree_i = trees[i]
            tree_j = trees[j]
            radius = radii_matrix[i, j]
            
            if i == j:
                # Handle self-type contacts
                neighbors = tree_i.query_ball_tree(tree_j, r=radius)
                # Calculate the average number of neighbors, subtracting 1 for self-counting
                counts = np.array([len(nbrs) - 1 for nbrs in neighbors])
                
                contact_matrix[i, j] = np.mean(counts)
                if q != None:
                    for k, q_k in enumerate(q):
                        if len(counts) == 0:
                            q_matrix[i,j,k] = 0
                        else:
                            q_matrix[i,j,k] = np.quantile(counts, q=q_k)
            else:
                # Cross-type contacts
                neighbors_ij = tree_i.query_ball_tree(tree_j, r=radius)

                
                counts_ij = np.array([len(nbrs) for nbrs in neighbors_ij])


                contact_matrix[i, j] = np.mean(counts_ij)


                if q != None:
                    for k, q_k in enumerate(q):
                        if len(counts_ij) == 0:
                             q_matrix[i,j,k] = 0
                        else:
                            q_matrix[i,j,k] = np.quantile(counts_ij, q=q_k)
    if q != None:
        return contact_matrix, q_matrix
    else:
        return contact_matrix
    
def rotate_vector_by_quaternion(vec, quat):
    """
    Rotate vector 'vec' (3,) by quaternion 'quat' = (w, x, y, z),
    returning (3,) in the global coordinate frame.

    Uses: v' = q * v * conj(q),
          where v is treated as a pure quaternion (0, vx, vy, vz).
    """
    w, x, y, z = quat
    vx, vy, vz = vec

    # q * v
    rw = -x*vx - y*vy - z*vz
    rx =  w*vx + y*vz - z*vy
    ry =  w*vy - x*vz + z*vx
    rz =  w*vz + x*vy - y*vx

    # multiply result by conj(q)
    w2, x2, y2, z2 = (w, -x, -y, -z)

    rw2 = rw*w2 - rx*x2 - ry*y2 - rz*z2
    rx2 = rw*x2 + rx*w2 + ry*z2 - rz*y2
    ry2 = rw*y2 - rx*z2 + ry*w2 + rz*x2
    rz2 = rw*z2 + rx*y2 - ry*x2 + rz*w2

    return np.array([rx2, ry2, rz2], dtype=float)

def build_patchy_contact_matrix(
    positions,
    contact_radius,       # float or (N_types, N_types) matrix
    particle_types,       # array (length N) with type indices
    type_list,            # list of distinct types
    orientations,         # (N, 4) array of orientation quaternions (w, x, y, z)
    patch_directions_dict,# dictionary: { type_idx: [ (px, py, pz), ... ], ... }
    alpha,                # patch half-angle in radians
    q=None
):
    """
    Builds a contact matrix exactly like build_contact_matrix but with an added 
    patch-angle check. For each candidate neighbor, if it is not the same particle,
    check if it forms a valid patch-patch contact by:
      - Rotating each particle's local patch directions (via its orientation)
        into the global frame,
      - Computing the dot product with the unit vector connecting the particles.
    A neighbor is counted only if at least one patch on particle i points toward j
    and at least one patch on particle j points back, within the half-angle alpha.
    
    When alpha >= pi, the angular check is disabled and every neighbor is counted,
    matching the distance-only function.
    
    :return: contact_matrix, or (contact_matrix, q_matrix) if q != None.
    """
    positions = np.asarray(positions, dtype=float)
    orientations = np.asarray(orientations, dtype=float)
    N = positions.shape[0]
    N_types = len(type_list)

    # Prepare output array
    contact_matrix = np.zeros((N_types, N_types))
    if q is not None:
        q = list(q)
        q_matrix = np.zeros((N_types, N_types, len(q)))
    else:
        q_matrix = None

    # Radii matrix from contact_radius
    if np.isscalar(contact_radius):
        radii_matrix = np.full((N_types, N_types), contact_radius, dtype=float)
    else:
        radii_matrix = np.array(contact_radius, dtype=float)

    # Build KDTree and type indices exactly as in build_contact_matrix
    trees = {}
    type_indices = {}
    #### can be an issue if types are enumerated in wierd way
    for t in range(N_types):
        inds = np.where(particle_types == t)[0]
        type_indices[t] = inds
        coords = positions[inds]
        trees[t] = KDTree(coords)

    # Helper: check if two particles form a patch contact
    def is_patch_contact(i_idx, j_idx, alpha):
        """
        Returns True if the patch-angle condition is met between particles i and j.
        
        For each neighbor, we:
          1. Compute the unit vector from i to j.
          2. For particle i: For each local patch direction, rotate it (using the orientation)
             into the global frame and compute the dot product with that unit vector.
          3. For particle j: Do the analogous check (using the negative of that unit vector).
          4. If at least one patch on i and one patch on j yield a dot product 
             >= cos(half_angle), return True.
        
        When half_angle >= pi, returns True immediately.
        """

        r_ij = positions[j_idx] - positions[i_idx]
        dist_ij = np.linalg.norm(r_ij)
        if dist_ij < 1e-9:
            return False
        # Unit vectors from i to j and vice versa
        nr_ij = r_ij / dist_ij
        nr_ji = -nr_ij

        ti = particle_types[i_idx]
        tj = particle_types[j_idx]

        # Get local patch directions (assumed defined in the particle's local frame)
        patches_i = patch_directions_dict[ti]
        patches_j = patch_directions_dict[tj]

        if len(patches_i) == 0 or len(patches_j) == 0:
            return False

        q_i = orientations[i_idx]
        q_j = orientations[j_idx]

        cos_limit = math.cos(alpha)


        for p_i_local in patches_i:
            p_i_global = rotate_vector_by_quaternion(p_i_local, q_i)
            norm_i = np.linalg.norm(p_i_global) + 1e-16
            p_i_global /= norm_i

            if np.dot(p_i_global, nr_ij) >= cos_limit:
                for p_j_local in patches_j:
                    p_j_global = rotate_vector_by_quaternion(p_j_local, q_j)
                    norm_j = np.linalg.norm(p_j_global) + 1e-16
                    p_j_global /= norm_j

                    if np.dot(p_j_global, nr_ji) >= cos_limit:
                        return True

        return False



    # Main loop: same aggregator as in build_contact_matrix, but with an extra angle test.
    for i in range(N_types):
        for j in range(i, N_types):
            radius_ij = radii_matrix[i, j]
            tree_i = trees[i]
            tree_j = trees[j]
            inds_i = type_indices[i]
            inds_j = type_indices[j]

            if i == j:
                neighbors = tree_i.query_ball_tree(tree_j, r=radius_ij)
                counts = []
                for idx_local, nb_list in enumerate(neighbors):
                    global_i = inds_i[idx_local]
                    c = 0
                    for idx in nb_list:
                        global_j = inds_j[idx]
                        if global_i == global_j:
                            continue
                        if is_patch_contact(global_i, global_j, alpha):
                            c += 1
                    counts.append(c)
                counts = np.array(counts, dtype=int)
                contact_matrix[i, j] = np.mean(counts) if len(counts) > 0 else 0.0
                if q is not None:
                    for k, q_val in enumerate(q):
                        q_matrix[i, j, k] = 0 if len(counts)==0 else np.quantile(counts, q=q_val)
            else:
                # Cross-type: i->j
                neighbors_ij = tree_i.query_ball_tree(tree_j, r=radius_ij)
                counts_ij = []
                for idx_local, nb_list in enumerate(neighbors_ij):
                    global_i = inds_i[idx_local]
                    c = 0
                    for idx in nb_list:
                        global_j = inds_j[idx]
                        if is_patch_contact(global_i, global_j, alpha):
                            c += 1
                    counts_ij.append(c)
                counts_ij = np.array(counts_ij, dtype=int)
                contact_matrix[i, j] = np.mean(counts_ij) if len(counts_ij) > 0 else 0.0

                # Cross-type: j->i
                neighbors_ji = tree_j.query_ball_tree(tree_i, r=radius_ij)
                counts_ji = []
                for idx_local, nb_list in enumerate(neighbors_ji):
                    global_j = inds_j[idx_local]
                    c = 0
                    for idx in nb_list:
                        global_i = inds_i[idx]
                        if is_patch_contact(global_j, global_i, alpha):
                            c += 1
                    counts_ji.append(c)
                counts_ji = np.array(counts_ji, dtype=int)
                contact_matrix[j, i] = np.mean(counts_ji) if len(counts_ji) > 0 else 0.0

                if q is not None:
                    for k, q_val in enumerate(q):
                        q_matrix[i, j, k] = 0 if len(counts_ij)==0 else np.quantile(counts_ij, q=q_val)
                        q_matrix[j, i, k] = 0 if len(counts_ji)==0 else np.quantile(counts_ji, q=q_val)

    if q is not None:
        return contact_matrix, q_matrix
    else:
        return contact_matrix

def build_contact_k_ratios(positions, contact_radius, particle_types, type_list, k):
    """
    For each pair of types (i, j), computes two ratios among all i-type particles:
      1) The fraction that have exactly k neighbors of type j.
      2) The fraction that have more than k neighbors of type j.

    Positions are binned via KDTree with radius queries, analogous to build_contact_matrix().

    Parameters
    ----------
    positions : np.ndarray, shape (N, 3)
        Coordinates of all particles.
    contact_radius : float or (N_types, N_types) array
        If a float, use the same cutoff radius for every pair (i, j).
        If an array, contact_radius[i, j] is the cutoff for type-pair (i, j).
    particle_types : 1D np.ndarray, length N
        For each particle, an integer type ID from 0 to N_types-1.
    type_list : list of str (or list of type IDs)
        Distinct types in the system, must match the integer IDs used in `particle_types`.
    k : int
        The threshold for "exactly k" and "more than k" neighbors.

    Returns
    -------
    ratio_exactly_k : np.ndarray, shape (N_types, N_types)
        ratio_exactly_k[i, j] = fraction of i-type particles that have exactly k neighbors of type j
    ratio_more_than_k : np.ndarray, shape (N_types, N_types)
        ratio_more_than_k[i, j] = fraction of i-type particles that have more than k neighbors of type j
    """
    positions = np.asarray(positions)
    N_types = len(type_list)

    # Initialize output arrays
    ratio_exactly_k  = np.zeros((N_types, N_types), dtype=np.float64)
    ratio_more_than_k = np.zeros((N_types, N_types), dtype=np.float64)

    # Check if contact_radius is a single value or a matrix
    if np.isscalar(contact_radius):
        radii_matrix = np.full((N_types, N_types), contact_radius)
    else:
        radii_matrix = np.asarray(contact_radius)
        assert radii_matrix.shape == (N_types, N_types), \
            "contact_radius must be a float or an (N_types, N_types) matrix."

    # Build a KDTree for each type
    trees = {}
    for t in range(N_types):
        coords_t = positions[particle_types == t]
        trees[t] = KDTree(coords_t)

    # Loop over type pairs
    for i in range(N_types):
        # All i-type positions
        n_i = np.sum(particle_types == i)  # # of i-type particles

        for j in range(N_types):
            # The radius for pair (i, j)
            r_ij = radii_matrix[i, j]

            # Query all i-type particles against the j-type KDTree
            neighbors_ij = trees[i].query_ball_tree(trees[j], r_ij)

            # Convert to neighbor counts
            # If i == j, subtract 1 so we don't count "self" as a neighbor
            if i == j:
                counts_ij = np.array([len(nbrs) - 1 for nbrs in neighbors_ij])
            else:
                counts_ij = np.array([len(nbrs) for nbrs in neighbors_ij])

            # Compute ratio of exactly k neighbors
            # Note: if n_i == 0, skip to avoid divide-by-zero
            if n_i > 0:
                ratio_exactly_k[i, j] = np.sum(counts_ij == k) / float(n_i)
                ratio_more_than_k[i, j] = np.sum(counts_ij > k) / float(n_i)
            else:
                ratio_exactly_k[i, j]   = 0.0
                ratio_more_than_k[i, j] = 0.0

    return ratio_exactly_k, ratio_more_than_k


def build_contact_k_ratios_patchy(
    positions,
    orientations,
    particle_types,
    type_list,
    patch_directions_dict,
    contact_radius,
    alpha,
    k
):
    """
    Для каждой пары типов (i, j) вычисляет две величины среди всех частиц типа i:
      1) Долю частиц типа i, у которых ровно k соседей типа j (с учётом patchy-критерия).
      2) Долю частиц типа i, у которых более k соседей типа j.
    
    Параметры
    ---------
    positions : np.ndarray, shape (N, 3)
        Координаты всех частиц.
    orientations : np.ndarray, shape (N, 4)
        Кватернионы частиц (w, x, y, z).
    particle_types : np.ndarray, shape (N,)
        Целочисленные ID типов (0..N_types-1).
    type_list : list
        Список всех уникальных типов (могут быть строки, главное, чтобы
        индекс в списке соответствовал `particle_types`).
    patch_directions_dict : dict
        Словарь вида: { type_idx: [(px1, py1, pz1), ...], ... }
        Задаёт направления патчей в ЛОКАЛЬНОЙ системе для каждого типа.
    contact_radius : float или (N_types, N_types)-массив
        Радиус для отбора соседей. Либо одно число, либо матрица
        размеров (N_types, N_types).
    alpha : float
        Полуугол патча (в радианах), при котором патчи считаются «смотрящими друг на друга».
    k : int
        Число соседей для условий "ровно k" и "более k".

    Возвращает
    ---------
    ratio_exactly_k : np.ndarray, shape (N_types, N_types)
        ratio_exactly_k[i, j] = доля (среди i-частиц), у которых ровно k j-соседей.
    ratio_more_than_k : np.ndarray, shape (N_types, N_types)
        ratio_more_than_k[i, j] = доля (среди i-частиц), у которых более k j-соседей.
    """

    positions = np.asarray(positions, dtype=float)
    orientations = np.asarray(orientations, dtype=float)
    N_types = len(type_list)

    # Результирующие массивы
    ratio_exactly_k  = np.zeros((N_types, N_types), dtype=np.float64)
    ratio_more_than_k = np.zeros((N_types, N_types), dtype=np.float64)

    # Преобразуем contact_radius в матрицу, если это скаляр
    if np.isscalar(contact_radius):
        radii_matrix = np.full((N_types, N_types), contact_radius, dtype=float)
    else:
        radii_matrix = np.asarray(contact_radius, dtype=float)
        assert radii_matrix.shape == (N_types, N_types), \
            "contact_radius должен быть float либо (N_types, N_types)-матрицей."

    # Собираем KDTree для каждого типа
    # Сохраним также глобальные индексы (inds_by_type[t]) частиц типа t
    trees = {}
    inds_by_type = {}
    for t_idx in range(N_types):
        inds = np.where(particle_types == t_idx)[0]
        coords_t = positions[inds]
        trees[t_idx] = KDTree(coords_t)
        inds_by_type[t_idx] = inds

    # Функция проверки patchy-контакта
    cos_alpha = math.cos(alpha)

    def is_patch_contact(i_idx, j_idx):
        """
        Проверяем, смотрят ли патчи между i-й и j-й частицами друг на друга
        при условии, что расстояние уже в нужном диапазоне.
        """
        r_ij = positions[j_idx] - positions[i_idx]
        dist_ij = np.linalg.norm(r_ij)
        if dist_ij < 1e-12:
            # Считаем, что идентичные позиции -> не учитываем
            return False
        
        nr_ij = r_ij / dist_ij
        nr_ji = -nr_ij

        ti = particle_types[i_idx]
        tj = particle_types[j_idx]

        patches_i = patch_directions_dict.get(ti, [])
        patches_j = patch_directions_dict.get(tj, [])

        # Если хотя бы у одного типа нет патчей, контакт невозможен
        if len(patches_i) == 0 or len(patches_j) == 0:
            return False

        quat_i = orientations[i_idx]
        quat_j = orientations[j_idx]

        # Для каждой пары патчей (p_i, p_j) проверяем, удовлетворяют ли углы
        for p_i_local in patches_i:
            # Переводим в глобальную систему
            p_i_global = rotate_vector_by_quaternion(p_i_local, quat_i)
            p_i_norm = p_i_global / (np.linalg.norm(p_i_global) + 1e-16)
            cos_angle_i = np.dot(p_i_norm, nr_ij)
            if cos_angle_i >= cos_alpha:
                # Нужно проверить патч j
                for p_j_local in patches_j:
                    p_j_global = rotate_vector_by_quaternion(p_j_local, quat_j)
                    p_j_norm = p_j_global / (np.linalg.norm(p_j_global) + 1e-16)
                    cos_angle_j = np.dot(p_j_norm, nr_ji)
                    if cos_angle_j >= cos_alpha:
                        return True
        return False

    # Основной цикл по парам типов (i, j)
    for i in range(N_types):
        n_i = len(inds_by_type[i])  # число i-частиц
        
        for j in range(N_types):
            r_ij = radii_matrix[i, j]

            # Выполняем поиск соседей: i-тип против j-тип
            # neighbors_ij[i_local] = список локальных индексов j-дерева
            neighbors_ij = trees[i].query_ball_tree(trees[j], r_ij)

            counts_patch = []
            inds_i = inds_by_type[i]
            inds_j = inds_by_type[j]

            # Для каждого i_local (частица типа i)
            for idx_local_i, nb_list_j in enumerate(neighbors_ij):
                global_i = inds_i[idx_local_i]

                count_ij = 0
                for idx_local_j in nb_list_j:
                    global_j = inds_j[idx_local_j]
                    if (i == j) and (global_i == global_j):
                        # Самая же частица -> исключаем
                        continue

                    # Проверяем patchy-критерий
                    if is_patch_contact(global_i, global_j):
                        count_ij += 1

                counts_patch.append(count_ij)

            counts_patch = np.array(counts_patch, dtype=int)

            if n_i > 0:
                ratio_exactly_k[i, j]  = np.sum(counts_patch == k) / float(n_i)
                ratio_more_than_k[i, j] = np.sum(counts_patch > k) / float(n_i)
            else:
                ratio_exactly_k[i, j]  = 0.0
                ratio_more_than_k[i, j] = 0.0

    return ratio_exactly_k, ratio_more_than_k


def calc_rdf_freud(snapshot, r_max=5.0, n_bins=100, reset=True):
    """
    Compute radial distribution function for a single frame using freud.

    Parameters
    ----------
    snapshot : gsd.hoomd.Snapshot
        The snapshot with positions and box info.
    r_max : float
        Maximum radius to compute RDF.
    n_bins : int
        Number of bins between r=0 and r=r_max.
    reset : bool
        If True, the RDF object is reset before computing (i.e., starts fresh).
        If False, you can accumulate data across multiple calls.

    Returns
    -------
    r_vals : np.ndarray
        The bin centers where g(r) is evaluated.
    g_r : np.ndarray
        The radial distribution function values at each bin center.
    """
    # Build a freud box from the snapshot
    box_data = snapshot.configuration.box  # [Lx, Ly, Lz, xy, xz, yz]
    Lx, Ly, Lz, xy, xz, yz = box_data
    fbox = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, xy=xy, xz=xz, yz=yz)


    # Positions array: shape (N, 3)
    positions = snapshot.particles.position

    # Construct a freud "state" object
    # "Query points" and "Reference points" can be the same for a single-species RDF
    system = (fbox, positions)

    # Create an RDF object. If you only have a single species, this will compute the total g(r).
    rdf_calc = freud.density.RDF(bins=n_bins, r_max=r_max)


    
    # Update the RDF with the system data
    rdf_calc.compute(system, reset=reset)

    # Extract r and g(r)
    r_vals = rdf_calc.bin_centers  # shape (n_bins,)
    g_r = rdf_calc.rdf             # shape (n_bins,)

    return r_vals, g_r

def calc_kinetic_energy(snapshot, mass=1.0):
    """
    Compute the average (per-particle) kinetic energy from a GSD snapshot.
    Assumes all particles have the same mass.

    Parameters
    ----------
    snapshot : gsd.hoomd.Snapshot
        A GSD snapshot containing particle velocities.
    mass : float, optional
        Mass of each particle (default 1.0).

    Returns
    -------
    float
        Average kinetic energy per particle = (1/N) * sum(1/2 * mass * v^2).
        Returns 0.0 if no velocities are present in the snapshot.
    """
    velocities = snapshot.particles.velocity
    if velocities is None:
        # GSD file may not have velocity information or it may not be saved
        return 0.0

    # velocities shape is (N, 3) or (N, 4) if storing angular velocity
    # we assume (N, 3)
    speeds_sq = np.sum(velocities**2, axis=1)

    ke_total = 0.5 * mass * np.sum(speeds_sq)
    n_part = len(speeds_sq)

    # Avoid division by zero if something is off
    if n_part == 0:
        return 0.0

    ke_per_particle = ke_total / n_part
    return ke_per_particle



def compute_msd_from_gsd(
    gsd_filename,
    start_frame=0,
    end_frame=None,
    step=1
):
    """
    Считывает кадры из GSD-файла и вычисляет MSD при помощи freud.
    
    Параметры
    ---------
    gsd_filename : str
        Путь к GSD-файлу с симуляцией.
    start_frame : int
        Номер первого фрейма для расчёта (по умолчанию 0).
    end_frame : int или None
        Номер последнего фрейма (исключительно). Если None, берется весь файл до конца.
    step : int
        Шаг по фреймам: например, step=1 обработает каждый фрейм,
        step=10 возьмет каждый десятый и т. д.

    Возвращает
    ----------
    frames : np.ndarray
        Массив индексов кадров, по которым считался MSD.
    msd_values : np.ndarray
        Массив значений MSD (той же длины, что и `frames`).
    
    Примечание:
    -----------
    freud.msd.MSD по умолчанию считает, что частицы между кадрами
    одни и те же (нет «обмена» частиц, константное N). Если это не так,
    то нужно вручную обрабатывать соответствие индексов частиц.
    """

    # Создаем объект MSD из freud
    msd_calc = freud.msd.MSD()

    # Открываем GSD-файл
    with gsd.hoomd.open(name=gsd_filename, mode="rb") as traj:
        n_frames = len(traj)
        
        if end_frame is None or end_frame > n_frames:
            end_frame = n_frames

        # Инициализируем список кадров
        frame_list = range(start_frame, end_frame, step)

        for f_idx, frame_no in enumerate(frame_list):
            snapshot = traj[frame_no]

            # Извлекаем параметры бокса из snapshot.configuration.box
            # Формат [Lx, Ly, Lz, xy, xz, yz], см. документацию HOOMD
            box_data = snapshot.configuration.box
            # Создаем box для freud:
            # Если xy, xz, yz = 0 (прямой ортогональный бокс), то:
            box = freud.box.Box(Lx=box_data[0], Ly=box_data[1], Lz=box_data[2])

            # Координаты частиц
            positions = snapshot.particles.position  # (N,3)

            # Передаём кадр, позиции и бокс в MSD-калькулятор
            # Аргумент `frame` говорит, что это f_idx-й "логический" кадр
            # (часто ставят просто frame=frame_no, но тогда MSD будет
            #  учитывать пропуски, если они есть).
            msd_calc.compute(
                box=box,
                positions=positions,
                frame=f_idx  # или frame=frame_no
            )

    # По окончании считывания MSD-калькулятор хранит:
    # msd_calc.msd     -> np.ndarray с MSD на каждом шаге
    # msd_calc.frames  -> номера кадров, которые мы передавали (см. аргумент frame=...)
    frames = np.asarray(msd_calc.frames)
    msd_values = np.asarray(msd_calc.msd)

    return frames, msd_values



def quaternion_conjugate(q):
    """
    Conjugate of a quaternion q = (qw, qx, qy, qz).
    Returns (qw, -qx, -qy, -qz).
    """
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)

def quaternion_multiply(q1, q2):
    """
    Quaternion multiplication, both in form (w, x, y, z).
    Returns q1 * q2.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
    ], dtype=float)

def get_body_frame_angmom(q_orient, q_angmom):
    """
    Compute the angular momentum vector in the BODY frame for a single particle.
    
    From the HOOMD docs:
    S (body) = 1/2 * Im(q^* * P),
    where
      q = orientation quaternion (qw, qx, qy, qz),
      P = angular momentum quaternion (L0, L1, L2, L3),
      q^* = conjugate(q).
    """
    # Conjugate of orientation
    q_conj = quaternion_conjugate(q_orient)
    # Multiply q^* by P
    q_prod = quaternion_multiply(q_conj, q_angmom)
    # The imaginary part of q_prod -> x,y,z
    # Then multiply by 1/2
    Sx, Sy, Sz = q_prod[1], q_prod[2], q_prod[3]
    return 0.5 * np.array([Sx, Sy, Sz], dtype=float)

def rotational_kinetic_energy_per_particle(snapshot):
    """
    Compute the rotational kinetic energy for each particle in a snapshot.
    Returns an array E_rot of shape (N,).

    snapshot.particles.angmom -> quaternion (N,4)
    snapshot.particles.orientation -> quaternion (N,4)
    snapshot.particles.moment_inertia -> diagonal inertia (N,3)
    
    E_rot(i) = 0.5 * [ (Sx^2)/Ix + (Sy^2)/Iy + (Sz^2)/Iz ],
    where S is the body-frame angular momentum from HOOMD's formula.
    """
    angmom_quat = snapshot.particles.angmom       # shape (N,4)
    orientation_quat = snapshot.particles.orientation  # shape (N,4)
    inertia = snapshot.particles.moment_inertia        # shape (N,3)
    
    N = angmom_quat.shape[0]
    E_rot = np.zeros(N, dtype=np.float64)
    
    for i in range(N):
        qL = angmom_quat[i]       # (L0, L1, L2, L3)
        q_orient = orientation_quat[i]  # (qw, qx, qy, qz)
        Ix, Iy, Iz = inertia[i]   # principal moments of inertia

        # Compute S in body frame
        S_body = get_body_frame_angmom(q_orient, qL)
        Sx, Sy, Sz = S_body

        # Avoid division by zero in case Ix,Iy,Iz=0
        if Ix < 1e-16 or Iy < 1e-16 or Iz < 1e-16:
            # e.g., 2D or degenerate inertia. Set E=0 or skip
            E_rot[i] = 0.0
        else:
            # E = 0.5 * sum( S^2 / I )
            E_rot[i] = 0.5 * ((Sx**2)/Ix + (Sy**2)/Iy + (Sz**2)/Iz)
    
    return E_rot

def compute_average_rotational_energy(snapshot):
    """
    Return the average rotational energy (per particle) in a snapshot.
    """
    E_rot_per_particle = rotational_kinetic_energy_per_particle(snapshot)
    return np.mean(E_rot_per_particle)

