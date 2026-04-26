import re

import numpy as np
import torch


def get_random_problems(batch_size, problem_size, num_groups):
    node_xy = torch.rand(size=(batch_size, problem_size, 2))

    if num_groups <= 0:
        raise ValueError("num_groups must be positive")

    if problem_size >= num_groups:
        base_groups = torch.arange(num_groups).unsqueeze(dim=0).expand(batch_size, num_groups)

        remaining = problem_size - num_groups
        if remaining > 0:
            random_groups = torch.randint(0, num_groups, (batch_size, remaining))
            all_groups = torch.cat([base_groups, random_groups], dim=1)
        else:
            all_groups = base_groups

        perm = torch.argsort(torch.rand(size=(batch_size, problem_size)), dim=1)
        all_groups = torch.gather(all_groups, 1, perm)

    else:
        all_groups = torch.zeros(batch_size, problem_size, dtype=torch.long)
        for b in range(batch_size):
            selected = torch.randperm(num_groups)[:problem_size]
            all_groups[b] = selected.sort()[0]

    node_priority = all_groups.float().unsqueeze(dim=2) + 1.0

    problems = torch.cat([node_xy, node_priority], dim=2)
    return problems


def augment_xy_data_by_8_fold(problems):
    node_xy = problems[:, :, :2]
    node_priority = problems[:, :, 2:]

    x = node_xy[:, :, [0]]
    y = node_xy[:, :, [1]]

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    aug_priority = torch.cat([node_priority] * 8, dim=0)

    return torch.cat([aug_xy, aug_priority], dim=2)


def _extract_required_int(content, field_name):
    match = re.search(rf"{field_name}\s*:\s*(\d+)", content)
    if match is None:
        raise ValueError(f"Missing required field: {field_name}")
    return int(match.group(1))


def _extract_edge_weight_format(content):
    match = re.search(r"EDGE_WEIGHT_FORMAT\s*:\s*([A-Z_]+)", content)
    if match is None:
        return None
    return match.group(1).upper()


def _extract_distance_matrix(content, dimension):
    matrix_section = re.search(
        r"EDGE_WEIGHT_SECTION\s*\n(.*?)\n\s*(?:GROUP_SECTION|EOF)",
        content,
        re.DOTALL,
    )
    if matrix_section is None:
        raise ValueError("Missing EDGE_WEIGHT_SECTION in CTSP-d instance")

    values = np.asarray(list(map(float, matrix_section.group(1).split())), dtype=np.float32)
    edge_weight_format = _extract_edge_weight_format(content)

    full_len = dimension * dimension
    upper_row_len = dimension * (dimension - 1) // 2
    diag_row_len = dimension * (dimension + 1) // 2
    dist_matrix = np.zeros((dimension, dimension), dtype=np.float32)

    if edge_weight_format == "FULL_MATRIX" or len(values) == full_len:
        if len(values) != full_len:
            raise ValueError(
                f"FULL_MATRIX instance has {len(values)} values, expected {full_len}"
            )
        dist_matrix = values.reshape(dimension, dimension).copy()

    elif edge_weight_format in (None, "UPPER_ROW") or len(values) == upper_row_len:
        if len(values) != upper_row_len:
            raise ValueError(
                f"UPPER_ROW instance has {len(values)} values, expected {upper_row_len}"
            )
        idx = 0
        for i in range(dimension):
            for j in range(i + 1, dimension):
                dist_matrix[i, j] = values[idx]
                dist_matrix[j, i] = values[idx]
                idx += 1

    elif edge_weight_format == "LOWER_ROW":
        if len(values) != upper_row_len:
            raise ValueError(
                f"LOWER_ROW instance has {len(values)} values, expected {upper_row_len}"
            )
        idx = 0
        for i in range(1, dimension):
            for j in range(i):
                dist_matrix[i, j] = values[idx]
                dist_matrix[j, i] = values[idx]
                idx += 1

    elif edge_weight_format == "UPPER_DIAG_ROW" or len(values) == diag_row_len:
        if len(values) != diag_row_len:
            raise ValueError(
                f"UPPER_DIAG_ROW instance has {len(values)} values, expected {diag_row_len}"
            )
        idx = 0
        for i in range(dimension):
            for j in range(i, dimension):
                dist_matrix[i, j] = values[idx]
                dist_matrix[j, i] = values[idx]
                idx += 1

    elif edge_weight_format == "LOWER_DIAG_ROW":
        if len(values) != diag_row_len:
            raise ValueError(
                f"LOWER_DIAG_ROW instance has {len(values)} values, expected {diag_row_len}"
            )
        idx = 0
        for i in range(dimension):
            for j in range(i + 1):
                dist_matrix[i, j] = values[idx]
                dist_matrix[j, i] = values[idx]
                idx += 1

    else:
        raise ValueError(
            f"Unsupported EDGE_WEIGHT_FORMAT: {edge_weight_format} "
            f"with {len(values)} values for dimension {dimension}"
        )

    dist_matrix[dist_matrix >= 99998] = 0.0
    np.fill_diagonal(dist_matrix, 0.0)

    if not np.allclose(dist_matrix, dist_matrix.T):
        dist_matrix = (dist_matrix + dist_matrix.T) / 2.0

    return dist_matrix


def _extract_priorities(content, dimension):
    priorities = torch.ones(dimension)
    group_section = re.search(r"GROUP_SECTION\s*\n(.*?)\n\s*EOF", content, re.DOTALL)

    if group_section is None:
        return priorities

    for line in group_section.group(1).strip().split("\n"):
        parts = list(map(int, line.split()))
        if len(parts) < 2 or parts[0] == -1:
            continue

        group_id = parts[0]
        nodes = parts[1:-1] if parts[-1] == -1 else parts[1:]
        for node_idx in nodes:
            if 1 <= node_idx <= dimension:
                priorities[node_idx - 1] = float(group_id)

    return priorities


def _coords_from_distance_matrix(dist_matrix):
    dist_matrix = np.asarray(dist_matrix, dtype=np.float32)
    n_nodes = dist_matrix.shape[0]

    if n_nodes == 1:
        return torch.zeros((1, 2))

    anchor_a = int(np.argmax(dist_matrix.sum(axis=1)))
    anchor_b = int(np.argmax(dist_matrix[anchor_a]))
    if anchor_b == anchor_a:
        anchor_b = (anchor_a + 1) % n_nodes

    coords = np.stack(
        [
            dist_matrix[:, anchor_a],
            dist_matrix[:, anchor_b],
        ],
        axis=1,
    ).astype(np.float32)

    if np.allclose(coords[:, 0], coords[:, 1]):
        coords[:, 1] = dist_matrix.mean(axis=1).astype(np.float32)

    coords_min = coords.min(axis=0, keepdims=True)
    coords_max = coords.max(axis=0, keepdims=True)
    coords_norm = (coords - coords_min) / np.maximum(coords_max - coords_min, 1e-8)

    return torch.from_numpy(coords_norm).float()


def parse_ctspd_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    dimension = _extract_required_int(content, "DIMENSION")
    num_groups = _extract_required_int(content, "GROUPS")
    relaxation_d = _extract_required_int(content, "RELAXATION_LEVEL")

    dist_matrix = _extract_distance_matrix(content, dimension)
    node_xy = _coords_from_distance_matrix(dist_matrix)
    node_priority = _extract_priorities(content, dimension)

    problems = torch.zeros(1, dimension, 3)
    problems[0, :, :2] = node_xy
    problems[0, :, 2] = node_priority

    return problems, torch.from_numpy(dist_matrix).float(), relaxation_d, num_groups


def load_ctspd_instance(filename):
    """
    Load a CTSP-d instance and keep the historical return signature used by older code.
    Returns:
        problems: (1, n, 3)
        node_priority: (n,)
        relaxation_d: int
        dist_matrix: (n, n)
    """
    problems, dist_matrix, relaxation_d, _ = parse_ctspd_file(filename)
    node_priority = problems[0, :, 2].clone()
    return problems, node_priority, relaxation_d, dist_matrix


def load_ctspd_tour(filename):
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    tour_section = re.search(r"TOUR_SECTION\s*\n(.*?)\n\s*-1", content, re.DOTALL)
    if tour_section:
        nodes = list(map(int, tour_section.group(1).split()))
        return [n - 1 for n in nodes if n > 0]
    return None
