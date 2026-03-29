
import torch
import numpy as np
import re
from sklearn.manifold import MDS


def get_random_problems(batch_size, problem_size, num_groups):
 
    node_xy = torch.rand(size=(batch_size, problem_size, 2))
    
    # CHANGE: 确保每个优先级至少有一个节点的分配逻辑
    if problem_size >= num_groups:
        # 基础分配：每个优先级至少一个节点 [0, 1, 2, ..., num_groups-1]
        base_groups = torch.arange(num_groups).unsqueeze(0).expand(batch_size, num_groups)
        # shape: (batch, num_groups)
        
        # 剩余节点随机分配到各优先级
        remaining = problem_size - num_groups
        if remaining > 0:
            random_groups = torch.randint(0, num_groups, (batch_size, remaining))
            all_groups = torch.cat([base_groups, random_groups], dim=1)
        else:
            all_groups = base_groups
        # shape: (batch, problem_size)
        
        # 对每个batch独立随机打乱，避免前num_groups个节点总是优先级1,2,3...
        perm = torch.argsort(torch.rand(batch_size, problem_size), dim=1)
        all_groups = torch.gather(all_groups, 1, perm)
        
    else:
        # 节点数不足时的降级策略：随机选择problem_size个不同的优先级（不重复）
        # 这样至少保证存在的优先级是连续的1..problem_size
        all_groups = torch.zeros(batch_size, problem_size, dtype=torch.long)
        for b in range(batch_size):
            # 随机选择problem_size个不同的优先级（0-based: 0..num_groups-1）
            selected = torch.randperm(num_groups)[:problem_size]
            all_groups[b] = selected.sort()[0]  # 排序使其相对有序
    
    # 转为1-based优先级并增加维度 (batch, problem, 1)
    node_priority = all_groups.float().unsqueeze(2) + 1.0
    
    # 拼接为 (batch, problem, 3)
    problems = torch.cat([node_xy, node_priority], dim=2)
    return problems



def augment_xy_data_by_8_fold(problems):
    # CHANGE: 分离坐标和优先级
    node_xy = problems[:, :, :2]      # (batch, problem, 2)
    node_priority = problems[:, :, 2:] # (batch, problem, 1)，保持3维便于cat
    
    # 对坐标进行8种几何变换（原有逻辑）
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
    # shape: (8*batch, problem, 2)
    
    # CHANGE: 对优先级进行8倍复制（几何变换不改变优先级）
    aug_priority = torch.cat([node_priority] * 8, dim=0)
    # shape: (8*batch, problem, 1)
    
    # CHANGE: 重新拼接为3D张量
    aug_problems = torch.cat([aug_xy, aug_priority], dim=2)
    # shape: (8*batch, problem, 3)

    return aug_problems

def load_ctspd_instance(filename):
    """
    加载 .ctspd 文件(TSPLIB格式),返回坐标、优先级、松弛度和原始距离矩阵
    
    适配任意规模(n20, n52, n100等)
    """
    with open(filename, 'r') as f:
        content = f.read()
    
    # 解析头部
    dimension = int(re.search(r'DIMENSION\s*:\s*(\d+)', content).group(1))
    num_groups = int(re.search(r'GROUPS\s*:\s*(\d+)', content).group(1))
    relaxation_d = int(re.search(r'RELAXATION_LEVEL\s*:\s*(\d+)', content).group(1))
    
    # 解析距离矩阵（上三角或全矩阵）
    matrix_section = re.search(r'EDGE_WEIGHT_SECTION\s*\n(.*?)\n\s*(?:GROUP_SECTION|EOF)', 
                               content, re.DOTALL)
    matrix_values = list(map(float, matrix_section.group(1).split()))
    
    # 重构距离矩阵 (n, n)
    dist_matrix = np.zeros((dimension, dimension))
    idx = 0
    for i in range(dimension):
        for j in range(i+1, dimension):
            if idx < len(matrix_values):
                dist_matrix[i, j] = matrix_values[idx]
                dist_matrix[j, i] = matrix_values[idx]
                idx += 1
    
    # 使用 MDS 将距离矩阵转为 2D 坐标（归一化到[0,1]）
    # n_components=2 确保输出是2维坐标，与训练数据一致
    mds = MDS(n_components=2, dissimilarity='precomputed', 
              random_state=42, normalized_stress=False, max_iter=500)
    coords = mds.fit_transform(dist_matrix)
    
    # 归一化到 [0, 1]（与训练数据一致）
    coords_min = coords.min(axis=0)
    coords_max = coords.max(axis=0)
    coords_norm = (coords - coords_min) / (coords_max - coords_min + 1e-8)
    
    # 转换为 PyTorch 张量
    node_xy = torch.from_numpy(coords_norm).float()  # (n, 2)
    
    # 解析 GROUP_SECTION（优先级分配）
    node_priority = torch.ones(dimension)  # 默认优先级1
    group_section = re.search(r'GROUP_SECTION\s*\n(.*?)\n\s*EOF', content, re.DOTALL)
    
    if group_section:
        lines = group_section.group(1).strip().split('\n')
        for line in lines:
            parts = list(map(int, line.split()))
            if len(parts) >= 2 and parts[0] != -1:
                group_id = parts[0]  # 组ID即优先级（1-based）
                nodes = parts[1:-1] if parts[-1] == -1 else parts[1:]  # -1是行结束符
                for node_idx in nodes:
                    if 1 <= node_idx <= dimension:
                        node_priority[node_idx - 1] = float(group_id)  # 转为0-based索引
    
    # 转为 (batch=1, node, 3) 格式，与训练数据一致
    problems = torch.zeros(1, dimension, 3)
    problems[0, :, :2] = node_xy
    problems[0, :, 2] = node_priority
    
    # 返回：归一化坐标、优先级、松弛度、原始距离矩阵（用于计算真实路径长度）
    return problems, node_priority, relaxation_d, torch.from_numpy(dist_matrix).float()


def load_ctspd_tour(filename):
    """
    加载 .tour 文件_最优路径),返回节点访问顺序(0-based索引)
    用于验证约束或计算 BKS
    """
    with open(filename, 'r') as f:
        content = f.read()
    
    # 解析 TOUR_SECTION
    tour_section = re.search(r'TOUR_SECTION\s*\n(.*?)\n\s*-1', content, re.DOTALL)
    if tour_section:
        nodes = list(map(int, tour_section.group(1).split()))
        # 转为 0-based 索引
        tour = [n - 1 for n in nodes if n > 0]
        return tour
    return None
