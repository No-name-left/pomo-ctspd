
import torch
import numpy as np


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
