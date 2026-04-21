
from dataclasses import dataclass
from typing import Optional

import torch

from CSTPd_cluster.CTSPd_ProblemDef import get_random_problems, augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 3)，因为加入了优先级
    relaxation_d: Optional[torch.Tensor] = None
    # shape: (batch,)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: Optional[torch.Tensor] = None
    # shape: (batch, pomo)
    ninf_mask: Optional[torch.Tensor] = None
    # shape: (batch, pomo, node)
    current_min_priority: Optional[torch.Tensor] = None
    # shape: (batch, pomo)
    relaxation_d: Optional[torch.Tensor] = None
    # shape: (batch, pomo)


class CTSPdEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = int(env_params['problem_size'])
        self.pomo_size = int(env_params['pomo_size'])
        # CHANGE: 新增CTSP-d参数读取
        self.num_groups = int(env_params.get('num_groups', 5))  # 优先级组数
        if self.num_groups <= 0:
            raise ValueError("num_groups must be positive")
        self.relaxation_d_setting = env_params.get('relaxation_d', 1)
        self.relaxation_d_min = int(env_params.get('relaxation_d_min', 0))
        raw_d_max = env_params.get('relaxation_d_max', None)
        self.relaxation_d_max = self.num_groups - 1 if raw_d_max is None else int(raw_d_max)
        if not (0 <= self.relaxation_d_min <= self.relaxation_d_max <= self.num_groups - 1):
            raise ValueError("relaxation_d range must satisfy 0 <= min <= max <= num_groups - 1")
        self.relaxation_d: Optional[torch.Tensor] = None
        self.d: Optional[torch.Tensor] = None

        # Const @Load_Problem
        ####################################
        self.batch_size: Optional[int] = None
        self.BATCH_IDX: Optional[torch.Tensor] = None
        self.POMO_IDX: Optional[torch.Tensor] = None
        # IDX.shape: (batch, pomo)
        self.problems: Optional[torch.Tensor] = None
        # shape: (batch, node, 3)  # [x, y, priority]
        
        # CHANGE: 新增成员变量，用于存储分离后的数据
        self.node_xy: Optional[torch.Tensor] = None           # shape: (batch, node, 2)
        self.node_priorities: Optional[torch.Tensor] = None   # shape: (batch, node)
        self.group_ids: Optional[torch.Tensor] = None         # shape: (batch, node), model-side priority groups

        # Dynamic
        ####################################
        self.selected_count: Optional[int] = None
        self.current_node: Optional[torch.Tensor] = None
        # shape: (batch, pomo)
        self.selected_node_list: Optional[torch.Tensor] = None
        # shape: (batch, pomo, 0~problem)

        # CHANGE: 新增动态状态变量
        self.visited_mask: Optional[torch.Tensor] = None      # shape: (batch, pomo, node), bool
        self.step_state: Optional[Step_State] = None
        self.current_min_priority: Optional[torch.Tensor] = None  # shape: (batch, pomo)，当前未访问节点中的最高优先级

    def load_problems(self, batch_size, aug_factor=1, problems=None, relaxation_d=None):
        """
        加载问题数据
        Args:
            batch_size: 批次大小
            aug_factor: 数据增强倍数（1或8）
            problems: 可选，外部传入的 (batch, node, 3) 数据，用于测试
            relaxation_d: 可选，外部传入的标量或 (batch,) 张量
        """
        if problems is None:
            # 训练模式：随机生成
            loaded_batch_size = int(batch_size)
            loaded_problems = get_random_problems(loaded_batch_size, self.problem_size, self.num_groups)
        else:
            # 测试模式：使用传入的数据
            loaded_problems = problems
            loaded_batch_size = int(problems.size(0))

        d_values = self._make_relaxation_d(loaded_batch_size, loaded_problems.device, relaxation_d)
        
        # 数据增强
        if aug_factor > 1:
            if aug_factor == 8:
                loaded_batch_size *= 8
                loaded_problems = augment_xy_data_by_8_fold(loaded_problems)
                d_values = d_values.repeat(8)
            else:
                raise NotImplementedError
        
        # 分离坐标和优先级
        self.problems = loaded_problems
        self.node_xy = loaded_problems[:, :, :2]
        self.node_priorities = loaded_problems[:, :, 2]
        self.relaxation_d = d_values.to(device=loaded_problems.device, dtype=torch.float)
        self.d = self.relaxation_d
        self.batch_size = loaded_batch_size
        
        # 初始化索引
        self.BATCH_IDX = torch.arange(loaded_batch_size)[:, None].expand(loaded_batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(loaded_batch_size, self.pomo_size)

    def load_problems_from_file(self, problems_tensor, d):
        """
        从文件加载的数据初始化环境（用于测试）
        
        Args:
            problems_tensor: (batch=1, problem, 3) 包含归一化坐标和优先级
            d: 松弛度
        """
        self.batch_size = 1
        self.problems = problems_tensor
        self.node_xy = problems_tensor[:, :, :2]        # (1, n, 2)
        self.node_priorities = problems_tensor[:, :, 2]  # (1, n)
        self.relaxation_d = torch.tensor([d], device=problems_tensor.device, dtype=torch.float)
        self.d = self.relaxation_d
        
        # 重新初始化索引
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def get_current_tour(self):
        """
        获取当前已选择的节点访问顺序
        用于测试时记录路径
        """
        if self.selected_node_list is None:
            raise RuntimeError("reset must be called before reading the current tour.")
        selected_node_list = self.selected_node_list
        return selected_node_list[0, 0].cpu().numpy().tolist()  # 返回第一个batch第一个pomo的路径

    def reset(self):
        batch_size = self.batch_size
        batch_idx = self.BATCH_IDX
        pomo_idx = self.POMO_IDX
        problems = self.problems
        node_priorities = self.node_priorities
        relaxation_d = self.relaxation_d
        if (
            batch_size is None
            or batch_idx is None
            or pomo_idx is None
            or problems is None
            or node_priorities is None
            or relaxation_d is None
        ):
            raise RuntimeError("load_problems must be called before reset.")

        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)

        # CHANGE: 初始化访问掩码（False表示未访问）
        self.visited_mask = torch.zeros((batch_size, self.pomo_size, self.problem_size), dtype=torch.bool)

        # CHANGE: 计算初始current_min_priority（全局最小优先级）
        # 在未访问节点中找最小值（初始时所有节点都未访问）
        global_min = node_priorities.min(dim=1, keepdim=True)[0]  # (batch, 1)
        self.current_min_priority = global_min.expand(batch_size, self.pomo_size)  # (batch, pomo)

        # CREATE STEP STATE
        step_state = Step_State(BATCH_IDX=batch_idx, POMO_IDX=pomo_idx)
        step_state.relaxation_d = relaxation_d[:, None].expand(batch_size, self.pomo_size)
        step_state.ninf_mask = torch.zeros((batch_size, self.pomo_size, self.problem_size))
        self.step_state = step_state
        # CHANGE: 新增 - 应用d-松弛优先级规则（覆盖上面的零张量，屏蔽不符合优先级范围的节点）
        self._apply_priority_mask()
        self.group_ids = node_priorities.long()  # (batch, problem)，用于传入模型

        reward = None
        done = False
        return Reset_State(problems, relaxation_d), reward, done

    def pre_step(self):
        if self.step_state is None:
            raise RuntimeError("reset must be called before pre_step.")

        step_state = self.step_state
        reward = None
        done = False
        return step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)
        step_state = self.step_state
        selected_count = self.selected_count
        selected_node_list = self.selected_node_list
        batch_idx = self.BATCH_IDX
        pomo_idx = self.POMO_IDX
        visited_mask = self.visited_mask
        node_priorities = self.node_priorities
        current_min_priority = self.current_min_priority
        batch_size = self.batch_size
        relaxation_d = self.relaxation_d
        if (
            step_state is None
            or selected_count is None
            or selected_node_list is None
            or batch_idx is None
            or pomo_idx is None
            or visited_mask is None
            or node_priorities is None
            or current_min_priority is None
            or batch_size is None
            or relaxation_d is None
        ):
            raise RuntimeError("reset must be called before step.")

        selected_count += 1
        self.selected_count = selected_count
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((selected_node_list, selected[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)

        # UPDATE STEP STATE
        step_state.current_node = selected
        # shape: (batch, pomo)
        # CHANGE: 更新访问掩码（标记已访问节点）
        visited_mask[batch_idx, pomo_idx, selected] = True
        
        # CHANGE: 更新ninf_mask的基础部分（已访问节点设为-inf）
        ninf_mask = step_state.ninf_mask
        if ninf_mask is None:
            raise RuntimeError("reset must initialize ninf_mask before step.")
        ninf_mask[batch_idx, pomo_idx, selected] = float('-inf')
        # 注意：这里我们保留原有的"已访问节点置-inf"逻辑，但接下来会重新计算完整的优先级掩码

        # CHANGE: 重新计算current_min_priority（在未访问节点中找最小优先级）
        if selected_count < self.problem_size:
            # 将已访问节点的优先级设为无穷大，这样min不会选到它们
            # 扩展维度以匹配visited_mask: (batch, node) -> (batch, pomo, node)
            expanded_priorities = node_priorities.float().unsqueeze(dim=1).expand(-1, self.pomo_size, -1).clone()
            expanded_priorities[visited_mask] = float('inf')
            # 在未访问节点中找最小值（即最高优先级）
            current_min_priority = expanded_priorities.min(dim=2)[0]  # (batch, pomo)
            self.current_min_priority = current_min_priority

        # CHANGE: 更新Step_State中的current_min_priority
        step_state.current_min_priority = current_min_priority
        step_state.relaxation_d = relaxation_d[:, None].expand(batch_size, self.pomo_size)
        
        # CHANGE: 重新应用d-松弛优先级掩码（这覆盖了原有的简单mask更新）
        self._apply_priority_mask()

        # returning values
        done = (selected_count == self.problem_size)
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return step_state, reward, done

    # CHANGE: 新增核心方法 - 应用d-松弛优先级掩码
    def _apply_priority_mask(self):
        """
        根据d-松弛优先级规则更新ninf_mask。
        规则:设p为未访问节点中的最高优先级(数值最小),
        则只能访问优先级在[p, p+d]范围内的节点。
        """
        # p_min: (batch, pomo) -> (batch, pomo, 1) 用于广播
        step_state = self.step_state
        current_min_priority = self.current_min_priority
        node_priorities = self.node_priorities
        visited_mask = self.visited_mask
        relaxation_d = self.relaxation_d
        if (
            step_state is None
            or current_min_priority is None
            or node_priorities is None
            or visited_mask is None
            or relaxation_d is None
        ):
            raise RuntimeError("reset must initialize priority mask state.")

        p_min = current_min_priority.unsqueeze(dim=2)
        d_values = relaxation_d[:, None, None].to(device=p_min.device, dtype=p_min.dtype)
        p_max = p_min + d_values  # 允许的最高优先级
        
        # 所有节点优先级: (batch, node) -> (batch, 1, node) 用于广播
        priorities = node_priorities.unsqueeze(dim=1)
        
        # 非法条件：优先级 < p_min（更高的优先级未访问完）或 > p_max（超出松弛范围）
        illegal_low = priorities < p_min   # 优先级比当前最小值还小（数值更小=更紧急）
        illegal_high = priorities > p_max  # 优先级超出松弛窗口
        illegal_priority = illegal_low | illegal_high
        
        # 合并非法条件：已访问 或 优先级不符合要求
        combined_illegal = visited_mask | illegal_priority
        
        # 更新mask：非法位置为-inf，合法位置保持原值（或设为0）
        # 注意：我们需要保留之前已访问节点的-inf标记，所以这里重新计算完整mask
        step_state.ninf_mask = torch.zeros_like(visited_mask, dtype=torch.float)
        step_state.ninf_mask.masked_fill_(combined_illegal, float('-inf'))

    def _make_relaxation_d(self, batch_size, device, relaxation_d=None):
        if relaxation_d is None:
            relaxation_d = self.relaxation_d_setting

        if isinstance(relaxation_d, str):
            mode = relaxation_d.lower()
            if mode in ('random', 'variable', 'sample', 'auto'):
                return torch.randint(
                    self.relaxation_d_min,
                    self.relaxation_d_max + 1,
                    (batch_size,),
                    device=device,
                    dtype=torch.long,
                )
            relaxation_d = int(relaxation_d)

        if torch.is_tensor(relaxation_d):
            d_values = relaxation_d.to(device=device, dtype=torch.long).flatten()
            if d_values.numel() == 1:
                d_values = d_values.expand(batch_size)
            elif d_values.numel() != batch_size:
                raise ValueError("relaxation_d tensor must have one value or batch_size values.")
        else:
            d_values = torch.full((batch_size,), int(relaxation_d), device=device, dtype=torch.long)

        if (d_values < 0).any() or (d_values > self.num_groups - 1).any():
            raise ValueError("relaxation_d values must be in [0, num_groups - 1].")
        return d_values

    def _get_travel_distance(self):
        batch_size = self.batch_size
        selected_node_list = self.selected_node_list
        node_xy = self.node_xy
        if batch_size is None or selected_node_list is None or node_xy is None:
            raise RuntimeError("reset must be called before calculating travel distance.")

        # CHANGE: 使用分离后的self.node_xy而不是self.problems（因为problems现在包含3维）
        gathering_index = selected_node_list.unsqueeze(dim=3).expand(batch_size, -1, self.problem_size, 2)
        # shape: (batch, pomo, problem, 2)
        
        # CHANGE: 使用self.node_xy（仅坐标部分）
        seq_expanded = node_xy[:, None, :, :].expand(batch_size, self.pomo_size, self.problem_size, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(dim=3).sqrt()
        # shape: (batch, pomo, problem)

        travel_distances = segment_lengths.sum(dim=2)
        # shape: (batch, pomo)
        return travel_distances
