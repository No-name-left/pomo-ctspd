
from dataclasses import dataclass
import torch

from CTSPd_ProblemDef import get_random_problems, augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 3)，因为加入了优先级


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)


class CTSPdEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
         # CHANGE: 新增CTSP-d参数读取
        self.num_groups = env_params.get('num_groups', 3)  # 优先级组数
        self.d = env_params.get('relaxation_d', 1)         # 松弛度d

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems = None
        # shape: (batch, node, 3)  # [x, y, priority]
        
        # CHANGE: 新增成员变量，用于存储分离后的数据
        self.node_xy = None           # shape: (batch, node, 2)
        self.node_priorities = None   # shape: (batch, node)

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)

        # CHANGE: 新增动态状态变量
        self.visited_mask = None      # shape: (batch, pomo, node), bool
        self.current_min_priority = None  # shape: (batch, pomo), 当前未访问节点中的最高优先级

    def load_problems(self, batch_size, aug_factor=1, problems=None):
        """
        加载问题数据
        Args:
            batch_size: 批次大小
            aug_factor: 数据增强倍数（1或8）
            problems: 可选，外部传入的 (batch, node, 3) 数据，用于测试
        """
        if problems is None:
            # 训练模式：随机生成
            self.batch_size = batch_size
            self.problems = get_random_problems(batch_size, self.problem_size, self.num_groups)
        else:
            # 测试模式：使用传入的数据
            self.problems = problems
            self.batch_size = problems.size(0)
        
        # 数据增强
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                self.problems = augment_xy_data_by_8_fold(self.problems)
            else:
                raise NotImplementedError
        
        # 分离坐标和优先级
        self.node_xy = self.problems[:, :, :2]
        self.node_priorities = self.problems[:, :, 2]
        
        # 初始化索引
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def load_problems_from_file(self, problems_tensor, d):
        """
        从文件加载的数据初始化环境（用于测试）
        
        Args:
            problems_tensor: (batch=1, problem, 3) 包含归一化坐标和优先级
            d: 松弛度
        """
        self.batch_size = 1
        self.problems = problems_tensor
        self.node_xy = self.problems[:, :, :2]        # (1, n, 2)
        self.node_priorities = self.problems[:, :, 2]  # (1, n)
        self.d = d
        
        # 重新初始化索引
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def get_current_tour(self):
        """
        获取当前已选择的节点访问顺序
        用于测试时记录路径
        """
        return self.selected_node_list[0, 0].cpu().numpy().tolist()  # 返回第一个batch第一个pomo的路径

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)

        # CHANGE: 初始化访问掩码（False表示未访问）
        self.visited_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size), dtype=torch.bool)

        # CHANGE: 计算初始current_min_priority（全局最小优先级）
        # 在未访问节点中找最小值（初始时所有节点都未访问）
        global_min = self.node_priorities.min(dim=1, keepdim=True)[0]  # (batch, 1)
        self.current_min_priority = global_min.expand(self.batch_size, self.pomo_size)  # (batch, pomo)

        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size))
        # CHANGE: 新增 - 应用d-松弛优先级规则（覆盖上面的零张量，屏蔽不符合优先级范围的节点）
        self._apply_priority_mask()
        self.group_ids = self.node_priorities.long()  # (batch, problem)，用于传入模型

        reward = None
        done = False
        return Reset_State(self.problems), reward, done

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)

        # UPDATE STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        # CHANGE: 更新访问掩码（标记已访问节点）
        self.visited_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = True
        
        # CHANGE: 更新ninf_mask的基础部分（已访问节点设为-inf）
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')
        # 注意：这里我们保留原有的"已访问节点置-inf"逻辑，但接下来会重新计算完整的优先级掩码

        # CHANGE: 重新计算current_min_priority（在未访问节点中找最小优先级）
        if self.selected_count < self.problem_size:
            # 将已访问节点的优先级设为无穷大，这样min不会选到它们
            # 扩展维度以匹配visited_mask: (batch, node) -> (batch, pomo, node)
            expanded_priorities = self.node_priorities.float().unsqueeze(1).expand(-1, self.pomo_size, -1).clone()
            expanded_priorities[self.visited_mask] = float('inf')
            # 在未访问节点中找最小值（即最高优先级）
            self.current_min_priority = expanded_priorities.min(dim=2)[0]  # (batch, pomo)

         # CHANGE: 更新Step_State中的current_min_priority（可选）
        if hasattr(self.step_state, 'current_min_priority'):
            self.step_state.current_min_priority = self.current_min_priority
        
        # CHANGE: 重新应用d-松弛优先级掩码（这覆盖了原有的简单mask更新）
        self._apply_priority_mask()

        # returning values
        done = (self.selected_count == self.problem_size)
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

     # CHANGE: 新增核心方法 - 应用d-松弛优先级掩码
    def _apply_priority_mask(self):
        """
        根据d-松弛优先级规则更新ninf_mask。
        规则:设p为未访问节点中的最高优先级(数值最小),
        则只能访问优先级在[p, p+d]范围内的节点。
        """
        # p_min: (batch, pomo) -> (batch, pomo, 1) 用于广播
        p_min = self.current_min_priority.unsqueeze(2)
        p_max = p_min + self.d  # 允许的最高优先级
        
        # 所有节点优先级: (batch, node) -> (batch, 1, node) 用于广播
        priorities = self.node_priorities.unsqueeze(1)
        
        # 非法条件：优先级 < p_min（更高的优先级未访问完）或 > p_max（超出松弛范围）
        illegal_low = priorities < p_min   # 优先级比当前最小值还小（数值更小=更紧急）
        illegal_high = priorities > p_max  # 优先级超出松弛窗口
        illegal_priority = illegal_low | illegal_high
        
        # 合并非法条件：已访问 或 优先级不符合要求
        combined_illegal = self.visited_mask | illegal_priority
        
        # 更新mask：非法位置为-inf，合法位置保持原值（或设为0）
        # 注意：我们需要保留之前已访问节点的-inf标记，所以这里重新计算完整mask
        self.step_state.ninf_mask = torch.zeros_like(self.visited_mask, dtype=torch.float)
        self.step_state.ninf_mask.masked_fill_(combined_illegal, float('-inf'))

    def _get_travel_distance(self):
        # CHANGE: 使用分离后的self.node_xy而不是self.problems（因为problems现在包含3维）
        gathering_index = self.selected_node_list.unsqueeze(3).expand(self.batch_size, -1, self.problem_size, 2)
        # shape: (batch, pomo, problem, 2)
        
        # CHANGE: 使用self.node_xy（仅坐标部分）
        seq_expanded = self.node_xy[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, problem)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances

