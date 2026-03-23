
import sys
import os

# 路径设置（保持你现在的做法）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from CTSPd_Env import CTSPdEnv  # 确保类名匹配你的实际类名

def test_ctspd_constraints():
    """验证d-松弛优先级规则的核心测试"""
    
    print("=" * 70)
    print("CTSP-d 约束规则验证测试")
    print("=" * 70)
    
    # 配置：小规模问题，便于观察
    problem_size = 6
    num_groups = 3  # 优先级 1, 2, 3
    batch_size = 1
    pomo_size = 1   # 单轨迹，便于跟踪
    
    # 构造确定性的测试数据
    test_xy = torch.rand(batch_size, problem_size, 2)
    test_priority = torch.tensor([[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]])  # 节点0-1:优先级1, 2-3:优先级2, 4-5:优先级3
    
    # ========== 测试场景1：严格优先级 (d=0) ==========
    print("\n【测试1】d=0（严格优先级）：必须按 1->2->3 顺序访问")
    print("-" * 70)
    
    env = CTSPdEnv(problem_size=problem_size, pomo_size=pomo_size, 
                   num_groups=num_groups, relaxation_d=0)
    env.batch_size = batch_size
    env.node_xy = test_xy
    env.node_priorities = test_priority
    env.d = 0
    env.BATCH_IDX = torch.arange(batch_size)[:, None].expand(batch_size, pomo_size)
    env.POMO_IDX = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
    env.problems = torch.cat([test_xy, test_priority.unsqueeze(2).float()], dim=2)
    
    # CHANGE 1: reset() 返回 Reset_State，需要调用 pre_step() 获取 Step_State
    reset_state, _, _ = env.reset()  # reset_state 只有 problems，没有 mask
    state, _, _ = env.pre_step()     # state 是 Step_State，包含 ninf_mask
    
    # 现在可以访问 mask 了
    initial_mask = state.ninf_mask[0, 0]  # (node,)
    legal_nodes = (initial_mask == 0).nonzero(as_tuple=True)[0].tolist()
    print(f"初始状态 - p_min=1, d=0")
    print(f"合法节点索引: {legal_nodes} (应为 [0, 1]，即优先级1的节点)")
    
    # 验证：只有优先级1的节点（索引0,1）应该是合法的
    assert set(legal_nodes) == {0, 1}, f"错误：初始合法节点应为0,1，得到{legal_nodes}"
    print("✓ 初始掩码正确")
    
    # CHANGE 2: step() 返回的就是 Step_State，可以直接使用
    state, _, done = env.step(torch.tensor([[0]]))  # 选择节点0（优先级1）
    print(f"\n选择节点0（优先级1）后")
    
    # 现在还有优先级1的节点1未访问，p_min应该仍为1
    legal_nodes = (state.ninf_mask[0, 0] == 0).nonzero(as_tuple=True)[0].tolist()
    print(f"合法节点: {legal_nodes} (应包含节点1，以及可能的其他...)")
    
    # 继续选完所有优先级1的节点
    state, _, _ = env.step(torch.tensor([[1]]))  # 选择节点1
    print(f"\n选择节点1后 - 当前已访问: [0,1]")
    print(f"p_min 应该更新为 2")
    
    # 现在应该只允许选优先级2（节点2,3）
    legal_nodes = (state.ninf_mask[0, 0] == 0).nonzero(as_tuple=True)[0].tolist()
    print(f"合法节点: {legal_nodes} (应为 [2, 3]，即优先级2的节点)")
    assert set(legal_nodes) == {2, 3}, f"错误：访问完优先级1后，应只能选优先级2，得到{legal_nodes}"
    
    print("✓ 测试1通过：严格优先级约束工作正常")
    
    # ========== 测试场景2：松弛优先级 (d=1) ==========
    print("\n【测试2】d=1（松弛优先级）：允许访问 p 到 p+1")
    print("-" * 70)
    
    env2 = CTSPdEnv(problem_size=problem_size, pomo_size=pomo_size,
                    num_groups=num_groups, relaxation_d=1)
    env2.batch_size = batch_size
    env2.node_xy = test_xy
    env2.node_priorities = test_priority
    env2.d = 1
    env2.BATCH_IDX = env.BATCH_IDX.clone()
    env2.POMO_IDX = env.POMO_IDX.clone()
    env2.problems = env.problems.clone()
    
    env2.reset()
    state, _, _ = env2.pre_step()  # CHANGE: 同样需要 pre_step()
    
    # 初始p_min=1, d=1，应该允许优先级1和2（节点0,1,2,3）
    legal_nodes = (state.ninf_mask[0, 0] == 0).nonzero(as_tuple=True)[0].tolist()
    print(f"初始状态 - p_min=1, d=1")
    print(f"合法节点: {legal_nodes} (应为 [0,1,2,3]，即优先级1和2)")
    
    assert set(legal_nodes) == {0, 1, 2, 3}, f"错误：d=1时初始应允许优先级1-2，得到{legal_nodes}"
    print("✓ 测试2通过：松弛优先级约束工作正常")
    
    # ========== 测试场景3：p_min动态更新 ==========
    print("\n【测试3】p_min动态更新验证")
    print("-" * 70)
    
    env3 = CTSPdEnv(problem_size=problem_size, pomo_size=pomo_size,
                    num_groups=num_groups, relaxation_d=1)
    env3.batch_size = batch_size
    env3.node_xy = test_xy
    env3.node_priorities = test_priority
    env3.d = 1
    env3.BATCH_IDX = env.BATCH_IDX.clone()
    env3.POMO_IDX = env.POMO_IDX.clone()
    env3.problems = env.problems.clone()
    
    env3.reset()
    env3.pre_step()
    
    # 先访问所有优先级1的节点
    env3.step(torch.tensor([[0]]))
    env3.step(torch.tensor([[1]]))
    print("已访问所有优先级1的节点 [0,1]")
    assert env3.current_min_priority[0, 0] == 2, f"p_min应为2，得到{env3.current_min_priority[0, 0]}"
    print(f"✓ p_min正确更新为: {env3.current_min_priority[0, 0]}")
    
    # 再访问所有优先级2的节点
    state, _, _ = env3.step(torch.tensor([[2]]))
    state, _, _ = env3.step(torch.tensor([[3]]))
    print("已访问所有优先级2的节点 [2,3]")
    assert env3.current_min_priority[0, 0] == 3, f"p_min应为3，得到{env3.current_min_priority[0, 0]}"
    print(f"✓ p_min正确更新为: {env3.current_min_priority[0, 0]}")
    
    # 验证现在只能选优先级3
    legal_nodes = (state.ninf_mask[0, 0] == 0).nonzero(as_tuple=True)[0].tolist()
    assert set(legal_nodes) == {4, 5}, f"最后应只能选优先级3的节点[4,5]，得到{legal_nodes}"
    print("✓ 测试3通过：p_min动态更新正常")
    
    print("\n" + "=" * 70)
    print("所有验证测试通过！CTSP-d约束逻辑工作正常。")
    print("=" * 70)


if __name__ == "__main__":
    test_ctspd_constraints()