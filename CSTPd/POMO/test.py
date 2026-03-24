import torch
import numpy as np
import os
import sys
import re

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

from CTSPd_Env import CTSPdEnv
from CTSPd_Model import CTSPdModel
from CTSPd_ProblemDef import augment_xy_data_by_8_fold  # 复用之前的数据增强
from sklearn.manifold import MDS  # pip install scikit-learn

# ==========================================
# TODO: 修改以下三个路径为你的实际路径
# ==========================================
MODEL_DIR = './result/20260324_223155_train__ctspd_n20'  # 模型文件夹路径
INSTANCE_FILE = 'C:/Users/lenovo/Desktop/graduation assignment/CTSP-d(用于对照的最优结果)/INSTANCES/Random_small'         # 测试文件路径
CHECKPOINT_EPOCH = 20                                    # 使用第几轮模型

def load_ctspd_simple(filepath):
    """简化版数据加载（复用之前提供的逻辑）"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    n = int(re.search(r'DIMENSION\s*:\s*(\d+)', content).group(1))
    d = int(re.search(r'RELAXATION_LEVEL\s*:\s*(\d+)', content).group(1))
    groups = int(re.search(r'GROUPS\s*:\s*(\d+)', content).group(1))
    
    # 距离矩阵
    matrix_vals = list(map(float, re.search(r'EDGE_WEIGHT_SECTION\s*\n(.*?)\nGROUP_SECTION', 
                           content, re.DOTALL).group(1).split()))
    dist_matrix = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            dist_matrix[i,j] = dist_matrix[j,i] = matrix_vals[idx]
            idx += 1
    
    # MDS还原坐标
    coords = MDS(n_components=2, dissimilarity='precomputed', random_state=42).fit_transform(dist_matrix)
    coords = (coords - coords.min()) / (coords.max() - coords.min() + 1e-8)
    
    # 解析优先级
    priorities = np.ones(n)
    grp_txt = re.search(r'GROUP_SECTION\s*\n(.*?)\nEOF', content, re.DOTALL).group(1)
    for line in grp_txt.strip().split('\n'):
        parts = list(map(int, line.split()))
        if len(parts) > 1:
            prio, nodes = parts[0], parts[1:-1] if parts[-1]==-1 else parts[1:]
            for nd in nodes:
                priorities[nd-1] = prio
    
    # 组装为 (1, n, 3) 张量
    problems = torch.zeros(1, n, 3)
    problems[0, :, :2] = torch.from_numpy(coords).float()
    problems[0, :, 2] = torch.from_numpy(priorities).float()
    
    return problems, torch.from_numpy(dist_matrix).float(), d, groups, n

def calc_real_length(tour, dist_matrix):
    """使用原始距离矩阵计算路径长度"""
    return sum(dist_matrix[tour[i], tour[(i+1)%len(tour)]].item() for i in range(len(tour)))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 加载数据
    print(f"\n加载: {INSTANCE_FILE}")
    problems, raw_dist, d, groups, n = load_ctspd_simple(INSTANCE_FILE)
    print(f"节点数: {n}, 松弛度 d: {d}, 优先级组数: {groups}")
    
    # 2. 加载模型
    model_path = f"{MODEL_DIR}/checkpoint-{CHECKPOINT_EPOCH}.pt"
    print(f"加载模型: {model_path}")
    
    model = CTSPdModel(
        embedding_dim=128, encoder_layer_num=6, qkv_dim=16, head_num=8,
        logit_clipping=10, ff_hidden_dim=512, eval_type='argmax',
        sqrt_embedding_dim=11.3137
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 3. 配置环境（动态适配52节点）
    env = CTSPdEnv(problem_size=n, pomo_size=n, num_groups=groups, relaxation_d=d)
    problems = problems.to(device)
    
    # 4. 推理（带8倍增强）
    print("\n开始推理...")
    with torch.no_grad():
        # 8倍增强
        probs_aug = augment_xy_data_by_8_fold(problems)
        env.load_problems(8, problems=probs_aug)
        
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)
        
        state, reward, done = env.pre_step()
        while not done:
            selected, _ = model(state)
            state, reward, done = env.step(selected)
    
    # 5. 计算结果
    best_reward = reward.max()
    best_idx = reward.argmax()
    
    # 获取最优路径
    tour = env.selected_node_list[best_idx//n, best_idx%n].cpu().tolist()
    
    # 关键：用原始距离矩阵计算真实长度
    real_length = calc_real_length(tour, raw_dist)
    
    print("\n" + "="*50)
    print("测试结果")
    print("="*50)
    print(f"归一化坐标路径长度: {-best_reward.item():.4f}")
    print(f"真实路径长度: {real_length:.2f}")
    print(f"LKH最优解: 7542")
    print(f"Gap: {((real_length-7542)/7542)*100:.2f}%")
    print("="*50)

if __name__ == '__main__':
    main()