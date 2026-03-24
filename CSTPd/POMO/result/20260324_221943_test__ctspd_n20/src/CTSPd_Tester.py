import torch
import numpy as np
import os
import time
from datetime import datetime
from glob import glob

# 导入你的模块（根据实际路径调整）
from CTSPd_Env import CTSPdEnv
from CTSPd_Model import CTSPdModel  # 假设你重命名了模型类
from CTSPd_ProblemDef import load_ctspd_instance, load_ctspd_tour


class CTSPdTester:
    def __init__(self, env_params, model_params, tester_params):
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params
        
        # 选择设备
        self.device = torch.device('cuda' if tester_params['use_cuda'] else 'cpu')
        
        # 初始化模型（需要加载训练好的权重）
        self.model = CTSPdModel(**model_params).to(self.device)
        self.model.eval()
        self._load_model(tester_params['model_load'])  # 传入整个字典，不是仅传 path
            # 创建结果文件夹
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_folder = os.path.join('./result', f'{time_stamp}_test__ctspd_n{env_params["problem_size"]}')
        os.makedirs(self.result_folder, exist_ok=True)
        os.makedirs(os.path.join(self.result_folder, 'img'), exist_ok=True)
        
        # 日志文件路径
        self.log_file = os.path.join(self.result_folder, 'run_log.txt')
        
        print(f"结果将保存到: {self.result_folder}")
        
        
    def _load_model(self, load_config):
        """
        加载训练好的模型权重
        load_config: dict with 'path' (directory) and 'epoch' (int)
        """
        checkpoint_dir = load_config['path']
        epoch = load_config['epoch']
        
        # 自动拼接成标准格式：checkpoint-{epoch}.pt
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint-{epoch}.pt')
        
        # 如果上面没找到，尝试 epoch-{epoch}.pt（兼容你之前的错误配置）
        if not os.path.exists(checkpoint_path):
            alt_path = os.path.join(checkpoint_dir, f'epoch-{epoch}.pt')
            if os.path.exists(alt_path):
                checkpoint_path = alt_path
                print(f"Warning: 使用非标准文件名格式 {alt_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"模型文件不存在: {checkpoint_path} (请确认训练保存的文件名是 checkpoint-{epoch}.pt 还是 epoch-{epoch}.pt)")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载模型: {checkpoint_path} (Epoch {checkpoint.get('epoch', 'unknown')})")
    
    def test_single_file(self, filename, use_aug=True):
        """
        测试单个 .ctspd 文件
        
        Args:
            filename: .ctspd 文件路径
            use_aug: 是否使用8倍数据增强（POMO特性）
        
        Returns:
            dict: 包含路径长度、时间、访问顺序等结果
        """
        # 1. 加载数据
        problems, priorities, d, raw_dist_matrix = load_ctspd_instance(filename)
        n_nodes = problems.size(1)
        
        print(f"\n测试文件: {os.path.basename(filename)}")
        print(f"节点数: {n_nodes}, 松弛度 d: {d}, 优先级组数: {int(priorities.max())}")
        
        # 2. 配置环境（动态适配节点数）
        env_params = self.env_params.copy()
        env_params['problem_size'] = n_nodes
        env_params['relaxation_d'] = d
        env_params['num_groups'] = int(priorities.max())
        
        # POMO size 通常等于节点数（从每个节点出发）
        env_params['pomo_size'] = n_nodes
        
        env = CTSPdEnv(**env_params)
        
        # 3. 数据增强（8倍旋转/翻转）
        if use_aug:
            from CTSPd_ProblemDef import augment_xy_data_by_8_fold
            aug_problems = augment_xy_data_by_8_fold(problems)
            env.load_problems(8, problems=aug_problems[0], priorities=aug_problems[0, :, 2], d=d)
            # 注意：增强后需要调整 raw_dist_matrix 的使用方式
            # 实际上增强后应该用对应的原始矩阵，这里简化处理：只用第一组做距离计算
            n_aug = 8
        else:
            env.load_problems_from_file(problems, d)
            n_aug = 1
        
        # 4. 模型推理
        start_time = time.time()
        
        with torch.no_grad():
            # 编码
            reset_state, _, _ = env.reset()
            self.model.pre_forward(reset_state)
            
            # POMO 推理
            state, reward, done = env.pre_step()
            while not done:
                selected, _ = self.model(state)
                state, reward, done = env.step(selected)
        
        inference_time = time.time() - start_time
        
        # 5. 处理结果
        # reward: (batch=1 or 8, pomo=n_nodes)
        # 取最好的结果（最短路径）
        best_reward = reward.max() if use_aug else reward[0].max()
        best_pomo_idx = reward.argmax() if use_aug else reward[0].argmax()
        
        # 获取最优路径的节点顺序（需要重新运行一次来记录路径，或修改Env保存路径）
        # 这里简化：使用 MDS 坐标计算的路径长度作为参考
        # 实际应该根据 selected_node_list 获取访问顺序
        
        # 6. 关键：用原始距离矩阵计算真实路径长度
        # 需要从 Env 获取访问顺序
        tour = env.get_current_tour()  # 这需要在 Env 中实现
        
        # 计算真实路径长度（基于原始距离矩阵）
        real_length = 0
        for i in range(len(tour)):
            from_node = tour[i]
            to_node = tour[(i+1) % len(tour)]
            real_length += raw_dist_matrix[from_node, to_node].item()
        
        result = {
            'filename': os.path.basename(filename),
            'n_nodes': n_nodes,
            'd': d,
            'predicted_length': -best_reward.item(),  # 归一化坐标下的长度（参考）
            'real_length': real_length,  # 基于原始距离矩阵的长度（与LKH对比）
            'time': inference_time,
            'tour': tour
        }
        
        print(f"预测路径长度: {real_length:.2f}")
        print(f"推理时间: {inference_time:.3f}s")
        
        return result
    
    def test_directory(self, data_dir, pattern='*.ctspd'):
        """
        测试目录下所有匹配的文件（适配不同规模）
        
        Args:
            data_dir: 测试数据目录
            pattern: 文件匹配模式，如 'berlin52*.ctspd' 或 '*.ctspd'
        """
        files = glob(os.path.join(data_dir, pattern))
        if not files:
            print(f"未找到匹配文件: {os.path.join(data_dir, pattern)}")
            return []
        
        print(f"找到 {len(files)} 个测试文件")
        
        results = []
        for filepath in sorted(files):
            try:
                result = self.test_single_file(filepath, use_aug=self.tester_params.get('use_aug', True))
                results.append(result)
            except Exception as e:
                print(f"测试失败 {filepath}: {str(e)}")
                continue
        
        # 汇总统计
        if results:
            print("\n" + "="*60)
            print("测试汇总")
            print("="*60)
            avg_time = np.mean([r['time'] for r in results])
            print(f"平均推理时间: {avg_time:.3f}s")
            print(f"成功测试: {len(results)}/{len(files)}")
            
            # 如果有 BKS 文件可以计算 Gap
            
        return results