import torch
import numpy as np
import os
import re
import json
from logging import getLogger
from glob import glob

from CTSPd_Env import CTSPdEnv as Env
from CTSPd_Model import CTSPdModel as Model
from CTSPd_ProblemDef import get_random_problems, augment_xy_data_by_8_fold
from utils.utils import *


class CTSPdTester:
    def __init__(self, env_params, model_params, tester_params):
        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        # cuda setup
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params).to(device)

        # Restore model
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f'Model loaded from {checkpoint_fullname}')

        # utility
        self.time_estimator = TimeEstimator()
        
        # For file-based testing
        self.test_data_dir = tester_params.get('test_data_dir', None)
        self.test_files = []
        if self.test_data_dir and os.path.exists(self.test_data_dir):
            pattern = tester_params.get('test_file_pattern', '*.ctspd')
            self.test_files = sorted(glob(os.path.join(self.test_data_dir, pattern)))
            self.logger.info(f'Found {len(self.test_files)} test files in {self.test_data_dir}')

    def run(self):
        """Main test loop - compatible with original test_n20.py"""
        self.time_estimator.reset()

        # File-based testing (new feature for CTSP-d)
        if self.test_files:
            self.logger.info(f'*** Testing on {len(self.test_files)} CTSP-d instances ***')
            self._test_from_files()
        else:
            # Random testing (original TSP style)
            self.logger.info(f'*** Testing on random instances ***')
            self._test_random()

    def _test_random(self):
        """Original random testing mode"""
        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        test_num_episode = self.tester_params['test_episodes']
        episode = 0

        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score, aug_score = self._test_one_batch(batch_size)

            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)

            episode += batch_size

            # Logs
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))

    def _test_from_files(self):
        """Test on .ctspd files - calculates real tour length using distance matrix"""
        results = []
        
        for idx, filepath in enumerate(self.test_files):
            filename = os.path.basename(filepath)
            self.logger.info(f'Testing {idx+1}/{len(self.test_files)}: {filename}')
            
            try:
                result = self._test_single_file(filepath)
                results.append(result)
                
                self.logger.info(f'  Real Length: {result["real_length"]:.2f}, Time: {result["time"]:.3f}s')
                
                # Save intermediate results
                if (idx + 1) % 10 == 0:
                    self._save_file_results(results)
                    
            except Exception as e:
                self.logger.error(f'Failed to test {filename}: {str(e)}')
                continue
        
        # Final summary
        self._save_file_results(results)
        self._print_summary(results)
    
    def _test_single_file(self, filepath):
        """Test single .ctspd file"""
        # Parse file
        problems, raw_dist_matrix, d, num_groups = self._load_ctspd_file(filepath)
        n_nodes = problems.size(1)
        
        # Update env params dynamically
        env_params = self.env_params.copy()
        env_params['problem_size'] = n_nodes
        env_params['pomo_size'] = n_nodes
        env_params['relaxation_d'] = d
        env_params['num_groups'] = num_groups
        
        # Create temporary env with correct size
        env = Env(**env_params)
        
        # Test with and without augmentation
        aug_factor = self.tester_params.get('aug_factor', 8) if self.tester_params.get('augmentation_enable', False) else 1
        
        # Move data to device
        problems = problems.to(self.device)
        raw_dist_matrix = raw_dist_matrix.to(self.device)
        
        if aug_factor > 1:
            problems = augment_xy_data_by_8_fold(problems)
        
        # Load problems
        env.load_problems(1, aug_factor=aug_factor, problems=problems)
        
        # Inference
        start_time = torch.cuda.Event(enable_timing=True) if self.tester_params['use_cuda'] else None
        end_time = torch.cuda.Event(enable_timing=True) if self.tester_params['use_cuda'] else None
        
        if start_time:
            start_time.record()
        else:
            import time
            t0 = time.time()
        
        with torch.no_grad():
            self.model.eval()
            reset_state, _, _ = env.reset()
            self.model.pre_forward(reset_state)
            
            state, reward, done = env.pre_step()
            while not done:
                selected, _ = self.model(state)
                state, reward, done = env.step(selected)
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time) / 1000.0
        else:
            inference_time = time.time() - t0
        
        # Get tour and calculate real length
        tour = self._get_tour_from_env(env, aug_factor)
        
        # Calculate real path length using original distance matrix
        if aug_factor == 1:
            real_length = self._calculate_real_length(tour, raw_dist_matrix)
            best_length = real_length
        else:
            # For augmentation, calculate length for each augmented version
            # Simplified: just use the best reward index
            best_idx = reward.argmax().item()
            best_tour = tour[best_idx] if isinstance(tour, list) else tour
            best_length = self._calculate_real_length(best_tour, raw_dist_matrix)
        
        return {
            'filename': os.path.basename(filepath),
            'n_nodes': n_nodes,
            'd': d,
            'real_length': best_length,
            'time': inference_time,
            'predicted_length': -reward.max().item()
        }

    def _test_one_batch(self, batch_size):
        """Original batch testing - random instances"""
        # Augmentation
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        self.model.eval()
        with torch.no_grad():
            # Generate random CTSP-d problems
            num_groups = self.env_params.get('num_groups', 3)
            problems = get_random_problems(batch_size, self.env_params['problem_size'], num_groups)
            problems = problems.to(self.device)
            
            if aug_factor > 1:
                problems = augment_xy_data_by_8_fold(problems)
                batch_size = batch_size * aug_factor
            
            self.env.load_problems(batch_size, aug_factor=1, problems=problems)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # POMO Rollout
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            state, reward, done = self.env.step(selected)

        # Return scores
        batch_size_orig = batch_size // aug_factor if aug_factor > 1 else batch_size
        aug_reward = reward.reshape(aug_factor, batch_size_orig, self.env.pomo_size)
        
        max_pomo_reward, _ = aug_reward.max(dim=2)
        no_aug_score = -max_pomo_reward[0, :].float().mean()
        
        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)
        aug_score = -max_aug_pomo_reward.float().mean()

        return no_aug_score.item(), aug_score.item()

    def _load_ctspd_file(self, filename):
        """Load .ctspd file and return problems tensor, distance matrix, d, num_groups"""
        with open(filename, 'r') as f:
            content = f.read()
        
        # Parse header
        dimension = int(re.search(r'DIMENSION\s*:\s*(\d+)', content).group(1))
        num_groups = int(re.search(r'GROUPS\s*:\s*(\d+)', content).group(1))
        relaxation_d = int(re.search(r'RELAXATION_LEVEL\s*:\s*(\d+)', content).group(1))
        
        # Parse distance matrix
        matrix_section = re.search(r'EDGE_WEIGHT_SECTION\s*\n(.*?)\n\s*(?:GROUP_SECTION|EOF)', 
                                   content, re.DOTALL)
        matrix_values = list(map(float, matrix_section.group(1).split()))
        
        dist_matrix = torch.zeros((dimension, dimension))
        idx = 0
        for i in range(dimension):
            for j in range(i+1, dimension):
                dist_matrix[i, j] = matrix_values[idx]
                dist_matrix[j, i] = matrix_values[idx]
                idx += 1
        
        # MDS to coordinates
        from sklearn.manifold import MDS
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress=False)
        coords = mds.fit_transform(dist_matrix.numpy())
        
        # Normalize to [0, 1]
        coords = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0) + 1e-8)
        
        # Parse priorities
        priorities = torch.ones(dimension)
        group_section = re.search(r'GROUP_SECTION\s*\n(.*?)\n\s*EOF', content, re.DOTALL)
        if group_section:
            lines = group_section.group(1).strip().split('\n')
            for line in lines:
                parts = list(map(int, line.split()))
                if len(parts) >= 2 and parts[0] != -1:
                    group_id = parts[0]
                    nodes = parts[1:-1] if parts[-1] == -1 else parts[1:]
                    for node_idx in nodes:
                        if 1 <= node_idx <= dimension:
                            priorities[node_idx - 1] = float(group_id)
        
        # Create problems tensor (batch=1, node, 3)
        problems = torch.zeros(1, dimension, 3)
        problems[0, :, :2] = torch.from_numpy(coords).float()
        problems[0, :, 2] = priorities
        
        return problems, dist_matrix, relaxation_d, num_groups

    def _get_tour_from_env(self, env, aug_factor):
        """Extract tour from environment"""
        # Get selected_node_list from env
        # Shape: (batch * aug_factor, pomo, problem) or similar
        tours = env.selected_node_list.cpu().numpy()
        
        if aug_factor == 1:
            return tours[0, 0].tolist()  # First batch, first pomo
        else:
            # Return list of tours for each augmentation
            return [tours[i, 0].tolist() for i in range(tours.shape[0])]
    
    def _calculate_real_length(self, tour, dist_matrix):
        """Calculate tour length using original distance matrix"""
        length = 0.0
        n = len(tour)
        for i in range(n):
            from_node = tour[i]
            to_node = tour[(i+1) % n]
            length += dist_matrix[from_node, to_node].item()
        return length
    
    def _save_file_results(self, results):
        """Save results to JSON"""
        result_file = os.path.join(self.result_folder, 'test_results.json')
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f'Results saved to {result_file}')
    
    def _print_summary(self, results):
        """Print summary statistics"""
        if not results:
            return
        
        lengths = [r['real_length'] for r in results]
        times = [r['time'] for r in results]
        
        self.logger.info("\n" + "="*60)
        self.logger.info("CTSP-d Test Summary")
        self.logger.info("="*60)
        self.logger.info(f"Instances tested: {len(results)}")
        self.logger.info(f"Average Length: {np.mean(lengths):.2f}")
        self.logger.info(f"Std Length: {np.std(lengths):.2f}")
        self.logger.info(f"Average Time: {np.mean(times):.3f}s")
        self.logger.info(f"Total Time: {np.sum(times):.2f}s")
        self.logger.info("="*60)