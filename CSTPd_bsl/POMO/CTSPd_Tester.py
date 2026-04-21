import torch
import numpy as np
import os
import json
import time
from pathlib import Path
from logging import getLogger
from glob import glob

from CSTPd_bsl.POMO.CTSPd_Env import CTSPdEnv as Env
from CSTPd_bsl.POMO.CTSPd_Model import CTSPdModel as Model
from CSTPd_bsl.CTSPd_ProblemDef import get_random_problems, parse_ctspd_file
from utils.utils import AverageMeter, TimeEstimator, get_result_folder


class CTSPdTester:
    def __init__(self, env_params, model_params, tester_params):
        # save arguments
        self.env_params = env_params
        self.model_params = dict(model_params)
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        # cuda setup
        USE_CUDA = bool(self.tester_params['use_cuda'])
        cuda_device_num = int(self.tester_params.get('cuda_device_num', 0))
        if USE_CUDA:
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
        else:
            device = torch.device('cpu')
        torch.set_default_dtype(torch.float32)
        set_default_device = getattr(torch, 'set_default_device', None)
        if set_default_device is not None:
            set_default_device(device)
        self.device = device

        # Restore model
        model_load = tester_params['model_load']
        checkpoint_fullname = self._resolve_checkpoint_fullname(model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self._sync_model_params_with_checkpoint(checkpoint['model_state_dict'])

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f'Model loaded from {checkpoint_fullname}')

        # utility
        self.time_estimator = TimeEstimator()
        self._file_aug_warning_emitted = False
        
        # For file-based testing
        self.test_data_dir = tester_params.get('test_data_dir', None)
        self.test_files = []
        if self.test_data_dir and os.path.exists(self.test_data_dir):
            pattern = tester_params.get('test_file_pattern', '*.ctspd')
            self.test_files = sorted(glob(os.path.join(self.test_data_dir, pattern)))
            self.logger.info(f'Found {len(self.test_files)} test files in {self.test_data_dir}')

    def _resolve_checkpoint_fullname(self, model_load):
        checkpoint_name = f"checkpoint-{model_load['epoch']}.pt"
        configured_path = Path(model_load['path']) / checkpoint_name
        if configured_path.exists():
            return str(configured_path)

        search_root = Path(__file__).resolve().parent / 'result'
        matches = sorted(
            search_root.glob(f'**/{checkpoint_name}'),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if matches:
            fallback = matches[0]
            self.logger.warning(
                f"Configured checkpoint {configured_path} not found. "
                f"Using latest matching checkpoint: {fallback}"
            )
            return str(fallback)

        raise FileNotFoundError(
            f"Could not find {checkpoint_name}. Checked configured path {configured_path} "
            f"and searched under {search_root}."
        )

    def _sync_model_params_with_checkpoint(self, state_dict):
        group_embedding_key = 'encoder.group_embedding.weight'
        if group_embedding_key not in state_dict:
            return

        inferred_num_groups = int(state_dict[group_embedding_key].size(0)) - 1
        configured_num_groups = self.model_params.get('num_groups')
        if configured_num_groups != inferred_num_groups:
            self.logger.info(
                f"Adjusting model num_groups from {configured_num_groups} "
                f"to checkpoint value {inferred_num_groups}"
            )
            self.model_params['num_groups'] = inferred_num_groups

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
        problems, raw_dist_matrix, d, num_groups = parse_ctspd_file(filepath)
        n_nodes = int(problems.size(1))

        model_num_groups = self.model_params.get('num_groups')
        if model_num_groups is not None and num_groups > model_num_groups:
            raise ValueError(
                f"Instance uses {num_groups} groups, but the loaded model supports "
                f"only {model_num_groups} groups."
            )
        
        # Update env params dynamically
        env_params = self.env_params.copy()
        env_params['problem_size'] = n_nodes
        env_params['pomo_size'] = n_nodes
        env_params['relaxation_d'] = d
        env_params['num_groups'] = num_groups
        
        # Create temporary env with correct size
        env = Env(**env_params)
        
        # File-based CTSP-d instances only provide a distance matrix. The 2D coordinates are
        # reconstructed features, so geometric augmentation is not length-preserving here.
        requested_aug_factor = self.tester_params.get('aug_factor', 8) if self.tester_params.get('augmentation_enable', False) else 1
        aug_factor = 1
        if requested_aug_factor != 1 and not self._file_aug_warning_emitted:
            self.logger.warning(
                'Disabling geometric augmentation for file-based CTSP-d instances because '
                'the reconstructed 2D features do not preserve the original distance matrix.'
            )
            self._file_aug_warning_emitted = True
        
        # Move data to device
        problems = problems.to(self.device)
        raw_dist_matrix = raw_dist_matrix.to(self.device)

        # Load problems. Env handles augmentation internally.
        env.load_problems(1, aug_factor=aug_factor, problems=problems)
        
        # Inference
        start_time = torch.cuda.Event(enable_timing=True) if self.tester_params['use_cuda'] else None
        end_time = torch.cuda.Event(enable_timing=True) if self.tester_params['use_cuda'] else None
        t0 = None
        
        if start_time is not None:
            start_time.record()
        else:
            t0 = time.time()
        
        with torch.no_grad():
            self.model.eval()
            reset_state, _, _ = env.reset()
            self.model.pre_forward(reset_state)
            
            state, reward, done = env.pre_step()
            while not done:
                selected, _ = self.model(state)
                state, reward, done = env.step(selected)

        if reward is None:
            raise RuntimeError("Environment finished without producing a reward.")
        
        if start_time is not None and end_time is not None:
            end_time.record()
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time) / 1000.0
        else:
            if t0 is None:
                raise RuntimeError("CPU timer was not initialized.")
            inference_time = time.time() - t0
        
        best_tour, best_batch_idx, best_pomo_idx = self._get_best_tour(env, reward)
        best_length = self._calculate_real_length(best_tour, raw_dist_matrix)
        predicted_length = -float(reward[best_batch_idx, best_pomo_idx].item())
        
        return {
            'filename': os.path.basename(filepath),
            'n_nodes': n_nodes,
            'd': d,
            'real_length': best_length,
            'time': inference_time,
            'predicted_length': predicted_length,
            'best_aug_index': best_batch_idx,
            'best_pomo_index': best_pomo_idx,
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

            self.env.load_problems(batch_size, aug_factor=aug_factor, problems=problems)
            loaded_batch_size = self.env.batch_size
            if loaded_batch_size is None:
                raise RuntimeError("Environment did not set batch_size after load_problems.")
            batch_size = loaded_batch_size
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # POMO Rollout
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            state, reward, done = self.env.step(selected)

        if reward is None:
            raise RuntimeError("Environment finished without producing a reward.")

        # Return scores
        batch_size_orig = batch_size // aug_factor if aug_factor > 1 else batch_size
        aug_reward = reward.reshape(int(aug_factor), int(batch_size_orig), int(self.env.pomo_size))
        
        max_pomo_reward, _ = aug_reward.max(dim=2)
        no_aug_score = -max_pomo_reward[0, :].float().mean()
        
        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)
        aug_score = -max_aug_pomo_reward.float().mean()

        return float(no_aug_score.item()), float(aug_score.item())

    def _load_ctspd_file(self, filename):
        """Load .ctspd file and return problems tensor, distance matrix, d, num_groups"""
        return parse_ctspd_file(filename)

    def _get_best_tour(self, env, reward):
        best_flat_idx = int(reward.argmax().item())
        pomo_size = int(reward.size(1))
        best_batch_idx = best_flat_idx // pomo_size
        best_pomo_idx = best_flat_idx % pomo_size
        selected_node_list = env.selected_node_list
        if selected_node_list is None:
            raise RuntimeError("Environment has no selected tour.")
        best_tour = [int(node) for node in selected_node_list[best_batch_idx, best_pomo_idx].cpu().tolist()]
        return best_tour, best_batch_idx, best_pomo_idx
    
    def _calculate_real_length(self, tour, dist_matrix):
        """Calculate tour length using original distance matrix"""
        length = 0.0
        n = len(tour)
        for i in range(n):
            from_node = int(tour[i])
            to_node = int(tour[(i+1) % n])
            length += float(dist_matrix[from_node, to_node].item())
        return length
    
    def _save_file_results(self, results):
        """Save results to JSON"""
        result_file = os.path.join(self.result_folder, 'test_results.json')
        with open(result_file, 'w', encoding='utf-8') as f:
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
