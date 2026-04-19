
"""
The MIT License

Copyright (c) 2021 Yeong-Dae Kwon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import time
import sys
import os
from datetime import datetime
import logging
import logging.config
import subprocess
import numpy as np
import json
import shutil
from typing import Any, Dict, Optional, Set

os.environ.setdefault("MPLBACKEND", "Agg")

try:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    _PLOT_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on local install
    matplotlib = None
    plt = None
    _PLOT_IMPORT_ERROR = exc


def _get_process_start_time():
    return datetime.now().astimezone()


def _format_result_timestamp(moment=None):
    if moment is None:
        moment = _get_process_start_time()
    return "{}日_{}点{}分".format(
        moment.strftime("%d"),
        moment.strftime("%H"),
        moment.strftime("%M"),
    )


def _make_default_result_folder():
    return './result/' + _format_result_timestamp() + '{desc}'


def _make_unique_folder_path(folder):
    if not os.path.exists(folder):
        return folder

    suffix = 2
    while True:
        candidate = "{}_run{:02d}".format(folder, suffix)
        if not os.path.exists(candidate):
            return candidate
        suffix += 1


process_start_time = _get_process_start_time()
result_folder = None


def get_result_folder():
    global result_folder
    if result_folder is None:
        result_folder = _make_default_result_folder()
    return result_folder


def set_result_folder(folder):
    global result_folder
    result_folder = folder


def create_logger(log_file: Optional[Dict[str, Any]] = None):
    log_config = {} if log_file is None else dict(log_file)
    auto_filepath = 'filepath' not in log_config

    if auto_filepath:
        log_config['filepath'] = get_result_folder()

    if 'desc' in log_config:
        log_config['filepath'] = log_config['filepath'].format(desc='_' + log_config['desc'])
    else:
        log_config['filepath'] = log_config['filepath'].format(desc='')

    if auto_filepath:
        log_config['filepath'] = _make_unique_folder_path(log_config['filepath'])

    set_result_folder(log_config['filepath'])

    if 'filename' in log_config:
        filename = log_config['filepath'] + '/' + log_config['filename']
    else:
        filename = log_config['filepath'] + '/' + 'log.txt'

    if not os.path.exists(log_config['filepath']):
        os.makedirs(log_config['filepath'])

    file_mode = 'a' if os.path.isfile(filename)  else 'w'

    root_logger = logging.getLogger()
    root_logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(filename)s(%(lineno)d) : %(message)s", "%Y-%m-%d %H:%M:%S")

    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)

    # write to file
    fileout = logging.FileHandler(filename, mode=file_mode, encoding='utf-8')
    fileout.setLevel(logging.INFO)
    fileout.setFormatter(formatter)
    root_logger.addHandler(fileout)

    # write to console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root_logger.addHandler(console)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += float(val) * int(n)
        self.count += int(n)

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0


class LogData:
    def __init__(self):
        self.keys: Set[str] = set()
        self.data: Dict[str, list] = {}

    def get_raw_data(self):
        return self.keys, self.data

    def set_raw_data(self, r_data):
        self.keys, self.data = r_data

    def append_all(self, key, *args):
        if len(args) == 1:
            value = [list(range(len(args[0]))), args[0]]
        elif len(args) == 2:
            value = [args[0], args[1]]
        else:
            raise ValueError('Unsupported value type')

        rows = np.stack(value, axis=1).tolist()
        if key in self.keys:
            self.data[key].extend(rows)
        else:
            self.data[key] = rows
            self.keys.add(key)

    def append(self, key, *args):
        if len(args) == 1:
            args = args[0]

            if isinstance(args, int) or isinstance(args, float):
                if self.has_key(key):
                    value = [len(self.data[key]), args]
                else:
                    value = [0, args]
            elif isinstance(args, tuple):
                value = list(args)
            elif isinstance(args, list):
                value = args
            else:
                raise ValueError('Unsupported value type')
        elif len(args) == 2:
            value = [args[0], args[1]]
        else:
            raise ValueError('Unsupported value type')

        if key in self.keys:
            self.data[key].append(value)
        else:
            self.data[key] = [value]
            self.keys.add(key)

    def get_last(self, key):
        if not self.has_key(key):
            return None
        return self.data[key][-1]

    def has_key(self, key):
        return key in self.keys

    def get(self, key):
        split = np.hsplit(np.array(self.data[key]), 2)

        return split[1].squeeze().tolist()

    def getXY(self, key, start_idx=0):
        split = np.hsplit(np.array(self.data[key]), 2)

        xs = split[0].squeeze().tolist()
        ys = split[1].squeeze().tolist()

        if not isinstance(xs, list):
            return xs, ys

        if start_idx == 0:
            return xs, ys
        elif start_idx in xs:
            idx = xs.index(start_idx)
            return xs[idx:], ys[idx:]
        else:
            raise KeyError('no start_idx value in X axis data.')

    def get_keys(self):
        return self.keys


class TimeEstimator:
    def __init__(self):
        self.logger = logging.getLogger('TimeEstimator')
        self.start_time = time.time()
        self.count_zero = 0

    def reset(self, count=1):
        self.start_time = time.time()
        self.count_zero = count-1

    def get_est(self, count, total):
        curr_time = time.time()
        elapsed_time = curr_time - self.start_time
        remain = total-count
        remain_time = elapsed_time * remain / (count - self.count_zero)

        elapsed_time /= 3600.0
        remain_time /= 3600.0

        return elapsed_time, remain_time

    def get_est_string(self, count, total):
        elapsed_time, remain_time = self.get_est(count, total)

        elapsed_time_str = "{:.2f}h".format(elapsed_time) if elapsed_time > 1.0 else "{:.2f}m".format(elapsed_time*60)
        remain_time_str = "{:.2f}h".format(remain_time) if remain_time > 1.0 else "{:.2f}m".format(remain_time*60)

        return elapsed_time_str, remain_time_str

    def print_est_time(self, count, total):
        elapsed_time_str, remain_time_str = self.get_est_string(count, total)

        self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
            count, total, elapsed_time_str, remain_time_str))


def util_print_log_array(logger, result_log: LogData):
    assert isinstance(result_log, LogData), 'use LogData Class for result_log.'

    for key in result_log.get_keys():
        logger.info('{} = {}'.format(key+'_list', result_log.get(key)))


def util_save_log_image_with_label(result_file_prefix,
                                   img_params,
                                   result_log: LogData,
                                   labels=None):
    if plt is None:
        raise RuntimeError(f"Matplotlib is not available: {_PLOT_IMPORT_ERROR}")

    dirname = os.path.dirname(result_file_prefix)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)

    _build_log_image_plt(img_params, result_log, labels)

    # 修复Y轴范围：根据实际数据动态调整，避免CTSP-d数据被截断
    if labels is None:
        label_list = list(result_log.get_keys())
    else:
        label_list = list(labels)
    
    all_values = []
    for label in label_list:
        if not result_log.has_key(label):
            continue

        values = result_log.get(label)
        if isinstance(values, list):
            all_values.extend(float(value) for value in values)
        else:
            all_values.append(float(values))
    
    if all_values:
        ax = plt.gca()
        y_min, y_max = min(all_values), max(all_values)
        if y_min == y_max:
            margin = abs(y_min) * 0.1 if y_min != 0 else 0.1
        else:
            margin = (y_max - y_min) * 0.15
        ax.set_ylim(y_min - margin, y_max + margin)
    
 
    plt.tight_layout()
    file_name = '_'.join(label_list)
    fig = plt.gcf()
    fig.savefig('{}-{}.jpg'.format(result_file_prefix, file_name))
    plt.close(fig)


def util_can_save_log_images(logger=None):
    if _PLOT_IMPORT_ERROR is not None:
        if logger is not None:
            logger.warning("Skipping log_image because Matplotlib import failed: %s", _PLOT_IMPORT_ERROR)
        return False

    code = (
        "import os, tempfile; "
        "os.environ.setdefault('MPLBACKEND', 'Agg'); "
        "import matplotlib; matplotlib.use('Agg', force=True); "
        "import matplotlib.pyplot as plt; "
        "fd, path = tempfile.mkstemp(suffix='.png'); os.close(fd); "
        "plt.figure(); plt.plot([0, 1], [0, 1]); plt.savefig(path); "
        "plt.close('all'); os.remove(path)"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
    except Exception as exc:
        if logger is not None:
            logger.warning("Skipping log_image because Matplotlib health check failed: %s", exc)
        return False

    if completed.returncode != 0:
        if logger is not None:
            stderr = completed.stderr.decode("utf-8", errors="replace").strip().splitlines()
            reason = stderr[0] if stderr else "no stderr"
            logger.warning(
                "Skipping log_image because Matplotlib savefig is not healthy "
                "(exit code %s): %s",
                completed.returncode,
                reason,
            )
        return False

    return True


def _build_log_image_plt(img_params,
                         result_log: LogData,
                         labels=None):
    assert isinstance(result_log, LogData), 'use LogData Class for result_log.'

    # Read json
    folder_name = img_params['json_foldername']
    file_name = img_params['filename']
    log_image_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder_name, file_name)

    if os.path.exists(log_image_config_file):
        with open(log_image_config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = {
            "figsize": {"x": 7, "y": 3.5},
            "xlim": {"min": None, "max": None},
            "ylim": {"min": None, "max": None},
            "grid": True,
        }

    figsize = (config['figsize']['x'], config['figsize']['y'])
    plt.figure(figsize=figsize)

    if labels is None:
        label_list = list(result_log.get_keys())
    else:
        label_list = list(labels)
    
    is_loss_plot = any('loss' in label.lower() for label in label_list)
    is_score_plot = any('score' in label.lower() for label in label_list)
    
    for label in label_list:
        plt.plot(*result_log.getXY(label), label=label, linewidth=2.5)

    # 动态计算Y轴范围，避免CTSP-d数据（~4.6）被硬编码的3.83-3.88截断
    ax = plt.gca()
    ymin, ymax = ax.dataLim.ymin, ax.dataLim.ymax
    
    # 添加15%边距确保曲线不贴边
    if ymin != ymax:
        margin = (ymax - ymin) * 0.15
    else:
        margin = abs(ymin) * 0.1 if ymin != 0 else 0.1
    
    # 使用数据实际范围覆盖配置文件中的硬编码值
    plt.ylim(ymin - margin, ymax + margin)

    # X轴范围保持原有逻辑（通常自动适应即可）
    xlim_min = config['xlim']['min']
    xlim_max = config['xlim']['max']
    if xlim_min is None:
        xlim_min = ax.dataLim.xmin
    if xlim_max is None:
        xlim_max = ax.dataLim.xmax
    if xlim_min == xlim_max:
        xlim_min -= 0.5
        xlim_max += 0.5
    plt.xlim(xlim_min, xlim_max)

    plt.rc('legend', **{'fontsize': 10})
    plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    
    if is_loss_plot:
        plt.ylabel('Policy Gradient Loss', fontsize=12, fontweight='bold')
        # 在设置ylim之后反转，确保反转的是正确的范围
        plt.gca().invert_yaxis()
    elif is_score_plot:
        plt.ylabel('Average Tour Length', fontsize=12, fontweight='bold')
    else:
        plt.ylabel('Value', fontsize=12, fontweight='bold')
    
    plt.grid(config["grid"], alpha=0.3, linestyle='--')
    
    if is_loss_plot:
        plt.title('POMO Training Loss Curve', fontsize=14, fontweight='bold', pad=15)
    elif is_score_plot:
        plt.title('POMO Training Score Curve', fontsize=14, fontweight='bold', pad=15)


def copy_all_src(dst_root):
    # execution dir
    if os.path.basename(sys.argv[0]).startswith('ipykernel_launcher'):
        execution_path = os.getcwd()
    else:
        execution_path = os.path.dirname(sys.argv[0])

    # home dir setting
    candidate_dirs = []
    for path_entry in sys.path[:3]:
        if not path_entry:
            continue
        candidate_dir = os.path.abspath(os.path.join(execution_path, path_entry))
        if os.path.isdir(candidate_dir):
            candidate_dirs.append(candidate_dir)

    if candidate_dirs:
        home_dir = min(candidate_dirs, key=len)
    else:
        home_dir = os.path.abspath(os.getcwd())

    # make target directory
    dst_path = os.path.join(dst_root, 'src')

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    for item in sys.modules.items():
        key, value = item

        if hasattr(value, '__file__') and value.__file__:
            src_abspath = os.path.abspath(value.__file__)

            try:
                is_project_file = os.path.commonpath([home_dir, src_abspath]) == home_dir
            except ValueError:
                is_project_file = False

            if is_project_file:
                dst_filepath = os.path.join(dst_path, os.path.basename(src_abspath))

                if os.path.exists(dst_filepath):
                    split = list(os.path.splitext(dst_filepath))
                    split.insert(1, '({})')
                    filepath = ''.join(split)
                    post_index = 0

                    while os.path.exists(filepath.format(post_index)):
                        post_index += 1

                    dst_filepath = filepath.format(post_index)

                # Debug: check whether file exists
                if os.path.exists(src_abspath):
                    shutil.copy2(src_abspath, dst_filepath)
