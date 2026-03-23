import sys
import os
sys.path.insert(0, './utils')
from utils import LogData, util_save_log_image_with_label

# 伪造数据
log_data = LogData()
log_data.append('train_loss', 1, -0.5)
log_data.append('train_loss', 5, -0.3)
log_data.append('train_loss', 10, -0.1)

img_params = {
    'json_foldername': 'log_image_style',
    'filename': 'style_loss_1.json'
}

# 保存到当前目录的 test_output 文件夹
output_dir = './test_output'
os.makedirs(output_dir, exist_ok=True)

util_save_log_image_with_label(
    os.path.join(output_dir, 'test'),  # 前缀：./test_output/test
    img_params,
    log_data,
    ['train_loss']
)

print(f"Test plot saved to: {output_dir}/test-train_loss.jpg")