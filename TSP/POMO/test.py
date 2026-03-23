##########################################################################################
# Machine Environment Config

import sys
print("=" * 50)
print("DEBUG: Python executable:", sys.executable)
print("DEBUG: Should contain 'py310-env':", "py310-env" in sys.executable)
print("=" * 50)

import os
os.environ['MPLBACKEND'] = 'Agg'  # 强制设置

import matplotlib
print("DEBUG: Matplotlib backend before use:", matplotlib.rcParams['backend'])
matplotlib.use('Agg', force=True)  # 强制切换
print("DEBUG: Matplotlib backend after use:", matplotlib.rcParams['backend'])
print("=" * 50)