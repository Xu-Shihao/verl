# Copyright 2024 Shihao Xu
# Rollout Progress Tracking
#
# 注意：进度追踪已内置在 verl/workers/rollout/sglang_rollout/sglang_rollout.py 中
# 本文件保留空操作函数以保持向后兼容
# 使用环境变量控制: VERL_ROLLOUT_PROGRESS=0 禁用, DISABLE_ROLLOUT_PROGRESS=1 禁用


def install_progress_tracker(enable: bool = True):
    """
    空操作函数 - 进度追踪已内置在 sglang_rollout.py 中。
    保留此函数以保持向后兼容。
    """
    pass


def uninstall_progress_tracker():
    """
    空操作函数 - 保持向后兼容。
    """
    pass
