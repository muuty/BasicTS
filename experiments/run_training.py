#!/usr/bin/env python
"""
Standalone training script for Ray queue.
Usage: python run_training.py <config_path> <gpu_id>
"""
import sys
import os

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_training.py <config_path> <gpu_id>")
        sys.exit(1)

    config_path = sys.argv[1]
    gpu_id = sys.argv[2]

    # Ensure we're in the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from basicts import launch_training
    launch_training(config_path, gpus=gpu_id)
