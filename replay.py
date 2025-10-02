import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"  #（代表仅使用第0，1号GPU）
import sys
import numpy as np
import time
from pathlib import Path
from datetime import datetime
from PIL import Image
import h5py
sys.path.append("./")
from script.eval_3dpolicy import Env
np.random.seed(42)


# Generate random action
def generate_random_action(action_dim=6):
    """Generate random action in range [-1, 1]"""
    return np.random.uniform(-0.5, 0.5, action_dim)

def main():
    # Test parameters
    task_name = "rotate_qrcode"  # Task name
    task_config = "demo_clean"  # Task configuration
    head_camera = "D435"  # Camera type
    seed = 1  # Starting seed
    num_tasks = 1  # Number of tasks to test
    instruction_type = "unseen"  # Instruction type
    name='DP3' # Policy name
    print("\n=== Initializing Environment ===")
    env_manager = Env()
    # model=... # load your model here
    data_path='/data/sea_disk0/cuihz/RoboTwin/data/rotate_qrcode/demo_clean/data/episode0.hdf5'
    with h5py.File(data_path, 'r') as f:
        # 读取所有joint action数据
        left_arm_actions = f['endpose']['left_endpose'][:]
        left_gripper_actions = f['endpose']['left_gripper'][:]
        right_arm_actions = f['endpose']['right_endpose'][:]
        right_gripper_actions = f['endpose']['right_gripper'][:]
    try:
        # Create specific task environment and get valid seeds
        print(f"\n=== Finding {num_tasks} valid scenes for {task_name} task ===")
        result = env_manager.Create_env(
            task_name=task_name,
            head_camera_type=head_camera, 
            seed=seed,
            task_num=num_tasks,
            instruction_type=instruction_type,
            task_config=task_config,
            policy_name=name
        )
        if not result:
            print("Failed to get valid seeds")
            return
        seed_list, id_list, episode_info_list_total = result
        print(f"Found {len(seed_list)} valid task seeds: {seed_list}")
        for i, (seed, task_id, episode_info_list) in enumerate(zip(seed_list, id_list, episode_info_list_total)):            
            inst=env_manager.Init_task_env(seed, task_id, episode_info_list, len(seed_list), name)
            test_stats = {"success": False, "steps_taken": 0}
            step=0
            while True:
                observation = env_manager.get_observation()
                # action = model.predict(observation)
                action = np.concatenate([
                    left_arm_actions[step],  # 7维
                    [left_gripper_actions[step]],  # 转换为1维数组
                    right_arm_actions[step],  # 7维
                    [right_gripper_actions[step]]  # 转换为1维数组
                ])         
                # print(action.shape)       
                step+=1
                action = [action]
                action = np.array(action)  # Ensure action is 2D array
                status = env_manager.Take_action(action,action_types='ee') #action_types can be 'qpos' or 'ee' or 'delta_ee' 
                # print(f"\rStep: {step}, Status: {status}", end='', flush=True)
                if status != "run":
                    test_stats["success"] = (status == "success")
                    test_stats["steps_taken"] = step + 1
                    break
            if status == "run":
                env_manager.Close_env()
                test_stats["steps_taken"] = 0
                raise RuntimeError("Environment closed unexpectedly during task execution.")
            
            print(f"Task {i+1} result: {'Success' if test_stats['success'] else 'Failed'}, Duration: {test_stats['steps_taken']} steps")
        
    except Exception as e:
        import traceback
        print(f"Error during testing: {e}")
        print(traceback.format_exc())
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()
