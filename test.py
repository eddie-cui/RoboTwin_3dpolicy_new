import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"  
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

def main():
    # Test parameters
    task_name = "beat_block_hammer"  # Task name
    task_config = "demo_clean"  # Task configuration
    head_camera = "D435"  # Camera type
    seed = 0  # Starting seed
    num_tasks = 4  # Number of tasks to test
    instruction_type = "unseen"  # Instruction type unseen/seen
    name='DP3' # Your Policy name
    print("\n=== Initializing Environment ===")
    env_manager = Env()
    model=... # load your model here
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
            inst=env_manager.Init_task_env(seed, task_id, episode_info_list, len(seed_list), name,) 
            test_stats = {"success": False, "steps_taken": 0}
            step=0
            while True:
                observation = env_manager.get_observation()
                action = model.predict(observation) # Get action from your model
                step+=1
                action = [action]
                action = np.array(action)  # Ensure action is 2D array
                status = env_manager.Take_action(action,action_types='qpos') #action_types can be 'qpos' or 'ee' or 'delta_ee' 
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
