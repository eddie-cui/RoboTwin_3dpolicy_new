import h5py
import numpy as np
import cv2
import os
import argparse
import concurrent.futures
from tqdm import tqdm
import subprocess
import tempfile
import shutil
def Generate_Mask(action, threshold=0.5, time_window=5):
    timesteps, dims = action.shape
    mask = np.zeros(timesteps, dtype=np.int32)
    for t in range(timesteps - time_window - 1):
        window_action = action[t:t + time_window + 1]
        action_magnitude = np.diff(window_action, axis=0)
        action_magnitude = np.linalg.norm(action_magnitude, axis=1)
        if np.max(action_magnitude) >= threshold:
            mask[t:t + time_window] = 1
    return mask

def copy_structure_with_mask(infile, outfile, mask, current_path=''):
    valid_indices = np.where(mask == 1)[0]
    
    for key in infile.keys():
        item_path = f"{current_path}/{key}" if current_path else key
        item = infile[key]
        
        if isinstance(item, h5py.Group):
            group = outfile.create_group(key)
            for attr_name, attr_value in item.attrs.items():
                group.attrs[attr_name] = attr_value
            copy_structure_with_mask(item, group, mask, item_path)
            
        elif isinstance(item, h5py.Dataset):
            data = item[:]
            if len(data.shape) > 0 and data.shape[0] == len(mask):
                filtered_data = data[valid_indices]
            else:
                filtered_data = data
            dataset = outfile.create_dataset(
                key, 
                data=filtered_data,
                dtype=item.dtype,
                compression=item.compression,
                compression_opts=item.compression_opts,
                shuffle=item.shuffle,
                fletcher32=item.fletcher32
            )
            for attr_name, attr_value in item.attrs.items():
                dataset.attrs[attr_name] = attr_value

def filter_action_data(input_file, output_file, threshold=0.5, time_window=5):
    with h5py.File(input_file, 'r') as infile, h5py.File(output_file, 'w') as outfile:
        action_data_ee_l = infile['endpose/left_endpose'][:]
        action_data_ee_r = infile['endpose/right_endpose'][:]
        action_data_ee_lg = infile['endpose/left_gripper'][:]
        action_data_ee_rg = infile['endpose/right_gripper'][:]
        
        action_data_ee_lg = np.expand_dims(action_data_ee_lg, axis=1)
        action_data_ee_rg = np.expand_dims(action_data_ee_rg, axis=1)
        
        action_data = np.concatenate((action_data_ee_l, action_data_ee_r, action_data_ee_lg, action_data_ee_rg), axis=1)
        mask = Generate_Mask(action_data, threshold, time_window)

        for attr_name, attr_value in infile.attrs.items():
            outfile.attrs[attr_name] = attr_value
        copy_structure_with_mask(infile, outfile, mask)
        
        return len(mask), np.sum(mask)

def visualize_filter(file_path, video_name, output_folder=None):
    with h5py.File(file_path, 'r') as f:
        head_img_data = f['observation/head_camera/rgb'][:]
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        for i, image_bit in enumerate(head_img_data):
            image = cv2.imdecode(np.frombuffer(image_bit, np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                cv2.imwrite(frame_path, image)
        
        video_path = os.path.join(output_folder, f'{video_name}.mp4')
        ffmpeg_cmd = [
            'ffmpeg',
            '-y', 
            '-r', '30', 
            '-i', os.path.join(temp_dir, 'frame_%06d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',
            '-preset', 'medium',
            video_path
        ]
        subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
            
    except Exception as e:
        print(f"\nError creating video for {file_path}: {e}")
    finally:
        shutil.rmtree(temp_dir)
def process_file(file_name, args):
    try:
        input_folder = os.path.join(args.base_folder, args.task_name, args.demo_config, 'data')
        output_folder = os.path.join(args.base_folder, args.task_name, args.demo_config, 'data_filtered')
        visualize_folder = os.path.join(args.base_folder, args.task_name, args.demo_config, 'visualize_filtered')

        input_file = os.path.join(input_folder, file_name)
        output_file = os.path.join(output_folder, file_name)
        input_size, output_size = filter_action_data(input_file, output_file, threshold=args.threshold, time_window=args.time_window)
        if args.visualize:
            video_name_original = os.path.splitext(file_name)[0] + '_original'
            visualize_filter(input_file, video_name_original, output_folder=visualize_folder)
            
            video_name_filtered = os.path.splitext(file_name)[0] + '_filtered'
            visualize_filter(output_file, video_name_filtered, output_folder=visualize_folder)
        return (file_name, input_size, output_size)
    except Exception as e:
        print(f"\nFailed to process {file_name}: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter action data in HDF5 files based on a threshold (Parallel Version).")
    parser.add_argument("--task_name", type=str, required=True, help="Task name")
    parser.add_argument("--base_folder", type=str, required=True, help="Base folder")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Threshold for filtering actions.")
    parser.add_argument("--time_window", "-w", type=int, default=5, help="Timesteps threshold for drop.")
    parser.add_argument("--demo_config", "-c", type=str, default="all", help="Demo config to filter, e.g., demo_clean.")
    parser.add_argument("--visualize", "-v", action='store_true', help="Whether to visualize the filtered results.")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel processes. Defaults to the number of CPU cores.")
    args = parser.parse_args()
    input_folder = os.path.join(args.base_folder, args.task_name, args.demo_config, 'data')
    output_folder = os.path.join(args.base_folder, args.task_name, args.demo_config, 'data_filtered')
    visualize_folder = os.path.join(args.base_folder, args.task_name, args.demo_config, 'visualize_filtered')
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(visualize_folder, exist_ok=True)
    hdf5_files = [name for name in os.listdir(input_folder) if name.endswith('.hdf5')]
    print("-" * 80)
    print(f"Task: {args.task_name}, Base Folder: {args.base_folder}, Demo Config: {args.demo_config}")
    print(f"Found {len(hdf5_files)} files to process with up to {args.workers or os.cpu_count()} workers.")
    filtered_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_file = {
            executor.submit(process_file, file, args): file 
            for file in hdf5_files
        }
        progress_bar = tqdm(concurrent.futures.as_completed(future_to_file), total=len(hdf5_files), desc="Processing files")
        for future in progress_bar:
            result = future.result()
            if result:
                filtered_results.append(result)
    print("\n\n--- Filtering Summary ---")
    filtered_results.sort(key=lambda x: x[0])
    total_input_size = 0
    total_output_size = 0

    for result in filtered_results:
        file_name, input_size, output_size = result
        total_input_size += input_size
        total_output_size += output_size
        retention = (output_size / input_size * 100) if input_size > 0 else 0
        print(f"File: {file_name:<30} | Original: {input_size:>5} | Filtered: {output_size:>5} | Retention: {retention:6.2f}%")

    total_retention = (total_output_size / total_input_size * 100) if total_input_size > 0 else 0
    print("-" * 80)
    print(f"TOTAL:{'':<25} | Original: {total_input_size:>5} | Filtered: {total_output_size:>5} | Retention: {total_retention:6.2f}%")