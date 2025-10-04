import h5py
import numpy as np
import cv2
import os

def Generate_Mask(action, threshold=0.5, time_window=5):
    """
    Generates a binary mask for the action data based on a threshold.

    Parameters:
    - action: np.ndarray, array of end effector action values. 16dim
    - threshold: float, threshold value to generate the mask.
    - time_window: int, timesteps threshold for drop.

    Returns:
    - mask: np.ndarray, binary mask where values above the threshold are 1, else 0.
    """
    timesteps, dims = action.shape
    # print(f"Action data shape: {action.shape}")
    mask = np.zeros(timesteps, dtype=np.int32)
    for t in range(timesteps - time_window - 1):
        window_action = action[t:t + time_window + 1]
        # print(f"Debug: window_action at t={t}: {window_action[0]}\n")
        action_magnitude = np.diff(window_action, axis=0)  # time_window-1, dims
        # print(f"Debug: action_diff at t={t}: {action_magnitude}\n")
        action_magnitude = np.linalg.norm(action_magnitude, axis=1)
        # print(f"Debug: action_diff Norm at t={t}: {action_magnitude}\n")
        if np.max(action_magnitude) >= threshold:
            mask[t:t + time_window] = 1
    return mask

def copy_structure_with_mask(infile, outfile, mask, current_path=''):
    """
    Recursively copy HDF5 file structure and filter data according to mask.
    
    Parameters:
    - infile: input HDF5 file object
    - outfile: output HDF5 file object
    - mask: binary mask for filtering
    - current_path: current path in the HDF5 structure
    """
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
                # print(f"Filtering {item_path}: {data.shape} -> {filtered_data.shape}")
            else:
                filtered_data = data
                # print(f"Direct copy {item_path}: {data.shape}")
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
    """
    Filters action data from an HDF5 file based on a threshold.

    Parameters:
    - input_file: str, input HDF5 file.
    - output_file: str, output HDF5 file.
    - threshold: float, threshold value for filtering actions.
    - time_window: int, timesteps threshold for drop.

    Returns:
    - input_size, output_size: tuple, sizes of the input and output datasets.
    """
    with h5py.File(input_file, 'r') as infile, h5py.File(output_file, 'w') as outfile:
        action_data_ee_l = infile['endpose/left_endpose']
        action_data_ee_r = infile['endpose/right_endpose']
        action_data_ee_lg = infile['endpose/left_gripper']
        action_data_ee_rg = infile['endpose/right_gripper']
        action_data_ee_lg = np.expand_dims(action_data_ee_lg, axis=1)
        action_data_ee_rg = np.expand_dims(action_data_ee_rg, axis=1)
        action_data = np.concatenate((action_data_ee_l, action_data_ee_r, action_data_ee_lg, action_data_ee_rg), axis=1)
        mask = Generate_Mask(action_data, threshold, time_window)

        # print(f"Original data length: {len(mask)}")
        # print(f"Filtered data length: {np.sum(mask)}")
        # print(f"Retention ratio: {np.sum(mask)/len(mask)*100:.2f}%")
        for attr_name, attr_value in infile.attrs.items():
            outfile.attrs[attr_name] = attr_value
        copy_structure_with_mask(infile, outfile, mask)
        
        # print(f"Filtering completed! Output file: {output_file}")
        return len(mask), np.sum(mask)

def visualize_filter(file, video_name, output_folder=None):
    """
    Visualize filtered data by creating a video from head camera images.
    
    Parameters:
    - file: str, path to the HDF5 file containing filtered data
    """
    import subprocess
    import tempfile
    import shutil
    
    with h5py.File(file, 'r') as f:
        head_img_data = f['observation/head_camera/rgb'][:]
        # print(f"Video frames data shape: {head_img_data.shape}")
        temp_dir = tempfile.mkdtemp()
        
        try:
            for i, image_bit in enumerate(head_img_data):
                image = cv2.imdecode(np.frombuffer(image_bit, np.uint8), cv2.IMREAD_COLOR)
                
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
            # print(f"Creating video with {len(head_img_data)} frames...")
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                
        finally:
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Filter action data in HDF5 files based on a threshold.")
    parser.add_argument("--task_name", type=str, help="task name")
    parser.add_argument("--base_folder", type=str, help="base folder")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Threshold for filtering actions.")
    parser.add_argument("--time_window", "-w", type=int, default=5, help="Timesteps threshold for drop.")
    parser.add_argument("--demo_config", "-c", type=str, default="all", help="Demo config to filter, e.g., demo_clean.")
    parser.add_argument("--visualize", "-v", action='store_true', help="Whether to visualize the filtered results.")
    args = parser.parse_args()

    input_folder = os.path.join(args.base_folder, args.task_name, args.demo_config, 'data')
    output_folder = os.path.join(args.base_folder, args.task_name, args.demo_config, 'data_filtered')
    visualize_folder = os.path.join(args.base_folder, args.task_name, args.demo_config, 'visualize_filtered')
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(visualize_folder, exist_ok=True)
    
    total_len = len([name for name in os.listdir(input_folder) if name.endswith('.hdf5')])
    count = 0
    filtered_results = []
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.hdf5'):
            count += 1
            input_file = os.path.join(input_folder, file_name)
            output_file = os.path.join(output_folder, file_name)
            
            print(f"\rProcessing file: {file_name:<30} Progress: {count:>3}/{total_len:>3} ({count/total_len*100:5.1f}%)", end='', flush=True)
            
            input_size, output_size = filter_action_data(input_file, output_file, threshold=args.threshold, time_window=args.time_window)
            filtered_results.append((file_name, input_size, output_size))
            
            if args.visualize:
                video_name_original = os.path.splitext(file_name)[0] + '_original'
                visualize_filter(input_file, video_name_original, output_folder=visualize_folder)
                
                video_name_filtered = os.path.splitext(file_name)[0] + '_filtered'
                visualize_filter(output_file, video_name_filtered, output_folder=visualize_folder)
                
    print("\nFiltering Summary:")
    for result in filtered_results:
        file_name, input_size, output_size = result
        print(f"File: {file_name}, Original Size: {input_size}, Filtered Size: {output_size}, Retention: {output_size/input_size*100:.2f}%\n")