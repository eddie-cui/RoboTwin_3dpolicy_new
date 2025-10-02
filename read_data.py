import h5py
import numpy as np

def print_hdf5_structure(name, obj):
    """递归打印HDF5文件结构"""
    indent = "  " * name.count('/')
    if isinstance(obj, h5py.Group):
        print(f"{indent}{name}/ (Group)")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}{name}: shape={obj.shape}, dtype={obj.dtype}")

def main():
    data_path = '/data/sea_disk0/cuihz/RoboTwin/data/rotate_qrcode/demo_clean/data/episode0.hdf5'
    
    print("=== HDF5 File Structure ===")
    with h5py.File(data_path, 'r') as f:
        f.visititems(print_hdf5_structure)
    
    print("\n=== Detailed endpose structure ===")
    with h5py.File(data_path, 'r') as f:
        if 'endpose' in f:
            endpose_group = f['endpose']
            print("Keys in endpose:", list(endpose_group.keys()))
            
            for key in endpose_group.keys():
                data = endpose_group[key]
                print(f"{key}:")
                print(f"  Shape: {data.shape}")
                print(f"  Dtype: {data.dtype}")
                print(f"  First few values: {data[:3] if len(data) > 0 else 'No data'}")
                print()

if __name__ == "__main__":
    main()