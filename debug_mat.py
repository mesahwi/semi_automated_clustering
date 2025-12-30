"""Debug MATLAB file structure."""
import scipy.io as sio
import numpy as np
import sys

def inspect_struct(obj, prefix="", depth=0):
    """Recursively inspect a MATLAB struct."""
    if depth > 2:
        return
    
    if hasattr(obj, '_fieldnames'):
        print(f"{prefix}Fields: {obj._fieldnames}")
        for field in obj._fieldnames[:10]:  # Limit to first 10 fields
            val = getattr(obj, field)
            if hasattr(val, '_fieldnames'):
                print(f"{prefix}  {field}: mat_struct")
                inspect_struct(val, prefix + "    ", depth + 1)
            elif isinstance(val, np.ndarray):
                print(f"{prefix}  {field}: ndarray shape={val.shape} dtype={val.dtype}")
            else:
                print(f"{prefix}  {field}: {type(val).__name__}")

def main(path):
    print(f"Loading {path}...")
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    
    keys = [k for k in mat.keys() if not k.startswith('__')]
    print(f"Top-level keys: {keys}")
    
    for key in keys:
        F = mat[key]
        print(f"\n=== {key} ===")
        inspect_struct(F)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
