import h5py
import json

def fix_h5_file(filepath):
    try:
        with h5py.File(filepath, 'r+') as f:
            if 'model_config' in f.attrs:
                config_str = f.attrs['model_config']
                if isinstance(config_str, bytes):
                    config_str = config_str.decode('utf-8')
                config = json.loads(config_str)
                
                # Recursively search and replace batch_shape with batch_input_shape
                def fix_config(obj):
                    if isinstance(obj, dict):
                        # Remove Keras 3 specific keys
                        if 'dtype_policy' in obj:
                            del obj['dtype_policy']
                        if 'batch_shape' in obj:
                            obj['batch_input_shape'] = obj.pop('batch_shape')
                        for k, v in list(obj.items()):
                            fix_config(v)
                    elif isinstance(obj, list):
                        for item in obj:
                            fix_config(item)

                fix_config(config)
                new_config_str = json.dumps(config).encode('utf-8')
                f.attrs['model_config'] = new_config_str
                print(f"Fixed API naming inside {filepath}")
            else:
                print(f"No model_config found in {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

models_to_fix = [
    r"D:\soft computing\models\DL\chest_xray_mobilenet_cpu.h5",
    r"D:\soft computing\models\DL\skin_mobilenet.h5",
    r"D:\soft computing\models\DL\eye_mobilenet.h5",
    r"D:\soft computing\models\DL\hair_mobilenet_best.h5",
]

for p in models_to_fix:
    fix_h5_file(p)
