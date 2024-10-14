
import sys, os, time
from datetime import datetime

DEBUG = True if sys.gettrace() else False


def check_and_mkdir(*args):
    """
        making dirs

        :Input
            - a list of paths of the directories 
    """
    i = 0
    while (i<len(args)):
        dir_path = args[i] 
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        i += 1
    return 


def timer(func):
    def wrapper(*args, **kwargs):
        if DEBUG:
            time_stamp = time.perf_counter()
            result = func(*args, **kwargs)
            print(f"FUNC {func.__name__} finished in: ({time.perf_counter() - time_stamp:.3e}) S")
        else:
            result = func(*args, **kwargs)
        return result
    return wrapper


def get_newest_checkpoint(directory):
    check_and_mkdir(directory)
    files = os.listdir(directory)
    checkpoint_files = [f for f in files if f.startswith('checkpoint') and f.endswith('.pth')]
    if not checkpoint_files:
        print(f"\nWarning: No checkpoint file found in {directory}. Return None.")
        return None
    elif len(checkpoint_files) == 1:
        return checkpoint_files[0]
    
    def extract_date(filename):
        date_str = filename.split('_')[1].split('.')[0]
        return datetime.strptime(date_str, '%Y%m%d-%H:%M')

    newest_file = max(checkpoint_files, key=lambda f: extract_date(f) if '_' in f else datetime.min)
    return newest_file


def get_time_stamp()->str:
    # Generate the timestamp
    return  datetime.now().strftime('%Y%m%d-%H:%M')

