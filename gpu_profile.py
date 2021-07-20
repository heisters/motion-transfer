import datetime
import linecache
import os

os.environ['CUDA_LAUNCH_BLOCKING']='1'

#import pynvml3
from py3nvml import py3nvml
import torch
import socket

# different settings
print_tensor_sizes = True
use_incremental = False


if 'GPU_DEBUG' in os.environ:
    gpu_profile_fn = f"Host_{socket.gethostname()}_gpu{os.environ['GPU_DEBUG']}_mem_prof-{datetime.datetime.now():%d-%b-%y-%H-%M-%S}.prof.txt"
    print('profiling gpu usage to ', gpu_profile_fn)


## Global variables
last_tensor_sizes = set()
last_meminfo_used = 0
lineno = None
func_name = None
filename = None
module_name = None


def humanbytes(B):
   'Return the given bytes as a human friendly KB, MB, GB, or TB string'
   B = float(B)
   KB = float(1024)
   MB = float(KB ** 2) # 1,048,576
   GB = float(KB ** 3) # 1,073,741,824
   TB = float(KB ** 4) # 1,099,511,627,776

   if B < KB:
      return '{0}  B'.format(B)
   elif KB <= B < MB:
      return '{0:.2f} KB'.format(B/KB)
   elif MB <= B < GB:
      return '{0:.2f} MB'.format(B/MB)
   elif GB <= B < TB:
      return '{0:.2f} GB'.format(B/GB)
   elif TB <= B:
      return '{0:.2f} TB'.format(B/TB)


def gpu_profile(frame, event, arg):
    # it is _about to_ execute (!)
    global last_tensor_sizes
    global last_meminfo_used
    global lineno, func_name, filename, module_name

    if event == 'line':
        try:
            # about _previous_ line (!)
            if lineno is not None:
                py3nvml.nvmlInit()
                handle = py3nvml.nvmlDeviceGetHandleByIndex(int(os.environ['GPU_DEBUG']))
                meminfo = py3nvml.nvmlDeviceGetMemoryInfo(handle)
                line = linecache.getline(filename, lineno)
                where_str = module_name+' '+func_name+':'+str(lineno)

                new_meminfo_used = meminfo.used
                mem_display = new_meminfo_used-last_meminfo_used if use_incremental else new_meminfo_used
                with open(gpu_profile_fn, 'a+') as f:
                    f.write(f"{where_str:<73}:"
                            f"{humanbytes(mem_display):>10} "
                            f"{line.rstrip()}\n")

                    last_meminfo_used = new_meminfo_used
                    if print_tensor_sizes is True:
                        for tensor in get_tensors():
                            if not hasattr(tensor, 'dbg_alloc_where'):
                                tensor.dbg_alloc_where = where_str
                        new_tensor_sizes = {(type(x), tuple(x.size()), x.element_size() * x.nelement(), x.dbg_alloc_where)
                                            for x in get_tensors()}
                        for t, s, b, loc in new_tensor_sizes - last_tensor_sizes:
                            f.write(f'+ {loc:<50} {str(s):<20} {humanbytes(b):>10} {str(t):<10}\n')
                        for t, s, b, loc in last_tensor_sizes - new_tensor_sizes:
                            f.write(f'- {loc:<50} {str(s):<20} {humanbytes(b):>10} {str(t):<10}\n')
                        last_tensor_sizes = new_tensor_sizes
                py3nvml.nvmlShutdown()

            # save details about line _to be_ executed
            lineno = None

            func_name = frame.f_code.co_name
            filename = frame.f_globals["__file__"]
            if (filename.endswith(".pyc") or
                    filename.endswith(".pyo")):
                filename = filename[:-1]
            module_name = frame.f_globals["__name__"]
            lineno = frame.f_lineno
            
            #only profile codes within the parenet folder, otherwise there are too many function calls into other pytorch scripts
            #need to modify the key words below to suit your case.
            if 'gpu_memory_profiling' not in os.path.dirname(os.path.abspath(filename)):   
                lineno = None  # skip current line evaluation

            if ('car_datasets' in filename
                    or '_exec_config' in func_name
                    or 'gpu_profile' in module_name
                    or 'tee_stdout' in module_name):
                lineno = None  # skip othe unnecessary lines
            
            return gpu_profile

        except (KeyError, AttributeError):
            pass

    return gpu_profile


def get_tensors(gpu_only=True):
    import gc
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensor = obj
            elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
                tensor = obj.data
            else:
                continue

            if tensor.is_cuda:
                yield tensor
        except Exception as e:
            pass
