# Adapted from https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/trainer/ray/utils.py#L1
import os

import ray
import torch
from slime.ray.ray_actor import RayActor

# Refer to
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/amd_gpu.py#L102-L103
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/npu.py#L94-L95
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/hpu.py#L116-L117
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/neuron.py#L108-L109
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/tpu.py#L171-L172
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/intel_gpu.py#L97-L98
NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
    "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
    "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
    "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
]

RAY_DEFAULT_ENV_VARS = {
    # Ray's uvloop integration has caused intermittent async actor issues.
    "RAY_USE_UVLOOP": "0",
}

_NESTED_RAY_ENV_ALLOWLIST = (
    # Nested Ray actors otherwise start from the cluster worker environment
    # rather than the submitting job's startup overlay.
    "PYTHONPATH",
    "PRORL_COMPILE_CACHE_ROOT",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TVM_FFI_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
    "CUDA_CACHE_PATH",
    "GEMMA4_SGLANG_SYNC_V_NORM",
    "GEMMA4_SGLANG_RESTORE_STATIC_BUFFERS",
    "GEMMA4_STATIC_BUFFER_AUDIT_PATH",
)


def get_torch_memory_saver_preload_env(package_file: str | None = None) -> dict[str, str]:
    """Return the native preload-hook environment for pauseable CUDA memory."""

    if package_file is None:
        import torch_memory_saver

        package_file = torch_memory_saver.__file__
    package_root = os.path.dirname(os.path.dirname(package_file))
    for filename in (
        "torch_memory_saver_hook_mode_preload_cu12.abi3.so",
        "torch_memory_saver_hook_mode_preload.abi3.so",
    ):
        dynlib_path = os.path.join(package_root, filename)
        if os.path.exists(dynlib_path):
            return {
                "LD_PRELOAD": dynlib_path,
                "TMS_INIT_ENABLE": "1",
                "TMS_INIT_ENABLE_CPU_BACKUP": "1",
            }
    raise FileNotFoundError(
        "Cannot find torch_memory_saver preload library; install the native CUDA hook."
    )


def add_default_ray_env_vars(env_vars: dict[str, str] | None = None) -> dict[str, str]:
    inherited = {
        name: os.environ[name]
        for name in _NESTED_RAY_ENV_ALLOWLIST
        if name in os.environ
    }
    return RAY_DEFAULT_ENV_VARS | inherited | (env_vars or {})


def ray_noset_visible_devices(env_vars=os.environ):
    return any(env_vars.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)


def get_physical_gpu_id():
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return str(props.uuid)


@ray.remote
class Lock(RayActor):
    def __init__(self):
        self._locked = False  # False: unlocked, True: locked

    def acquire(self):
        """
        Try to acquire the lock. Returns True if acquired, False otherwise.
        Caller should retry until it returns True.
        """
        if not self._locked:
            self._locked = True
            return True
        return False

    def release(self):
        """Release the lock, allowing others to acquire."""
        assert self._locked, "Lock is not acquired, cannot release."
        self._locked = False
