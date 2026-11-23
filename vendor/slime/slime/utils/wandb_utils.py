import logging
import os
import shlex
from contextlib import contextmanager
from copy import deepcopy

import wandb

logger = logging.getLogger(__name__)


_PRIVATE_WANDB_ENV_LOADED = False
_PROXY_ENV_KEYS = (
    "WANDB_HTTP_PROXY",
    "WANDB_HTTPS_PROXY",
    "WANDB_PROXY_URL",
    "WANDB_PROXY",
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
)


def _drop_empty_proxy_envs():
    for key in _PROXY_ENV_KEYS:
        if os.environ.get(key) == "":
            os.environ.pop(key, None)


def _load_private_wandb_env():
    global _PRIVATE_WANDB_ENV_LOADED
    if _PRIVATE_WANDB_ENV_LOADED:
        _drop_empty_proxy_envs()
        return
    _PRIVATE_WANDB_ENV_LOADED = True

    env_path = os.environ.get("CLAWGYM_PRIVATE_ENV", "")
    if not env_path or not os.path.exists(env_path):
        _drop_empty_proxy_envs()
        return

    try:
        with open(env_path, encoding="utf-8") as file_handle:
            for raw_line in file_handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                try:
                    parts = shlex.split(line, comments=True, posix=True)
                except ValueError:
                    continue
                if not parts or "=" not in parts[0]:
                    continue
                key, value = parts[0].split("=", 1)
                if key.startswith("WANDB_") and not os.environ.get(key):
                    os.environ[key] = value
    except OSError as exc:
        logger.warning("Failed to read private W&B env file %s: %s", env_path, exc)

    _drop_empty_proxy_envs()


@contextmanager
def _wandb_network_context():
    _load_private_wandb_env()
    proxy_url = (
        os.environ.get("WANDB_PROXY_URL")
        or os.environ.get("WANDB_PROXY")
        or os.environ.get("WANDB_HTTPS_PROXY")
        or os.environ.get("WANDB_HTTP_PROXY")
    )
    http_proxy = os.environ.get("WANDB_HTTP_PROXY") or proxy_url
    https_proxy = os.environ.get("WANDB_HTTPS_PROXY") or proxy_url or http_proxy
    proxy_disabled = os.environ.get("WANDB_USE_PROXY", "1").lower() in {"0", "false", "no", "off"}
    managed_keys = [
        "WANDB_HTTP_PROXY",
        "WANDB_HTTPS_PROXY",
        "WANDB_PROXY_URL",
        "WANDB_PROXY",
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "no_proxy",
        "NO_PROXY",
    ]
    old_values = {key: os.environ.get(key) for key in managed_keys}
    try:
        if (not proxy_disabled) and (http_proxy or https_proxy):
            os.environ["WANDB_HTTP_PROXY"] = http_proxy or https_proxy
            os.environ["WANDB_HTTPS_PROXY"] = https_proxy or http_proxy
            os.environ["WANDB_PROXY_URL"] = proxy_url or https_proxy or http_proxy
            os.environ["WANDB_PROXY"] = proxy_url or https_proxy or http_proxy
            os.environ["http_proxy"] = http_proxy or https_proxy
            os.environ["https_proxy"] = https_proxy or http_proxy
            os.environ["HTTP_PROXY"] = http_proxy or https_proxy
            os.environ["HTTPS_PROXY"] = https_proxy or http_proxy

            raw_no_proxy = os.environ.get("WANDB_NO_PROXY") or os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
            no_proxy_parts = [part.strip() for part in raw_no_proxy.split(",") if part.strip()]
            no_proxy_parts = [part for part in no_proxy_parts if part not in {"*", "api.wandb.ai", ".wandb.ai"}]
            for local_part in ("localhost", "127.0.0.1", "::1"):
                if local_part not in no_proxy_parts:
                    no_proxy_parts.append(local_part)
            sanitized_no_proxy = ",".join(no_proxy_parts)
            os.environ["no_proxy"] = sanitized_no_proxy
            os.environ["NO_PROXY"] = sanitized_no_proxy
            logger.info("W&B proxy enabled only for wandb initialization.")
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _wandb_login_key(args):
    _load_private_wandb_env()
    return args.wandb_key or os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_KEY") or None


def _is_offline_mode(args) -> bool:
    """Detect whether W&B should run in offline mode.

    Priority order:
    1) args.wandb_mode if provided
    2) WANDB_MODE environment variable
    """
    if args.wandb_mode:
        return args.wandb_mode == "offline"
    return os.environ.get("WANDB_MODE") == "offline"


def init_wandb_primary(args):
    if not args.use_wandb:
        args.wandb_run_id = None
        return

    # Set W&B mode if specified (overrides WANDB_MODE env var)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
        if args.wandb_mode == "offline":
            logger.info("W&B offline mode enabled. Data will be saved locally.")
        elif args.wandb_mode == "disabled":
            logger.info("W&B disabled mode enabled. No data will be logged.")
        elif args.wandb_mode == "online":
            logger.info("W&B online mode enabled. Data will be uploaded to cloud.")

    offline = _is_offline_mode(args)

    # Only perform explicit login when NOT offline
    wandb_key = _wandb_login_key(args)
    if (not offline) and wandb_key is not None:
        with _wandb_network_context():
            wandb.login(key=wandb_key, host=args.wandb_host)

    # Prepare wandb init parameters
    # add random 6 length string with characters
    if args.wandb_random_suffix:
        group = args.wandb_group + "_" + wandb.util.generate_id()
        run_name = f"{group}-RANK_{args.rank}"
    else:
        group = args.wandb_group
        run_name = args.wandb_group

    # Prepare wandb init parameters
    init_kwargs = {
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "group": group,
        "name": run_name,
        "config": _compute_config_for_logging(args),
    }

    # Configure settings based on offline/online mode
    if offline:
        init_kwargs["settings"] = wandb.Settings(mode="offline")
    else:
        init_kwargs["settings"] = wandb.Settings(mode="shared", x_primary=True)

    # Add custom directory if specified
    if args.wandb_dir:
        # Ensure directory exists to avoid backend crashes
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir
        logger.info(f"W&B logs will be stored in: {args.wandb_dir}")

    if offline:
        wandb.init(**init_kwargs)
    else:
        with _wandb_network_context():
            wandb.init(**init_kwargs)

    _init_wandb_common()

    # Set wandb_run_id in args for easy access throughout the training process
    args.wandb_run_id = wandb.run.id


def _compute_config_for_logging(args):
    output = _args_to_config_dict(args)

    whitelist_env_vars = [
        "SLURM_JOB_ID",
        # We may insert more default values here, and may also allow users to configure a whitelist
    ]
    output["env_vars"] = {k: v for k, v in os.environ.items() if k in whitelist_env_vars}

    if getattr(args, "use_critic", False):
        critic_args = _get_role_args_for_logging(args, role="critic")
        output.update(_prefix_config_keys(_args_to_config_dict(critic_args), "critic"))

    return output


def _args_to_config_dict(args):
    return deepcopy(args.__dict__)


def _prefix_config_keys(config, prefix):
    return {f"{prefix}/{key}": value for key, value in config.items()}


def _get_role_args_for_logging(args, role):
    if getattr(args, "megatron_config_path", None) is None:
        return args

    from slime.utils.arguments import parse_megatron_role_args

    return parse_megatron_role_args(args, args.megatron_config_path, role=role)


def _compute_secondary_config_for_logging(args, role=None):
    config = _args_to_config_dict(args)
    if role == "critic":
        return _prefix_config_keys(config, "critic")
    return config


# https://docs.wandb.ai/guides/track/log/distributed-training/#track-all-processes-to-a-single-run
def init_wandb_secondary(args, role=None):
    wandb_run_id = getattr(args, "wandb_run_id", None)
    if wandb_run_id is None:
        return

    # Set W&B mode if specified (same as primary)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    offline = _is_offline_mode(args)

    wandb_key = _wandb_login_key(args)
    if (not offline) and wandb_key is not None:
        with _wandb_network_context():
            wandb.login(key=wandb_key, host=args.wandb_host)

    # Configure settings based on offline/online mode
    if offline:
        settings_kwargs = dict(mode="offline")
    else:
        settings_kwargs = dict(
            mode="shared",
            x_primary=False,
            x_update_finish_state=False,
        )

    init_kwargs = {
        "id": wandb_run_id,
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "config": _compute_secondary_config_for_logging(args, role=role),
        "resume": "allow",
        "reinit": True,
        "settings": wandb.Settings(**settings_kwargs),
    }

    # Add custom directory if specified
    if args.wandb_dir:
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir

    if offline:
        wandb.init(**init_kwargs)
    else:
        with _wandb_network_context():
            wandb.init(**init_kwargs)

    _init_wandb_common()


def _init_wandb_common():
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("rollout/step")
    wandb.define_metric("rollout/*", step_metric="rollout/step")
    wandb.define_metric("multi_turn/*", step_metric="rollout/step")
    wandb.define_metric("passrate/*", step_metric="rollout/step")
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    wandb.define_metric("perf/*", step_metric="rollout/step")
    wandb.define_metric("clawgym/*", step_metric="rollout/step")
    wandb.define_metric("openclaw/*", step_metric="rollout/step")
    wandb.define_metric("polar/*", step_metric="rollout/step")
