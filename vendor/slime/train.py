import logging
import os

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.utils.misc import should_run_periodic_action, should_run_periodic_eval
from slime.utils.fatal_resume import emit_fatal_resume


def _disk_updater_onloads_colocated_weights(args) -> bool:
    """Whether the updater must export before SGLang weights are restored."""
    return bool(
        args.offload_rollout
        and args.colocate
        and args.update_weight_mode == "full"
        and args.update_weight_transport == "disk"
    )


def _get_train_actor_cls(args):
    if args.train_backend == "automodel":
        from slime.backends.automodel_utils import AutoModelTrainRayActor

        return AutoModelTrainRayActor
    if args.train_backend == "nemo_megatron":
        from slime.backends.nemo_megatron_utils import NemoMegatronTrainRayActor

        return NemoMegatronTrainRayActor
    return None


def train(args):
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = create_training_models(
        args,
        pgs,
        rollout_manager,
        actor_cls=_get_train_actor_cls(args),
    )

    if args.offload_rollout and not _disk_updater_onloads_colocated_weights(args):
        ray.get(rollout_manager.onload_weights.remote())

    # Always push actor weights to rollout once weights are loaded.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        eval_rollout_id = (
            args.eval_only_rollout_id
            if args.eval_only_rollout_id is not None
            else args.start_rollout_id
        )
        if eval_rollout_id is None:
            eval_rollout_id = 0
        ray.get(rollout_manager.eval.remote(rollout_id=eval_rollout_id))

    def offload_train(actor_trains_this_step):
        # Each model auto-offloads after train() when offload_train is set,
        # so we only need clear_memory for the non-offload case.
        if not args.offload_train:
            if not args.use_critic or actor_trains_this_step:
                actor_model.clear_memory()
            else:
                critic_model.clear_memory()

    def save(rollout_id):
        actor_trains_this_step = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        if actor_trains_this_step:
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.use_critic:
            critic_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))

    # train loop.
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        os.environ["PRORL_CURRENT_ROLLOUT_ID"] = str(rollout_id)
        # "eval before train" applies to a resumed run as well.  Restricting
        # this to rollout 0 silently removes the initial evaluation whenever a
        # valid Megatron checkpoint advances start_rollout_id.
        if (
            args.eval_interval is not None
            and rollout_id == args.start_rollout_id
            and not args.skip_eval_before_train
        ):
            loaded_rollout_id = rollout_id if rollout_id == 0 else rollout_id - 1
            ray.get(rollout_manager.eval.remote(loaded_rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        actor_trains_this_step = (not args.use_critic) or rollout_id >= args.num_critic_only_steps

        if args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            if actor_trains_this_step:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))
            else:
                ray.get(value_refs)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        periodic_save_due = should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        )
        eval_due = should_run_periodic_eval(
            rollout_id, args.eval_interval, num_rollout_per_epoch
        )
        # Align HF/object-store snapshots with visible eval ids.  The old
        # count-based save lands at 19,39,... while eval is logged at
        # 20,40,...; defer the preceding save and publish once at eval_due.
        next_eval_due = should_run_periodic_eval(
            rollout_id + 1, args.eval_interval, num_rollout_per_epoch
        )
        # Per-step recovery must include the step immediately before eval.
        save_due = periodic_save_due and (args.save_interval == 1 or not next_eval_due)
        if save_due or eval_due:
            save(rollout_id)

        offload_train(actor_trains_this_step)
        if args.offload_rollout and not _disk_updater_onloads_colocated_weights(args):
            ray.get(rollout_manager.onload_weights.remote())
        actor_model.update_weights()

        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if eval_due:
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    try:
        train(args)
    except Exception as exc:
        emit_fatal_resume(
            logging.getLogger(__name__),
            "uncaught_training_exception",
            component="slime_train",
            current_rollout_id=os.environ.get("PRORL_CURRENT_ROLLOUT_ID", "unknown"),
            exception_type=type(exc).__name__,
            exception=exc,
        )
        raise
