import logging
import os

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.utils.misc import should_run_periodic_action, should_run_periodic_eval


logger = logging.getLogger(__name__)


def _recover_rollout_engines_after_eval(args, actor_model, rollout_manager) -> None:
    """Refit any engine lost during eval before generation resumes."""
    if not args.use_fault_tolerance:
        return
    if not ray.get(rollout_manager.has_dead_updatable_engines.remote()):
        return
    logger.warning(
        "Evaluation quarantined at least one rollout engine; recovering and "
        "synchronizing current actor weights before the next rollout"
    )
    actor_model.update_weights()


def _get_train_actor_cls(args):
    if args.train_backend == "automodel":
        from slime.backends.automodel_utils import AutoModelTrainRayActor

        return AutoModelTrainRayActor
    if args.train_backend == "nemo_megatron":
        from slime.backends.nemo_megatron_utils import NemoMegatronTrainRayActor

        return NemoMegatronTrainRayActor
    return None


# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
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

    skip_actor_weight_updates = os.environ.get("SLIME_SKIP_CRITIC_ONLY_ACTOR_WEIGHT_SYNC", "0") == "1"
    if skip_actor_weight_updates:
        assert args.use_critic and args.num_critic_only_steps >= args.num_rollout, (
            "SLIME_SKIP_CRITIC_ONLY_ACTOR_WEIGHT_SYNC=1 is only valid when the "
            "entire run is critic-only"
        )
        logger.info(
            "Skipping actor-to-rollout weight sync for a pure critic-only run; "
            "actor and rollout engines must start from the same checkpoint."
        )
    else:
        # Push actor weights to rollout once weights are loaded.
        actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.eval_interval is not None and not args.skip_eval_before_train:
        ray.get(rollout_manager.eval.remote(rollout_id=args.start_rollout_id))
        _recover_rollout_engines_after_eval(args, actor_model, rollout_manager)

    if args.start_rollout_id >= args.num_rollout:
        ray.get(rollout_manager.dispose.remote())
        finish_tracking(args)
        return

    # async train loop.
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = ray.get(rollout_data_next_future)

        # Start the next rollout early.
        if rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        if args.use_critic:
            actor_trains_this_step = rollout_id >= args.num_critic_only_steps
            value_refs = critic_model.async_train(rollout_id, rollout_data_curr_ref)
            if actor_trains_this_step:
                ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref, external_data=value_refs))
            else:
                ray.get(value_refs)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))

        periodic_save_due = should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        )
        eval_due = should_run_periodic_eval(
            rollout_id, args.eval_interval, num_rollout_per_epoch
        )
        # Align HF/object-store snapshots with visible eval ids rather than
        # the preceding zero-based count ids.
        next_eval_due = should_run_periodic_eval(
            rollout_id + 1, args.eval_interval, num_rollout_per_epoch
        )
        save_due = periodic_save_due and not next_eval_due
        if save_due or eval_due:
            if (not args.use_critic) or rollout_id >= args.num_critic_only_steps:
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

        if not skip_actor_weight_updates and (rollout_id + 1) % args.update_weights_interval == 0:
            # sync generate before update weights to prevent update weight in the middle of generation
            rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
            rollout_data_next_future = None
            actor_model.update_weights()

        if eval_due:
            ray.get(rollout_manager.eval.remote(rollout_id))
            _recover_rollout_engines_after_eval(args, actor_model, rollout_manager)

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
