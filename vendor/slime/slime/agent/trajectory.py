"""Build a per-session training trajectory from multi-turn conversation data.

The :class:`TrajectoryManager` builds one trajectory per session. ``record_turn``
feeds in each turn (prompt messages + the served model's sglang snapshot),
routing it into a per-sid message tree; ``get_trajectory`` then linearizes that
tree into a ``list[Sample]`` of loss-masked training rows, tolerating TITO
re-tokenization drift via fork/replace.
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import logging
from collections.abc import Iterator
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


# ===========================================================================
# TurnRecord
# ===========================================================================


@dataclasses.dataclass(frozen=True)
class TurnRecord:
    """One sglang ``/generate`` snapshot: the contract between an adapter and the
    manager. Adapters build it from a turn's prompt/output token ids; ``record_turn``
    consumes it."""

    prompt_ids: list[int]
    output_ids: list[int]
    finish_reason: str
    output_log_probs: list[float] = dataclasses.field(default_factory=list)
    ill_formed: bool = False
    # Set by token-lineage-aware adapters when the model-visible history was
    # genuinely rewritten (compact/snip/repair/unsafe boundary).  Such a turn
    # must start a new trainable segment even if the token drift is short enough
    # that the legacy REALIGN heuristic could otherwise absorb it.
    segment_boundary_reason: str | None = None


# ===========================================================================
# MessageNode
# ===========================================================================


class MessageNode:
    """One node in a session's routing tree, carrying a single chat message
    (``None`` for the dummy root and for an assistant leaf we generated but
    whose ``response_message`` was empty).

    The two kinds are distinguished by whether ``turn`` is set, which reflects
    WHERE the message came from:

    * **generated** (``turn is not None``): an assistant message the model
      actually generated this turn, fed in via ``record_turn``. ``turn`` holds
      its :class:`TurnRecord` -- the prompt/output ids, logprobs and finish
      reason that ``get_trajectory`` linearizes into training tokens.
    * **routing-only** (``turn is None``): the message came from the prompt, not
      from generation, so it only exists to route. This is every
      system/user/tool node, AND any assistant we did NOT generate: a foreign
      assistant the client replayed in a later prompt, or a prior generated turn
      demoted by the rewrite-merge in ``_try_merge_assistant_rewrite``.
    """

    def __init__(
        self,
        *,
        role: str | None = None,
        message: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        parent: MessageNode | None = None,
    ) -> None:
        self.role = role
        self.message = message
        self.metadata = dict(metadata or {})
        self.parent: MessageNode | None = parent
        self.children: list[MessageNode] = []
        self.turn: TurnRecord | None = None  # the generated TurnRecord, else None (routing-only)
        self.turn_index: int | None = None
        self.request_messages: list[dict[str, Any]] | None = None
        self.request_tools: list[dict[str, Any]] | None = None
        self.chat_template_kwargs: dict[str, Any] = {}
        # Shared by sibling leaf paths; the first to reach it trains on it, the rest
        # re-emit it as loss_mask=0 context -- so each response is trained exactly once.
        self.response_trained: bool = False

    @property
    def is_root(self) -> bool:
        return self.parent is None

    def add_child(self, child: MessageNode) -> MessageNode:
        child.parent = self
        self.children.append(child)
        return child

    def path_from_root(self) -> list[MessageNode]:
        """Ordered list of nodes from the first non-root ancestor down to self."""
        chain: list[MessageNode] = []
        node: MessageNode | None = self
        while node is not None and not node.is_root:
            chain.append(node)
            node = node.parent
        chain.reverse()
        return chain

    def leaves(self) -> Iterator[MessageNode]:
        if not self.children:
            yield self
            return
        for c in self.children:
            yield from c.leaves()


# ===========================================================================
# drift classification — how an incoming turn's prompt relates to held tokens
# ===========================================================================


def _common_prefix_len(a: list[int], b: list[int], chunk: int = 4096) -> int:
    limit = min(len(a), len(b))
    matched = 0
    while matched < limit:
        chunk_end = min(matched + chunk, limit)
        if a[matched:chunk_end] == b[matched:chunk_end]:
            matched = chunk_end
        else:
            while matched < chunk_end and a[matched] == b[matched]:
                matched += 1
            return matched
    return matched


class DriftKind(enum.Enum):
    CLEAN = "clean"  # drift == 0: prompt_ids exactly extends held tokens; append the tail beyond them
    REALIGN = "realign"  # drift inside the most-recent response span and short incoming response; replace that span (loss_mask=0)
    FORK = "fork"  # everything else: close this builder, open a fresh one as a fork


_PREPARE_COUNT_KEYS = (
    "initial",
    "clean",
    "realign",
    "token_fork",
    "message_branch",
    "rewrite_merge",
    "retry_prune",
)


def _new_prepare_diagnostics() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "counts": {key: 0 for key in _PREPARE_COUNT_KEYS},
        "realign_discarded_trainable_tokens": 0,
        "events": [],
    }


def _bump_prepare_count(diagnostics: dict[str, Any], key: str) -> None:
    counts = diagnostics["counts"]
    counts[key] = int(counts.get(key, 0)) + 1


# ===========================================================================
# SampleBuilder — accumulates turns into one trainable Sample (fork closes it)
# ===========================================================================


class _SampleBuilder:
    """Accumulates a chain's turns into the token sequence of one ``Sample``.

    A chain of turns is appended one at a time via :meth:`append_turn`. Ideally
    each turn's prompt exactly extends the tokens we already hold, but a replayed
    turn rarely re-tokenizes byte-for-byte: TITO round-trips and chat-template
    re-rendering both perturb the ids of content we've already seen. The builder
    handles this drift in a source-agnostic way, classified by where and how far
    the prompt diverges from the held tokens (see :meth:`classify_token_drift`):

    * **CLEAN** -- no drift; append the prompt tail beyond what we hold.
    * **REALIGN** -- a short divergence inside the most-recent response span;
      overwrite that span from the prompt as loss_mask=0 and keep accumulating.
    * **FORK** -- divergence too large or too early to absorb; this builder is
      rejected and the caller closes it and opens a fresh one. That boundary is
      the "fork".

    Each surviving builder yields one Sample.
    """

    def __init__(self, fork_threshold: int) -> None:
        self._fork_threshold = fork_threshold
        self.tokens: list[int] = []
        self.loss_mask: list[int] = []
        self.logprobs: list[float] = []
        self.last_response_start_idx: int | None = None
        self.leading_prompt_len: int = 0
        self.turn_contexts: list[dict[str, Any]] = []

    def classify_token_drift(self, turn: TurnRecord) -> DriftKind:
        """Decide how this builder should absorb ``turn``'s prompt.

        The incoming turn's prompt is expected to match the tokens this builder
        already holds as an exact prefix. When token drift has occurred -- the
        prompt diverges from the held tokens -- we decide whether to REALIGN
        (heal a short divergence inside the most-recent response span) or to FORK
        (``len(turn.output_ids) >= fork_threshold``, or the divergence sits too
        early to absorb). With no drift the turn is handled the CLEAN way -- a
        plain prefix extension.
        """
        if turn.segment_boundary_reason:
            return DriftKind.FORK

        realign_at = _common_prefix_len(self.tokens, turn.prompt_ids)
        drift = len(self.tokens) - realign_at

        if drift == 0:
            return DriftKind.CLEAN

        # REALIGN only heals drift that falls inside the most-recent response span
        # (and is short); divergence anywhere earlier, or an empty builder, forks.
        start = self.last_response_start_idx
        if start is not None and realign_at >= start and len(turn.output_ids) < self._fork_threshold:
            return DriftKind.REALIGN
        return DriftKind.FORK

    def append_turn(
        self,
        turn: TurnRecord,
        kind: DriftKind,
        *,
        trained: bool = True,
        turn_index: int | None = None,
        request_messages: list[dict[str, Any]] | None = None,
        request_tools: list[dict[str, Any]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Append one turn into this SampleBuilder, branching on ``kind``: for REALIGN
        we overwrite the already-saved response span, for CLEAN we just append this
        turn's prompt tail."""
        assert kind is not DriftKind.FORK, "append_turn called on a builder that would fork"

        is_first_turn = self.last_response_start_idx is None

        # --- append this turn's prompt tail (loss_mask=0) ---
        if kind is DriftKind.REALIGN:
            self._align_to_prompt(turn.prompt_ids)  # drop the drifted tail, re-append from prompt
        else:  # CLEAN: held tokens are an exact prefix of prompt_ids; append the tail beyond them
            self._append_tokens(turn.prompt_ids[len(self.tokens) :], loss_mask=0)

        # --- append this turn's generated response (loss_mask=1 unless re-emitted as context) ---
        self.last_response_start_idx = len(self.tokens)
        self._append_tokens(
            turn.output_ids, loss_mask=int(trained), logprobs=turn.output_log_probs if trained else None
        )
        if trained:
            self.turn_contexts.append({
                "schema_version": 1,
                "turn_index": turn_index,
                "messages": copy.deepcopy(request_messages or []),
                "tools": copy.deepcopy(request_tools or []),
                "chat_template_kwargs": copy.deepcopy(chat_template_kwargs or {}),
                "prompt_ids": list(turn.prompt_ids),
                "output_ids": list(turn.output_ids),
                "output_start": self.last_response_start_idx,
                "output_end": len(self.tokens),
            })

        if is_first_turn:
            self.leading_prompt_len = len(turn.prompt_ids)

    def _align_to_prompt(self, prompt_ids: list[int]) -> None:
        """Heal REALIGN drift by overwriting the most-recent response span with
        ``prompt_ids`` as loss_mask=0: the drifted tokens carry no signal, and re-appending
        from the prompt keeps the builder contiguous. Earlier turns are untouched."""
        response_start = self.last_response_start_idx
        self.turn_contexts = [
            context for context in self.turn_contexts
            if int(context["output_end"]) <= response_start
        ]
        tail = prompt_ids[response_start:]
        self.tokens[response_start:] = tail
        self.loss_mask[response_start:] = [0] * len(tail)
        self.logprobs[response_start:] = [0.0] * len(tail)

    def _append_tokens(self, ids: list[int], *, loss_mask: int, logprobs: list[float] | None = None) -> None:
        self.tokens.extend(ids)
        self.loss_mask.extend([loss_mask] * len(ids))
        self.logprobs.extend(logprobs if logprobs else [0.0] * len(ids))

    def has_trained_response(self) -> bool:
        return any(self.loss_mask[self.leading_prompt_len :])

    def to_sample(
        self, base_sample: Sample, extra_metadata: dict[str, Any] | None, max_sample_tokens: int = 0
    ) -> Sample:
        """Emit the accumulated tokens as one ``Sample``, stripping the first-turn
        prompt so loss_mask / logprobs cover only the response region."""
        prompt_end = self.leading_prompt_len  # first-turn prompt stripped; response region starts here
        tokens = list(self.tokens)
        loss_mask = self.loss_mask
        logprobs = self.logprobs
        if max_sample_tokens and len(tokens) > max_sample_tokens:
            tokens = tokens[:max_sample_tokens]
            loss_mask = loss_mask[:max_sample_tokens]
            logprobs = logprobs[:max_sample_tokens]
        # A context cap can cut inside the first-turn prompt.  Clamp the
        # response boundary to the retained prefix so response_length never
        # becomes negative; callers drop the resulting all-zero segment.
        start = min(prompt_end, len(tokens))
        md = dict(extra_metadata or {})
        response_turns: list[dict[str, Any]] = []
        for context in self.turn_contexts:
            output_start = int(context["output_start"])
            output_end = min(int(context["output_end"]), len(tokens))
            if output_start >= output_end:
                continue
            response_start = output_start - start
            response_end = output_end - start
            if response_start < 0 or response_end > len(loss_mask[start:]):
                continue
            output_ids = list(context["output_ids"])[: output_end - output_start]
            if tokens[output_start:output_end] != output_ids:
                continue
            response_context = copy.deepcopy(context)
            response_context["output_ids"] = output_ids
            response_context["output_start"] = response_start
            response_context["output_end"] = response_end
            response_turns.append(response_context)
        clawgym_metadata = dict(md.get("clawgym_nanobot") or {})
        clawgym_metadata["opsd_turns"] = response_turns
        md["clawgym_nanobot"] = clawgym_metadata
        return Sample(
            index=base_sample.index,
            group_index=base_sample.group_index,
            rollout_id=base_sample.rollout_id if base_sample.rollout_id is not None else base_sample.index,
            prompt=base_sample.prompt,
            label=base_sample.label,
            tokens=tokens,
            response_length=len(loss_mask) - start,
            loss_mask=loss_mask[start:],
            rollout_log_probs=logprobs[start:],
            reward=0.0,
            status=Sample.Status.COMPLETED,
            metadata=md,
        )


# ===========================================================================
# TrajectoryManager
# ===========================================================================


class TrajectoryManager:
    """Record message histories and emit token-aligned training samples.

    Existing callers keep response realignment enabled. ResearchClaw can also
    opt into reference-only retry pruning without changing other task profiles.
    """

    def __init__(
        self,
        *,
        fork_threshold_tokens: int | None = None,
        enable_realign: bool = True,
        prune_unacknowledged_empty_retries: bool = False,
    ) -> None:
        self._fork_threshold: int = 1024 if fork_threshold_tokens is None else fork_threshold_tokens
        self._enable_realign = bool(enable_realign)
        self._prune_unacknowledged_empty_retries = bool(prune_unacknowledged_empty_retries)
        self._trees: dict[str, MessageNode] = {}
        self._turn_count: dict[str, int] = {}
        self._prepare_diagnostics: dict[str, dict[str, Any]] = {}

    # -------------------- public ------------------------------------------

    def has_session(self, sid: str) -> bool:
        return sid in self._trees

    def turn_count(self, sid: str) -> int:
        return self._turn_count.get(sid, 0)

    def record_turn(
        self,
        sid: str,
        *,
        turn: TurnRecord,
        prompt_messages: list[dict[str, Any]],
        response_message: dict[str, Any] | None,
        request_messages: list[dict[str, Any]] | None = None,
        request_tools: list[dict[str, Any]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not prompt_messages:
            logger.warning("record_turn(sid=%s): empty prompt_messages; skipping", sid)
            return
        assert not turn.output_log_probs or len(turn.output_log_probs) == len(turn.output_ids), (
            f"turn.output_log_probs length {len(turn.output_log_probs)} != "
            f"turn.output_ids length {len(turn.output_ids)}"
        )

        root = self._trees.setdefault(sid, MessageNode())
        diagnostics = self._prepare_diagnostics.setdefault(sid, _new_prepare_diagnostics())

        node, depth = self._find_mount_point(root, prompt_messages)
        turn_index = self._turn_count.get(sid, 0) + 1
        pruned_retry = (
            self._prune_unacknowledged_empty_retry(sid, node)
            if self._prune_unacknowledged_empty_retries
            else None
        )
        if pruned_retry is not None:
            _bump_prepare_count(diagnostics, "retry_prune")
            diagnostics["events"].append({
                "turn_index": turn_index,
                "action": "retry_prune",
                "message_depth": depth,
                "abandoned_turn_index": pruned_retry.turn_index,
                "abandoned_response_tokens": len(pruned_retry.turn.output_ids),
            })
        unmatched_depth = depth
        branch_candidate = depth < len(prompt_messages) and bool(node.children)
        existing_child_roles = [child.role for child in node.children]
        node, depth = self._try_merge_assistant_rewrite(sid, node, prompt_messages, depth)
        rewrite_merged = depth > unmatched_depth
        if rewrite_merged:
            _bump_prepare_count(diagnostics, "rewrite_merge")
            merged = node.metadata.get("merged_rewrite") or {}
            diagnostics["events"].append({
                "turn_index": turn_index,
                "action": "rewrite_merge",
                "message_depth": unmatched_depth,
                "abandoned_turn_index": merged.get("abandoned_turn_index"),
                "abandoned_response_tokens": merged.get("abandoned_response_tokens"),
            })
        node = self._mount_prompt_messages(node, prompt_messages[depth:])
        if not rewrite_merged:
            if branch_candidate:
                branch_depth = unmatched_depth
                incoming_role = prompt_messages[unmatched_depth].get("role")
                branch_child_roles = existing_child_roles
            elif node.children:
                branch_depth = len(prompt_messages)
                incoming_role = "assistant"
                branch_child_roles = [child.role for child in node.children]
            else:
                branch_depth = None
            if branch_depth is not None:
                _bump_prepare_count(diagnostics, "message_branch")
                diagnostics["events"].append({
                    "turn_index": turn_index,
                    "action": "message_branch",
                    "message_depth": branch_depth,
                    "incoming_role": incoming_role,
                    "existing_child_roles": branch_child_roles,
                })
        self._attach_assistant_leaf(
            sid,
            node,
            turn=turn,
            response_message=response_message,
            request_messages=request_messages if request_messages is not None else prompt_messages,
            request_tools=request_tools,
            chat_template_kwargs=chat_template_kwargs,
            metadata=metadata,
        )

    def get_trajectory(
        self,
        sid: str,
        *,
        base_sample: Sample,
        reward: float = 0.0,
        extra_metadata: dict[str, Any] | None = None,
        max_sample_tokens: int = 0,
    ) -> list[Sample]:
        """Linearize this sid's routing tree into slime ``Sample`` objects and
        consume the session.

        Each routing leaf yields one or more Samples; ``reward`` is assigned in
        full to every emitted Sample (not split across them), so each trained
        turn carries the trajectory's outcome reward. The sid is dropped
        afterwards, so a second call for the same sid returns ``[]``.
        """
        root = self._trees.get(sid)
        if root is None:
            return []

        samples: list[Sample] = []
        routing_leaves = [routing_leaf for routing_leaf in root.leaves() if not routing_leaf.is_root]
        for credit_chain_index, routing_leaf in enumerate(routing_leaves):
            chain = routing_leaf.path_from_root()
            samples.extend(
                self._chain_to_samples(
                    sid,
                    chain,
                    base_sample=base_sample,
                    extra_metadata=extra_metadata,
                    max_sample_tokens=max_sample_tokens,
                    credit_chain_index=credit_chain_index,
                    routing_leaf_count=len(routing_leaves),
                )
            )

        diagnostics = copy.deepcopy(self._prepare_diagnostics.get(sid, _new_prepare_diagnostics()))
        for s in samples:
            s.reward = reward
            metadata = dict(s.metadata or {})
            clawgym_metadata = metadata.get("clawgym_nanobot")
            if isinstance(clawgym_metadata, dict):
                clawgym_metadata = dict(clawgym_metadata)
                clawgym_metadata["trajectory_prepare"] = copy.deepcopy(diagnostics)
                metadata["clawgym_nanobot"] = clawgym_metadata
            else:
                metadata["trajectory_prepare"] = copy.deepcopy(diagnostics)
            s.metadata = metadata

        self._trees.pop(sid, None)
        self._turn_count.pop(sid, None)
        self._prepare_diagnostics.pop(sid, None)
        return samples

    def drop_session(self, sid: str) -> None:
        self._trees.pop(sid, None)
        self._turn_count.pop(sid, None)
        self._prepare_diagnostics.pop(sid, None)

    # -------------------- internals ----------------------------------------

    def _find_mount_point(self, root: MessageNode, messages: list[dict[str, Any]]) -> tuple[MessageNode, int]:
        """Walk down the tree matching each message by role and dict equality (==),
        returning the deepest node that still matches and where to mount the rest."""
        node = root
        depth = 0
        while depth < len(messages):
            msg = messages[depth]
            next_child = None
            for child in node.children:
                if child.role == msg.get("role") and child.message == msg:
                    next_child = child
                    break
            if next_child is None:
                break
            node = next_child
            depth += 1
        return node, depth

    def _prune_unacknowledged_empty_retry(
        self,
        sid: str,
        mount: MessageNode,
    ) -> MessageNode | None:
        """Remove the latest wire-empty response when the client retries it.

        Nanobot does not append an unusable empty/reasoning-only response to its
        history. The following request therefore mounts at the same parent. The
        abandoned attempt is a retry, not a second trajectory branch.
        """
        latest_turn = self._turn_count.get(sid)
        if latest_turn is None:
            return None
        for child in mount.children:
            message = child.message or {}
            if (
                child.role == "assistant"
                and child.turn is not None
                and child.turn_index == latest_turn
                and not child.children
                and not str(message.get("content") or "").strip()
                and not message.get("tool_calls")
            ):
                mount.children.remove(child)
                return child
        return None

    def _try_merge_assistant_rewrite(
        self,
        sid: str,
        node: MessageNode,
        prompt_messages: list[dict[str, Any]],
        depth: int,
    ) -> tuple[MessageNode, int]:
        """Merge a short assistant-rewrite onto its node instead of forking.

        A harness may replay a prior assistant message slightly re-rendered (e.g.
        whitespace) in a later prompt. It no longer matches the node we generated,
        so it would fork -- stranding the original generated turn as a dead-end
        leaf that still emits its own training Sample. Instead we overwrite that
        node's message in place and stop training its generated content (demote to
        routing-only), so only the live branch trains. This only applies below
        ``fork_threshold``: a long abandoned response carries enough real signal
        to fork and train standalone.

        Forking is always safe (a rewrite mounts as routing-only); this is purely
        a cleanup. So we merge only when the mount point has exactly one assistant
        child that is a leaf, generated (``turn`` set), and short (response <
        ``fork_threshold``), and fork otherwise, since absorbing destroys a
        generated TurnRecord irreversibly.
        """
        if not self._enable_realign or self._fork_threshold <= 0:
            return node, depth  # feature off
        if depth >= len(prompt_messages) or prompt_messages[depth].get("role") != "assistant":
            return node, depth  # genuine non-assistant history fork -> leave it

        asst_children = [c for c in node.children if c.role == "assistant"]
        if len(asst_children) != 1:
            if len(asst_children) > 1:
                logger.warning(
                    "record_turn(sid=%s turn=%s): %d assistant children at mount "
                    "point; can't tell which the rewrite targets, so forking.",
                    sid,
                    self._turn_count.get(sid, 0) + 1,
                    len(asst_children),
                )
            return node, depth

        rewritten_node = asst_children[0]
        if (
            rewritten_node.children
            or rewritten_node.turn is None
            or len(rewritten_node.turn.output_ids) >= self._fork_threshold
        ):
            return node, depth

        rewritten_node.metadata["merged_rewrite"] = {
            "abandoned_turn_index": rewritten_node.turn_index,
            "abandoned_response_tokens": len(rewritten_node.turn.output_ids),
        }
        rewritten_node.turn = None
        rewritten_node.turn_index = None
        rewritten_node.message = prompt_messages[depth]
        return rewritten_node, depth + 1

    def _mount_prompt_messages(
        self,
        node: MessageNode,
        remaining_messages: list[dict[str, Any]],
    ) -> None:
        for m in remaining_messages:
            node = node.add_child(MessageNode(role=m.get("role"), message=m))
        return node

    def _attach_assistant_leaf(
        self,
        sid: str,
        node: MessageNode,
        *,
        turn: TurnRecord,
        response_message: dict[str, Any] | None,
        request_messages: list[dict[str, Any]],
        request_tools: list[dict[str, Any]] | None,
        chat_template_kwargs: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
    ) -> MessageNode:
        asst = MessageNode(
            role="assistant",
            message=response_message,
            metadata=dict(metadata or {}),
        )
        asst.turn = turn
        asst.turn_index = self._turn_count.get(sid, 0) + 1
        asst.request_messages = copy.deepcopy(request_messages)
        asst.request_tools = copy.deepcopy(request_tools)
        asst.chat_template_kwargs = copy.deepcopy(chat_template_kwargs or {})
        node.add_child(asst)
        self._turn_count[sid] = asst.turn_index

    def _split_chain_into_builders(self, sid: str, chain: list[MessageNode]) -> list[_SampleBuilder]:
        """Pack the chain's generated turns into per-Sample token builders.

        Turns flow into the current builder until one can't extend it as an
        exact prefix (re-tokenization drift past what we can drop); that turn
        opens a new builder -- a fork. A generated turn shared by sibling leaves
        is trained only on the first leaf to claim it; later leaves re-emit it
        as loss_mask=0 context so the shared prefix isn't double-counted.
        """
        asst_nodes = [n for n in chain if n.role == "assistant" and n.turn is not None]
        diagnostics = self._prepare_diagnostics.setdefault(sid, _new_prepare_diagnostics())

        builders: list[_SampleBuilder] = []
        for asst_node in asst_nodes:
            trained = not asst_node.response_trained
            asst_node.response_trained = True

            if not builders:
                if trained:
                    _bump_prepare_count(diagnostics, "initial")
                builders.append(_SampleBuilder(self._fork_threshold))
                builders[-1].append_turn(
                    asst_node.turn,
                    DriftKind.CLEAN,
                    trained=trained,
                    turn_index=asst_node.turn_index,
                    request_messages=asst_node.request_messages,
                    request_tools=asst_node.request_tools,
                    chat_template_kwargs=asst_node.chat_template_kwargs,
                )
                continue

            builder = builders[-1]
            classified_kind = builder.classify_token_drift(asst_node.turn)
            realign_disabled = classified_kind is DriftKind.REALIGN and not self._enable_realign
            kind = DriftKind.FORK if realign_disabled else classified_kind
            held_len = len(builder.tokens)
            realign_at = _common_prefix_len(builder.tokens, asst_node.turn.prompt_ids)
            drift_tokens = held_len - realign_at
            last_response_start = builder.last_response_start_idx
            if trained:
                if kind is DriftKind.CLEAN:
                    _bump_prepare_count(diagnostics, "clean")
                elif kind is DriftKind.REALIGN:
                    _bump_prepare_count(diagnostics, "realign")
                    discarded = sum(builder.loss_mask[last_response_start:]) if last_response_start is not None else 0
                    diagnostics["realign_discarded_trainable_tokens"] += discarded
                    diagnostics["events"].append({
                        "turn_index": asst_node.turn_index,
                        "action": "realign",
                        "held_len": held_len,
                        "prompt_len": len(asst_node.turn.prompt_ids),
                        "realign_at": realign_at,
                        "drift_tokens": drift_tokens,
                        "last_response_start": last_response_start,
                        "current_output_len": len(asst_node.turn.output_ids),
                        "discarded_trainable_tokens": discarded,
                    })
                else:
                    _bump_prepare_count(diagnostics, "token_fork")
                    if asst_node.turn.segment_boundary_reason:
                        fork_reason = "segment_boundary"
                    elif realign_disabled:
                        fork_reason = "realign_disabled"
                    else:
                        fork_reason = "drift_before_last_response"
                    diagnostics["events"].append({
                        "turn_index": asst_node.turn_index,
                        "action": "token_fork",
                        "reason": fork_reason,
                        "segment_boundary_reason": asst_node.turn.segment_boundary_reason,
                        "held_len": held_len,
                        "prompt_len": len(asst_node.turn.prompt_ids),
                        "realign_at": realign_at,
                        "drift_tokens": drift_tokens,
                        "last_response_start": last_response_start,
                        "current_output_len": len(asst_node.turn.output_ids),
                    })
            if kind is DriftKind.FORK:
                builders.append(_SampleBuilder(self._fork_threshold))
                kind = DriftKind.CLEAN
            builders[-1].append_turn(
                asst_node.turn,
                kind,
                trained=trained,
                turn_index=asst_node.turn_index,
                request_messages=asst_node.request_messages,
                request_tools=asst_node.request_tools,
                chat_template_kwargs=asst_node.chat_template_kwargs,
            )
        return builders

    def _chain_to_samples(
        self,
        sid: str,
        chain: list[MessageNode],
        *,
        base_sample: Sample,
        extra_metadata: dict[str, Any] | None,
        max_sample_tokens: int = 0,
        credit_chain_index: int = 0,
        routing_leaf_count: int = 1,
    ) -> list[Sample]:

        asst_nodes = [n for n in chain if n.role == "assistant" and n.turn is not None]
        truncated = bool(asst_nodes) and asst_nodes[-1].turn.finish_reason == "length"
        use_tool = any(bool((n.message or {}).get("tool_calls")) for n in asst_nodes)
        ill_formed = any(n.turn.ill_formed for n in asst_nodes)
        md = {
            **(extra_metadata or {}),
            "truncated": truncated,
            "use_tool": use_tool,
            "ill_formed": ill_formed,
        }
        samples = [
            builder.to_sample(base_sample, md, max_sample_tokens)
            for builder in self._split_chain_into_builders(sid, chain)
            if builder.has_trained_response()
        ]
        samples = [sample for sample in samples if sample.loss_mask and any(sample.loss_mask)]
        sequential = routing_leaf_count == 1
        segment_count = len(samples)
        relation = (
            "single_segment"
            if sequential and segment_count <= 1
            else "sequential_token_fork"
            if sequential
            else "message_tree_branch"
        )
        for segment_index, sample in enumerate(samples):
            metadata = dict(sample.metadata or {})
            clawgym_metadata = dict(metadata.get("clawgym_nanobot") or {})
            clawgym_metadata["cross_trajectory"] = {
                "schema_version": 1,
                "credit_chain_index": int(credit_chain_index),
                "segment_index": int(segment_index),
                "segment_count": int(segment_count),
                "is_terminal_segment": segment_index == segment_count - 1,
                "eligible": bool(sequential),
                "relation": relation,
            }
            metadata["clawgym_nanobot"] = clawgym_metadata
            sample.metadata = metadata
        return samples


__all__ = [
    "TrajectoryManager",
    "TurnRecord",
]
