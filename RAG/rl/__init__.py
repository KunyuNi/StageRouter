"""RAG-local cache and trajectory primitives.

The modules in this package intentionally do not depend on ``nl2sql``.  They
share the useful persistence boundaries from that project while using the
seven-stage MultiHop-RAG workflow and its own cache identities.
"""

from .stage_mapping import (
    DECISION_TO_PHYSICAL,
    FIXED_STAGE_DEFAULT_TIER,
    FIXED_STAGE_IDXS,
    PHYSICAL_STAGE_NAMES,
)
from .workflow_identity import WorkflowIdentity
from .transition_cache import TransitionCache, TransitionKey
from .trajectory_buffer import TrajectoryBuffer, TrajectoryRecord
from .route_result_cache import RouteResultCache, route_key
from .fixed_route_collector import (
    BASELINE_ORDER,
    FIXED_BASELINE_ROUTES,
    FIXED_ROUTES,
    FixedRouteCollector,
    collect_route,
)
from .query_filter import (
    QUERY_FILTER_POLICY,
    filter_trajectory_file,
    invalid_query_ids,
    latest_route_statuses,
)
from .env_wrapper import (
    RAGEnvironmentWrapper,
    STAGE_INPUT_FIELDS,
    compute_stage_input_hash,
)
from .decision_session import (
    DECISION_BUNDLES,
    LEGAL_MASKS,
    DecisionObservation,
    DecisionStepResult,
    RAGDecisionSession,
)
from .counterfactual_route_ladder import (
    ANCHOR_PROFILE,
    get_route_profile,
)
from .state_v1_schema import (
    STATE_V1_DIM,
    STATE_V1_SCHEMA,
    STATE_V1_SCHEMA_ID,
    StateV1Blocks,
    legal_action_mask,
)
from .state_v1_encoder import EncodedState, StateV1Encoder, StateV1Normalizer
from .prepared_artifacts import PreparedReplayDataset, PreparedReplayStore

__all__ = [
    "PHYSICAL_STAGE_NAMES",
    "DECISION_TO_PHYSICAL",
    "FIXED_STAGE_IDXS",
    "FIXED_STAGE_DEFAULT_TIER",
    "WorkflowIdentity",
    "TransitionCache",
    "TransitionKey",
    "TrajectoryBuffer",
    "TrajectoryRecord",
    "RouteResultCache",
    "route_key",
    "RAGEnvironmentWrapper",
    "STAGE_INPUT_FIELDS",
    "compute_stage_input_hash",
    "DECISION_BUNDLES",
    "LEGAL_MASKS",
    "DecisionObservation",
    "DecisionStepResult",
    "RAGDecisionSession",
    "FIXED_ROUTES",
    "FIXED_BASELINE_ROUTES",
    "BASELINE_ORDER",
    "FixedRouteCollector",
    "collect_route",
    "QUERY_FILTER_POLICY",
    "filter_trajectory_file",
    "invalid_query_ids",
    "latest_route_statuses",
    "ANCHOR_PROFILE",
    "get_route_profile",
    "STATE_V1_DIM",
    "STATE_V1_SCHEMA",
    "STATE_V1_SCHEMA_ID",
    "StateV1Blocks",
    "legal_action_mask",
    "EncodedState",
    "StateV1Encoder",
    "StateV1Normalizer",
    "PreparedReplayDataset",
    "PreparedReplayStore",
]
