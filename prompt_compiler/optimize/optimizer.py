"""Public optimizer API.

The historical chunk/operator implementation lives in ``legacy_optimizer``
for controlled ablations. New runs use feedback-conditioned full-prompt search.
"""

from prompt_compiler.optimize.feedback_optimizer import (
    FeedbackEvaluationReport,
    FeedbackOptimizationResult,
    optimize_prompt,
)
from prompt_compiler.optimize.legacy_optimizer import optimize_prompt_legacy

__all__ = [
    "FeedbackEvaluationReport",
    "FeedbackOptimizationResult",
    "optimize_prompt",
    "optimize_prompt_legacy",
]
