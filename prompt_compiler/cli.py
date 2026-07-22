from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from prompt_compiler.env import load_env_file
from prompt_compiler.eval.contract_checks import OutputContract
from prompt_compiler.eval.embedding_distance import DEFAULT_EMBEDDING_MODEL, make_drift_scorer
from prompt_compiler.eval.evaluator import EvaluationWeights
from prompt_compiler.models.base import GenerateParams, ModelClient
from prompt_compiler.models.mock import RuleBasedMockModel
from prompt_compiler.models.openai_client import OpenAIResponsesModel
from prompt_compiler.operators.full_prompt_proposer import (
    LLMFullPromptProposer,
    ProposalContext,
)
from prompt_compiler.optimize.feedback_optimizer import optimize_prompt
from prompt_compiler.prompt.template import PromptTemplate
from prompt_compiler.tokenizer import make_tokenizer


DEFAULT_OPENAI_PROPOSER_MODEL = "gpt-5.4-mini-2026-03-17"
DEFAULT_OPENAI_PROPOSER_REASONING_EFFORT = "medium"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Feedback-conditioned behavioral prompt compression"
    )
    parser.add_argument("--model", default="mock", help="Target model id, or 'mock'.")
    parser.add_argument("--provider", default="auto", choices=("auto", "mock", "openai"))
    parser.add_argument("--proposer-model", default=DEFAULT_OPENAI_PROPOSER_MODEL)
    parser.add_argument("--proposer-max-output-tokens", type=int, default=None)
    parser.add_argument(
        "--proposer-reasoning-effort",
        default=DEFAULT_OPENAI_PROPOSER_REASONING_EFFORT,
    )
    parser.add_argument("--prompt", required=True, help="Path to the original prompt template.")
    parser.add_argument("--inputs", required=True, help="JSONL file with {'id','input'} rows.")
    parser.add_argument("--output-dir", required=True, help="Directory for run artifacts.")
    parser.add_argument("--rounds", "--epochs", dest="rounds", type=int, default=8)
    parser.add_argument("--batch-size", "--population-size", dest="batch_size", type=int, default=8)
    parser.add_argument("--convergence-patience", type=int, default=3)
    parser.add_argument("--min-frontier-improvement", type=float, default=1e-4)
    parser.add_argument("--frontier-parent-limit", type=int, default=None)
    parser.add_argument("--recent-contrast-limit", type=int, default=None)
    parser.add_argument("--worst-example-limit", type=int, default=None)
    parser.add_argument("--repeat-top-k", type=int, default=None)
    parser.add_argument("--max-candidate-rollouts", type=int, default=2)
    parser.add_argument("--repeat-close-within", type=float, default=0.02)
    parser.add_argument("--baseline-repeats", type=int, default=2)
    parser.add_argument(
        "--selection-behavior-penalty",
        type=float,
        default=1.0,
        help="Explicit behavior-loss penalty used only to select one prompt from the final Pareto frontier.",
    )
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="Ablation: withhold measured candidate outcomes from later proposal rounds.",
    )
    parser.add_argument(
        "--preview-proposals",
        action="store_true",
        help="Generate one initial full-prompt proposal batch without target-model evaluation.",
    )
    parser.add_argument(
        "--example-limit",
        "--input-limit",
        dest="input_limit",
        type=int,
        default=None,
    )
    parser.add_argument("--require-json", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--send-openai-sampling-params", action="store_true")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--env-file", default=".env.local")
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--live-log-file", default="runs/live_run_events.jsonl")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--tokenizer",
        default="auto",
        help="Tokenizer spec: auto, approx, tiktoken:<encoding>, or model:<model-name>.",
    )
    parser.add_argument(
        "--embedding-provider",
        default="auto",
        choices=("auto", "lexical", "sentence-transformers", "hf-inference"),
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--hf-provider", default=None)
    parser.add_argument("--semantic-drift-normalization", type=float, default=2.0)
    args = parser.parse_args()

    load_env_file(Path(args.env_file))
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY was not found in the environment or env file.")

    prompt = PromptTemplate(Path(args.prompt).read_text(encoding="utf-8"))
    inputs = _read_jsonl(Path(args.inputs))
    if args.input_limit is not None:
        inputs = inputs[: args.input_limit]
    tokenizer = make_tokenizer(_tokenizer_spec(args))
    proposer = _build_prompt_proposer(args, tokenizer)

    if args.preview_proposals:
        proposals = proposer.propose(
            ProposalContext(
                original_prompt=prompt.text,
                target_model_name=args.model,
                target_tokenizer_name=tokenizer.name,
            ),
            batch_size=args.batch_size,
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "initial_proposals.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for proposal in proposals:
                handle.write(json.dumps(proposal.__dict__, ensure_ascii=False, sort_keys=True) + "\n")
        print(json.dumps({"proposal_count": len(proposals), "path": str(path)}, indent=2))
        return 0

    model = _build_model(args)
    params = GenerateParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_output_tokens,
        system_prompt=args.system_prompt,
        reasoning_effort=args.reasoning_effort,
    )
    result = optimize_prompt(
        target_model=model,
        prompt_proposer=proposer,
        original_prompt=prompt,
        inputs=inputs,
        output_dir=Path(args.output_dir),
        rounds=args.rounds,
        batch_size=args.batch_size,
        convergence_patience=args.convergence_patience,
        min_frontier_improvement=args.min_frontier_improvement,
        parent_limit=args.frontier_parent_limit,
        recent_limit=args.recent_contrast_limit,
        worst_example_limit=args.worst_example_limit,
        repeat_top_k=args.repeat_top_k,
        max_candidate_rollouts=args.max_candidate_rollouts,
        repeat_close_within=args.repeat_close_within,
        baseline_repeats=args.baseline_repeats,
        feedback_enabled=not args.no_feedback,
        selection_behavior_penalty=args.selection_behavior_penalty,
        tokenizer=tokenizer,
        evaluation_weights=EvaluationWeights(
            semantic_drift_normalization=args.semantic_drift_normalization
        ),
        drift_scorer=make_drift_scorer(
            _embedding_provider(args.embedding_provider, inputs),
            model_name=args.embedding_model,
            api_key=os.environ.get("HF_TOKEN"),
            hf_provider=args.hf_provider,
        ),
        output_contract=OutputContract(require_json=args.require_json),
        params=params,
        max_concurrency=args.max_concurrency,
        log_to_stderr=not args.quiet,
        live_log_path=Path(args.live_log_file) if args.live_log_file else None,
    )
    print(json.dumps(result.evaluation_report.to_dict(), ensure_ascii=False, indent=2))
    print(f"best_prompt={Path(args.output_dir) / 'best_prompt.txt'}")
    return 0


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _build_model(args: argparse.Namespace) -> ModelClient:
    provider = _target_provider(args)
    if provider == "mock":
        if args.model not in {"mock", "rule-based-mock"}:
            raise SystemExit("--provider mock only supports --model mock")
        return RuleBasedMockModel()
    if provider == "openai":
        return OpenAIResponsesModel(
            name=args.model,
            send_sampling_params=args.send_openai_sampling_params,
        )
    raise SystemExit(f"Unknown provider: {provider}")


def _build_prompt_proposer(args: argparse.Namespace, tokenizer) -> LLMFullPromptProposer:
    model = OpenAIResponsesModel(
        name=args.proposer_model,
        send_sampling_params=args.send_openai_sampling_params,
    )
    return LLMFullPromptProposer(
        model=model,
        tokenizer=tokenizer,
        params=GenerateParams(
            max_tokens=args.proposer_max_output_tokens,
            reasoning_effort=args.proposer_reasoning_effort,
        ),
        trace_path=Path(args.output_dir) / "proposer_traces.jsonl",
    )


def _target_provider(args: argparse.Namespace) -> str:
    if args.provider == "auto":
        return "mock" if args.model == "mock" else "openai"
    return args.provider


def _tokenizer_spec(args: argparse.Namespace) -> str:
    if args.tokenizer != "auto":
        return args.tokenizer
    if _target_provider(args) == "openai":
        return f"model:{args.model}"
    return "approx"


def _embedding_provider(provider: str, inputs: list[dict]) -> str:
    if provider != "auto":
        return provider
    if inputs and all(isinstance(row.get("expected"), dict) for row in inputs):
        return "lexical"
    return "sentence-transformers"


if __name__ == "__main__":
    raise SystemExit(main())
