from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from prompt_compiler.env import load_env_file
from prompt_compiler.eval.contract_checks import OutputContract
from prompt_compiler.eval.embedding_distance import DEFAULT_EMBEDDING_MODEL, make_drift_scorer
from prompt_compiler.models.mock import RuleBasedMockModel
from prompt_compiler.models.openai_client import OpenAIResponsesModel
from prompt_compiler.models.base import GenerateParams, ModelClient
from prompt_compiler.optimize.optimizer import optimize_prompt
from prompt_compiler.operators.proposer import LLMRewriteProposer, RewriteProposer
from prompt_compiler.prompt.template import PromptTemplate
from prompt_compiler.tokenizer import make_tokenizer


DEFAULT_CHUNKERS = "paragraph,sentence,markdown,schema_aware"
DEFAULT_OPENAI_PROPOSER_MODEL = "gpt-5.4-mini-2026-03-17"
DEFAULT_OPENAI_PROPOSER_REASONING_EFFORT = "medium"


def main() -> int:
    parser = argparse.ArgumentParser(description="Behavioral prompt compression compiler")
    parser.add_argument("--model", default="mock", help="Model id. Use 'mock' or an OpenAI model id.")
    parser.add_argument("--provider", default="auto", choices=("auto", "mock", "openai"))
    parser.add_argument("--proposer", default="auto", choices=("auto", "rule", "openai"))
    parser.add_argument(
        "--proposer-model",
        default=None,
        help=f"Model used to propose prompt variations. Defaults to {DEFAULT_OPENAI_PROPOSER_MODEL} for OpenAI proposers.",
    )
    parser.add_argument("--proposer-max-output-tokens", type=int, default=16384)
    parser.add_argument("--proposer-reasoning-effort", default=DEFAULT_OPENAI_PROPOSER_REASONING_EFFORT)
    parser.add_argument("--prompt", required=True, help="Path to original prompt template")
    parser.add_argument("--inputs", required=True, help="JSONL file with {'id','input'} rows")
    parser.add_argument("--output-dir", required=True, help="Directory for run artifacts")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--input-limit", type=int, default=None, help="Use only the first N inputs from the JSONL file")
    parser.add_argument("--require-json", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--reasoning-effort", default=None, help="OpenAI reasoning effort, e.g. minimal, low, medium")
    parser.add_argument(
        "--send-openai-sampling-params",
        action="store_true",
        help="Send temperature/top_p to OpenAI Responses. GPT-5 Nano currently rejects these controls.",
    )
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--env-file", default=".env.local", help="Local env file to load before adapter setup")
    parser.add_argument("--max-concurrency", type=int, default=1, help="Maximum candidates evaluated concurrently")
    parser.add_argument(
        "--chunkers",
        default=DEFAULT_CHUNKERS,
        help=(
            "Comma-separated chunkers to explore. Available: paragraph,sentence,markdown,"
            "schema_aware,instruction_role,token_window"
        ),
    )
    parser.add_argument(
        "--live-log-file",
        default="runs/live_run_events.jsonl",
        help="Stable JSONL log mirrored for live tailing. Use '' to disable.",
    )
    parser.add_argument("--quiet", action="store_true", help="Write run_events.jsonl without echoing progress logs")
    parser.add_argument(
        "--tokenizer",
        default="approx",
        help="Tokenizer spec: approx, tiktoken:<encoding>, or model:<model-name>",
    )
    parser.add_argument(
        "--embedding-provider",
        default="lexical",
        choices=("lexical", "sentence-transformers", "hf-inference"),
        help="Drift scorer backend. Use hf-inference or sentence-transformers for Mixedbread embeddings.",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--hf-provider", default=None, help="Optional Hugging Face Inference Provider name")
    args = parser.parse_args()

    load_env_file(Path(args.env_file))

    prompt = PromptTemplate(Path(args.prompt).read_text(encoding="utf-8"))
    inputs = _read_jsonl(Path(args.inputs))
    if args.input_limit is not None:
        inputs = inputs[: args.input_limit]
    tokenizer = make_tokenizer(args.tokenizer)
    model = _build_model(args)
    rewrite_proposer = _build_rewrite_proposer(args, prompt, tokenizer)
    chunker_names = _parse_csv(args.chunkers)
    live_log_path = Path(args.live_log_file) if args.live_log_file else None
    params = GenerateParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_output_tokens,
        system_prompt=args.system_prompt,
        reasoning_effort=args.reasoning_effort,
    )
    result = optimize_prompt(
        target_model=model,
        original_prompt=prompt,
        inputs=inputs,
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        population_size=args.population_size,
        tokenizer=tokenizer,
        drift_scorer=make_drift_scorer(
            args.embedding_provider,
            model_name=args.embedding_model,
            api_key=os.environ.get("HF_TOKEN"),
            hf_provider=args.hf_provider,
        ),
        output_contract=OutputContract(require_json=args.require_json),
        params=params,
        rewrite_proposer=rewrite_proposer,
        max_concurrency=args.max_concurrency,
        log_to_stderr=not args.quiet,
        live_log_path=live_log_path,
        chunker_names=chunker_names,
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


def _parse_csv(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    return items or None


def _build_model(args: argparse.Namespace) -> ModelClient:
    provider = _target_provider(args)
    if provider == "mock":
        if args.model not in {"mock", "rule-based-mock"}:
            raise SystemExit("--provider mock only supports --model mock")
        return RuleBasedMockModel()
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY was not found in the environment or env file.")
        return OpenAIResponsesModel(name=args.model, send_sampling_params=args.send_openai_sampling_params)
    raise SystemExit(f"Unknown provider: {provider}")


def _build_rewrite_proposer(args: argparse.Namespace, prompt: PromptTemplate, tokenizer) -> RewriteProposer | None:
    proposer = args.proposer
    if proposer == "auto":
        proposer = "rule" if _target_provider(args) == "mock" else "openai"
    if proposer == "rule":
        return None
    if proposer == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY was not found in the environment or env file.")
        proposer_model_name = args.proposer_model or DEFAULT_OPENAI_PROPOSER_MODEL
        if not proposer_model_name:
            raise SystemExit("--proposer-model is required when --proposer openai and target model is not OpenAI.")
        proposer_model = OpenAIResponsesModel(
            name=proposer_model_name,
            send_sampling_params=args.send_openai_sampling_params,
        )
        return LLMRewriteProposer(
            model=proposer_model,
            original_prompt=prompt.text,
            target_model_name=args.model,
            tokenizer=tokenizer,
            params=GenerateParams(
                max_tokens=args.proposer_max_output_tokens,
                reasoning_effort=args.proposer_reasoning_effort,
            ),
            trace_path=Path(args.output_dir) / "proposer_traces.jsonl",
            event_log_path=Path(args.output_dir) / "run_events.jsonl",
            event_log_paths=(Path(args.live_log_file),) if args.live_log_file else (),
        )
    raise SystemExit(f"Unknown proposer: {proposer}")


def _target_provider(args: argparse.Namespace) -> str:
    if args.provider == "auto":
        return "mock" if args.model == "mock" else "openai"
    return args.provider


if __name__ == "__main__":
    raise SystemExit(main())
