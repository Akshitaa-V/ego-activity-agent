"""Command line: ``python -m egoagent demo`` and ``python -m egoagent ask "..."``."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from .agent import Agent, build_tools
from .llm import OpenAICompatibleLLM, RuleBasedLLM
from .pipeline import PipelineConfig, run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="egoagent", description="Egocentric activity agent")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the full experiment and build the moment index")
    demo.add_argument("--sessions", type=int, default=32)
    demo.add_argument("--epochs", type=int, default=8)
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--ddp", type=int, default=2, help="DDP processes; 0 to skip")
    demo.add_argument("--quick", action="store_true", help="tiny run for a smoke test")
    demo.add_argument("--out", default="artifacts")

    ask = sub.add_parser("ask", help="ask the agent about the indexed sessions")
    ask.add_argument("question")
    ask.add_argument(
        "--llm",
        choices=["rule", "openai"],
        default="rule",
        help="'openai' uses any OpenAI-compatible endpoint (see README)",
    )
    ask.add_argument("--out", default="artifacts")

    args = parser.parse_args(argv)
    if args.command == "demo":
        cfg = PipelineConfig(
            n_sessions=args.sessions, epochs=args.epochs, seed=args.seed, ddp_world_size=args.ddp, out_dir=args.out
        )
        if args.quick:
            cfg.n_sessions, cfg.duration_s, cfg.epochs = 8, 30.0, 2
        run_pipeline(cfg)
        return 0

    index_path = Path(args.out) / "index.pkl"
    if not index_path.exists():
        parser.error(f"{index_path} not found; run 'python -m egoagent demo' first")
    with open(index_path, "rb") as f:
        index = pickle.load(f)  # noqa: S301 - file written by this tool
    llm = OpenAICompatibleLLM() if args.llm == "openai" else RuleBasedLLM()
    result = Agent(llm, build_tools(index)).run(args.question)
    for step in result.trace:
        status = "ok" if step.output.get("ok") else "error"
        print(f"[tool] {step.tool}({step.arguments}) -> {status} in {step.latency_ms:.1f} ms")
    print(result.answer)
    return 0
