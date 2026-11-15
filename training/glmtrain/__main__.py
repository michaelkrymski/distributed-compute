import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="GLM: trusted-team collaborative Hivemind training"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser(
        "prepare", help="Train a tokenizer and create one immutable run bundle"
    )
    prepare.add_argument("--train", required=True)
    prepare.add_argument("--validation", required=True)
    prepare.add_argument("--config", default="configs/baseline.json")
    prepare.add_argument("--output", required=True)
    bootstrap = sub.add_parser(
        "bootstrap", help="Keep a lightweight rendezvous node online"
    )
    train = sub.add_parser("train", help="Contribute one full model replica")
    status = sub.add_parser("status", help="List currently visible workers")
    inspect = sub.add_parser(
        "inspect", help="Verify the bundle and show the parameter count"
    )
    for p in (bootstrap, train, status, inspect):
        p.add_argument("--bundle", required=True)
    for p in (bootstrap, train, status):
        p.add_argument(
            "--host",
            default="127.0.0.1",
            help="Specific local LAN/VPN IP (loopback by default)",
        )
        p.add_argument("--port", type=int, default=40100 if p is bootstrap else 0)
    for p in (train, status):
        p.add_argument(
            "--peer",
            action="append",
            required=True,
            help="Bootstrap multiaddr; repeat for multiple peers",
        )
    bootstrap.add_argument("--identity", default="output/bootstrap.key")
    bootstrap.add_argument("--peers-file", default="output/peers.json")
    train.add_argument("--name", required=True, help="Display name for logs")
    train.add_argument(
        "--shard",
        type=int,
        required=True,
        help="Unique assigned shard index, starting at zero",
    )
    train.add_argument(
        "--output", required=True, help="Worker-specific local checkpoint directory"
    )
    train.add_argument("--microbatch", type=int, default=4)
    train.add_argument("--local-steps", type=int, default=None, help="Local optimizer steps per Hivemind sync (default: config local_steps or 1)")
    train.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    train.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    train.add_argument("--threads", type=int, default=2)
    mode = train.add_mutually_exclusive_group()
    mode.add_argument(
        "--start-new",
        action="store_true",
        help="First worker only; initialize a fresh model",
    )
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Restore this worker's latest.pt before joining",
    )
    train.add_argument(
        "--min-peers",
        type=int,
        default=1,
        help="Startup barrier, not a permanent quorum",
    )
    train.add_argument("--join-timeout", type=float, default=120)
    train.add_argument("--matchmaking-time", type=float, default=10)
    train.add_argument("--averaging-timeout", type=float, default=120)
    train.add_argument("--checkpoint-every", type=int, default=100)
    train.add_argument(
        "--max-microsteps",
        type=int,
        help="Optional per-process safety limit for testing",
    )
    args = parser.parse_args()
    try:
        from .data import prepare as prepare_bundle, load_bundle

        if args.command == "prepare":
            manifest = prepare_bundle(
                args.train, args.validation, args.config, args.output
            )
            print(json.dumps(manifest, indent=2))
            return
        manifest = load_bundle(args.bundle)
        if args.command == "inspect":
            from .model import ModelConfig

            print(
                json.dumps(
                    {
                        **manifest,
                        "parameters": ModelConfig(
                            **manifest["model"]
                        ).parameter_count(),
                    },
                    indent=2,
                )
            )
        elif args.command == "bootstrap":
            from .network import bootstrap as run

            run(args, manifest)
        elif args.command == "train":
            if (
                args.join_timeout <= 0
                or args.matchmaking_time <= 0
                or args.averaging_timeout <= args.matchmaking_time
            ):
                raise ValueError(
                    "Timeouts must be positive; averaging timeout must exceed matchmaking time"
                )
            from .worker import train as run

            run(args, manifest)
        else:
            from .network import start_dht, check_contract, workers

            dht = start_dht(args.host, args.port, args.peer)
            try:
                check_contract(dht, manifest)
                print(json.dumps(workers(dht, manifest), indent=2))
            finally:
                dht.shutdown()
    except KeyboardInterrupt:
        print("Stopped. Resume from the last completed-epoch checkpoint.")
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
