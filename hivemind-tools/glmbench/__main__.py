import argparse
import getpass
import json
import os
from pathlib import Path
from .client import Client


def main():
    parser = argparse.ArgumentParser(
        description="Model-independent Hivemind team benchmark"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("coordinator", "agent", "upload", "doctor"):
        p = sub.add_parser(name)
        if name != "doctor":
            p.add_argument(
                "--api",
                required=True,
                help="Backend HTTPS origin; never put keys in the URL",
            )
        if name in ("coordinator", "agent"):
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument(
                "--port", type=int, default=40100 if name == "coordinator" else 0
            )
        if name == "coordinator":
            p.add_argument(
                "--identity",
                help="Optional persistent peer identity file; defaults to a fresh identity per lobby",
            )
        if name in ("agent", "doctor"):
            p.add_argument(
                "--device", choices=["auto", "cuda", "rocm", "cpu"], default="auto"
            )
            p.add_argument(
                "--device-index",
                type=int,
                default=0,
                help="GPU index within CUDA/ROCm visible devices",
            )
        if name == "agent":
            p.add_argument("--workload-factory", help="Explicit local module:function for an external project; never loaded from a server-supplied import path")
            p.add_argument("--max-parameters", type=int, default=25_000_000, help="Local consent limit for the model or gradient payload; increase deliberately for larger projects")
            p.add_argument("--ready", action="store_true")
            p.add_argument("--output", default="results")
            p.add_argument("--threads", type=int, default=2)
            p.add_argument("--local-steps", type=int, default=None, help="Local optimizer steps per Hivemind sync")
        if name == "upload":
            p.add_argument("--file", required=True)
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            from .device import doctor

            doctor(args)
            return
        key = os.environ.get("GLM_TEAM_KEY") or os.environ.get("GATOR_TEAM_KEY") or getpass.getpass(
            "Your private team access key: "
        )
        client = Client(args.api, key)
        if args.command == "upload":
            data = json.loads(Path(args.file).read_text())
            client.post(
                action="result",
                sessionId=data["sessionId"],
                instanceId=data["instanceId"],
                result=data["result"],
            )
            print("Result uploaded")
        else:
            from .benchmark import coordinator, agent

            if args.command == "agent" and (args.threads < 1 or args.device_index < 0 or not 1 <= args.max_parameters <= 2_000_000_000):
                raise ValueError(
                    "--threads must be positive, --device-index nonnegative, and --max-parameters between 1 and 2B"
                )
            (coordinator if args.command == "coordinator" else agent)(args, client)
    except KeyboardInterrupt:
        print("Stopped. An interrupted benchmark is not a complete result.")
    except (RuntimeError, ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
