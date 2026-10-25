# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

API_URL = "https://api.runpod.io/v2"
CPU_FLAVOR = "cpu3c"
VCPUS = 16
RAM_GB = 32
DISK_GB = 100
IMAGE = "ghcr.io/astral-sh/uv:python3.12-trixie-slim"
PROJECT = Path(__file__).resolve().parent
REMOTE_PROJECT = "/workspace/benchmark"

BOOT_COMMAND = r"""set -eu
apt-get update >&2
apt-get install -y --no-install-recommends openssh-server git ca-certificates tar >&2
mkdir -p /run/sshd /root/.ssh /workspace/benchmark
chmod 700 /root/.ssh
printf '%s\n' "$BENCHMARK_SSH_PUBLIC_KEY" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
ssh-keygen -A
exec /usr/sbin/sshd -D -e \
    -o PasswordAuthentication=no \
    -o KbdInteractiveAuthentication=no \
    -o PermitRootLogin=prohibit-password \
    -o UsePAM=no
"""


def log(message: str) -> None:
    print(f"[benchmark] {message}", file=sys.stderr, flush=True)


def api(method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{API_URL}/{path}",
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}", "Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read() or b"{}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a command on a disposable Runpod CPU Pod (16 vCPUs, 32 GB RAM).",
        epilog="Example: uv run benchmark.py -- python benchmarks/bench_dataloader.py --n-epochs 1",
    )
    parser.add_argument("--startup-timeout", type=float, default=600, metavar="SECONDS", help="Pod/SSH readiness timeout (default: 600)")
    parser.add_argument("--timeout", type=float, default=3600, metavar="SECONDS", help="Benchmark command timeout (default: 3600)")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to execute in the remote project's uv environment, after --")
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("provide a command after --")
    if args.timeout <= 0 or args.startup_timeout <= 0:
        parser.error("timeouts must be positive")
    return args


def archive_project(destination: Path) -> None:
    files = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=PROJECT)
    with tarfile.open(destination, "w:gz") as archive:
        for name in sorted(set(os.fsdecode(files).split("\0")) - {""}):
            relative = Path(name)
            if any(part in {".git", ".venv", ".ssh", ".aws", "__pycache__"} or part == ".env" or part.startswith(".env.") for part in relative.parts):
                continue
            if relative.suffix in {".pem", ".key"} or relative.name in {"id_rsa", "id_ed25519"}:
                continue
            path = PROJECT / relative
            if not path.exists():
                continue
            if path.is_symlink() and (path.readlink().is_absolute() or not path.resolve().is_relative_to(PROJECT)):
                raise RuntimeError(f"Cannot copy symlink outside the project: {relative}")
            if path.is_file() or path.is_symlink():
                archive.add(path, arcname=name, recursive=False)


def ssh_command(endpoint: dict[str, Any], key: Path) -> list[str]:
    return [
        "ssh",
        "-F",
        "/dev/null",
        "-T",
        "-i",
        str(key),
        "-p",
        str(endpoint["port"]),
        "-oBatchMode=yes",
        "-oIdentitiesOnly=yes",
        "-oStrictHostKeyChecking=accept-new",
        f"-oUserKnownHostsFile={key.parent / 'known_hosts'}",
        "-oConnectTimeout=10",
        "-oServerAliveInterval=15",
        "-oServerAliveCountMax=3",
        "-oLogLevel=ERROR",
        f"{endpoint['username']}@{endpoint['host']}",
    ]


def wait_for_ssh(pod_id: str, key: Path, timeout: float) -> list[str]:
    deadline = time.monotonic() + timeout
    last_error = "No direct SSH address assigned yet"
    while time.monotonic() < deadline:
        pod = api("GET", f"pods/{pod_id}")
        if pod["status"] in {"ERROR", "EXITED", "TERMINATED"}:
            raise RuntimeError(f"Pod {pod_id} entered {pod['status']} before SSH was ready")
        if pod["cpu"]["vcpuCount"] != VCPUS or pod["cpu"]["memory"] != RAM_GB:
            raise RuntimeError(f"Unexpected Pod allocation: {pod['cpu']}")
        endpoint = pod["ssh"]["direct"]
        if endpoint:
            command = ssh_command(endpoint, key)
            try:
                probe = subprocess.run(
                    [*command, "printf benchmark-ready"],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    timeout=15,
                )
                if probe.returncode == 0 and probe.stdout == b"benchmark-ready":
                    return command
                last_error = probe.stderr.decode(errors="replace").strip()
            except subprocess.TimeoutExpired:
                last_error = "SSH connection timed out"
        time.sleep(3)
    raise RuntimeError(f"Pod {pod_id} was not ready within {timeout:g}s: {last_error}")


def interrupted(signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt(signum)


def cleanup(pod_id: str | None, name: str) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    # Recover a Pod if creation succeeded but its response was lost or interrupted.
    if pod_id is None:
        pod_id = next((pod["id"] for pod in api("GET", "pods")["pods"] if pod["name"] == name), None)
    if pod_id is None:
        return
    log(f"Terminating Pod {pod_id}")
    for attempt in range(4):
        try:
            api("DELETE", f"pods/{pod_id}")
            if api("GET", f"pods/{pod_id}")["status"] == "TERMINATED":
                return
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return
            if attempt == 3:
                raise
        except OSError:
            if attempt == 3:
                raise
        time.sleep(2**attempt)
    raise RuntimeError(f"Could not confirm termination of Pod {pod_id}")


def run(args: argparse.Namespace, directory: Path) -> int:
    key = directory / "id_ed25519"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True)
    archive = directory / "project.tar.gz"
    archive_project(archive)

    flavor = api("GET", f"catalog/cpus/{CPU_FLAVOR}")
    price = flavor["price"]["securePerVcpu"] * VCPUS
    log(f"Creating {VCPUS}-vCPU / {RAM_GB}-GB CPU Pod ({CPU_FLAVOR}, ${price:.2f}/hour compute, plus {DISK_GB} GB disk)")

    name = f"benchmark-{uuid.uuid4().hex}"
    log(f"Pod request name: {name}")
    pod_id: str | None = None
    exit_code = 1
    try:
        pod = api(
            "POST",
            "pods",
            {
                "name": name,
                "cloud": "SECURE",
                "cpu": {"id": CPU_FLAVOR, "vcpuCount": VCPUS},
                "image": IMAGE,
                "entrypoint": ["/bin/sh", "-c"],
                "cmd": [BOOT_COMMAND],
                "disk": DISK_GB,
                "ports": ["22/tcp"],
                "env": {"BENCHMARK_SSH_PUBLIC_KEY": key.with_suffix(".pub").read_text().strip(), "PYTHONUNBUFFERED": "1"},
            },
        )
        pod_id = pod["id"]
        log(f"Pod {pod_id}; waiting for SSH")
        ssh = wait_for_ssh(pod_id, key, args.startup_timeout)

        log("Copying checkout (including uncommitted files; excluding ignored files and common credential files)")
        with archive.open("rb") as source:
            subprocess.run([*ssh, f"tar -xzf - -C {REMOTE_PROJECT}"], stdin=source, check=True, timeout=600)
        log("Installing locked Python dependencies")
        subprocess.run(
            [*ssh, f"cd {REMOTE_PROJECT} && uv sync --locked --no-dev"],
            stdin=subprocess.DEVNULL,
            stdout=sys.stderr,
            check=True,
            timeout=900,
        )
        log(f"Running: {shlex.join(args.command)}")
        command = f"cd {REMOTE_PROJECT} && exec env PYTHONUNBUFFERED=1 uv run --no-sync -- {shlex.join(args.command)}"
        result = subprocess.run([*ssh, command], stdin=subprocess.DEVNULL, timeout=args.timeout)
        exit_code = result.returncode if result.returncode >= 0 else 128 - result.returncode
    finally:
        try:
            cleanup(pod_id, name)
        except (OSError, RuntimeError, ValueError) as error:
            log(f"Cleanup failed: {error}. Check the Runpod console for {pod_id or name} and terminate it manually.")
            if exit_code == 0:
                exit_code = 1
        finally:
            signal.signal(signal.SIGINT, interrupted)
            signal.signal(signal.SIGTERM, interrupted)
    return exit_code


def main() -> int:
    args = parse_args()
    if not os.environ.get("RUNPOD_API_KEY"):
        log("Set RUNPOD_API_KEY to a Runpod API key before running this script")
        return 1
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with tempfile.TemporaryDirectory(prefix="cabo-benchmark-") as directory:
            return run(args, Path(directory))
    except KeyboardInterrupt as error:
        signum = error.args[0] if error.args else signal.SIGINT
        log("Interrupted")
        return 128 + signum
    except subprocess.TimeoutExpired as error:
        log(f"Command timed out after {error.timeout:g}s")
        return 124
    except urllib.error.HTTPError as error:
        log(f"{error}: {error.read().decode(errors='replace')}")
        return 1
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        log(str(error))
        return 1


if __name__ == "__main__":
    sys.exit(main())
