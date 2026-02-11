#!/usr/bin/env python3
# Lazyllama — reproducible LoRA SFT tool for LLaMA-Factory (repo-tool)

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.status import Status
from rich import box

console = Console()

# ------------------------------ Constants ------------------------------

PROFILE_NAME = "lazyllama.json"

STEP_VENV = "venv"
STEP_CLONE = "clone"
STEP_INSTALL = "install"
STEP_DATASET = "dataset"
STEP_REGISTER = "register"
STEP_CONFIGS = "configs"
STEP_TRAIN = "train"
STEP_MERGE = "merge"
STEP_SMOKETEST = "smoketest"

STEPS_ORDER = [
    STEP_VENV,
    STEP_CLONE,
    STEP_INSTALL,
    STEP_DATASET,
    STEP_REGISTER,
    STEP_CONFIGS,
    STEP_TRAIN,
    STEP_MERGE,
    STEP_SMOKETEST,
]

DEFAULTS = {
    "workdir": str(Path.home() / "lazyllama_run"),
    "venv_dir": None,  # default: <workdir>/venv
    "repo_dir": None,  # default: <workdir>/LLaMA-Factory

    "model": "Qwen/Qwen3-1.7B",
    "template": "qwen3_nothink",

    "dataset_name": "custom_dataset",
    "dataset_file": "custom_dataset.json",
    "dataset_src": None,

    "auto_format": True,
    "prefer_format": "alpaca",  # alpaca|sharegpt

    "cutoff_len": 2048,
    "max_samples": 1000,
    "preproc_workers": 16,
    "dataloader_workers": 4,

    "lora_rank": 8,
    "batch_size": 1,
    "grad_accum": 8,
    "lr": "1.0e-4",
    "epochs": "3.0",
    "lr_sched": "cosine",
    "warmup_ratio": "0.1",
    "bf16": True,

    "train_outdir": "saves/qwen3-1.7b/lora/sft",
    "merged_outdir": "merged-qwen3-1.7b",

    "export_size": 5,
    "export_device": "cpu",  # cpu|auto
    "export_legacy": False,

    # behavior
    "overwrite": False,
    "yes": False,
    "smoketest": True,
}

# ------------------------------ Errors & utils ------------------------------


class LazyllamaError(RuntimeError):
    pass


def die(msg: str) -> None:
    raise LazyllamaError(msg)


def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def ensure_python_version() -> None:
    if sys.version_info < (3, 10):
        die("Python >= 3.10 is required.")


def is_windows() -> bool:
    return platform.system().lower().startswith("win")


def venv_bin_dir(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts" if is_windows() else "bin")


def venv_exe(venv_dir: Path, name: str) -> Path:
    suffix = ".exe" if is_windows() else ""
    return venv_bin_dir(venv_dir) / f"{name}{suffix}"


def venv_python(venv_dir: Path) -> Path:
    return venv_exe(venv_dir, "python")


def venv_pip(venv_dir: Path) -> Path:
    return venv_exe(venv_dir, "pip")


def run_cmd(
    args: List[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    console.print(f"[bold cyan]$[/bold cyan] {' '.join(args)}")
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=check,
        text=True,
    )


def now_run_id() -> str:
    # Example: 20260210-170523
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def bytes_to_human(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0:
            return f"{x:.1f}{u}"
        x /= 1024.0
    return f"{x:.1f}PB"


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def llama_identity(run_id: Optional[str] = None, risk: Optional[str] = None) -> None:
    llama = r"""
          z z z
           z z
        (\____/)
         ( o  o)   Lazyllama
         /  __\    fine-tuning so you don't have to
        /| |  |\
       (_|_|__|_)
    """

    title = "[bold magenta]🦙 Lazyllama[/bold magenta]"

    badges = []
    if run_id:
        badges.append(f"[bold cyan]run[/bold cyan]=[white]{run_id}[/white]")
    if risk:
        color = {"SAFE": "green", "WARN": "yellow", "RISK": "red"}.get(risk, "blue")
        badges.append(f"[bold]risk[/bold]=[{color}]{risk}[/{color}]")

    subtitle = (
        "  ".join(badges)
        if badges
        else "[dim]reproducible LoRA fine-tuning, done calmly[/dim]"
    )

    console.print(
        Panel.fit(
            f"{title}\n{subtitle}\n[dim]{llama}[/dim]",
            border_style="magenta",
        )
    )

# ------------------------------ Dataset engine ------------------------------

DatasetRecord = Dict[str, Any]


def load_dataset_any(path: Path) -> Tuple[str, List[DatasetRecord]]:
    """
    Load dataset from:
      - JSONL: one JSON object per line
      - JSON: array of objects OR {"data":[...]}
    Returns (container_format, records)
    """
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        die(f"Dataset file is empty: {path}")

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    # JSONL heuristic: multiple lines, each line is { ... }
    if len(lines) >= 2 and all(ln.startswith("{") and ln.endswith("}") for ln in lines[: min(10, len(lines))]):
        records: List[DatasetRecord] = []
        for idx, ln in enumerate(lines, start=1):
            try:
                obj = json.loads(ln)
            except json.JSONDecodeError as e:
                die(f"JSONL parse error at line {idx}: {e}")
            if not isinstance(obj, dict):
                die(f"JSONL record at line {idx} is not an object.")
            records.append(obj)
        return "jsonl", records

    # Standard JSON
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        die(f"Dataset is not valid JSON or JSONL: {e}")

    if isinstance(parsed, list):
        if not all(isinstance(i, dict) for i in parsed):
            die("JSON dataset must be a list of objects.")
        return "json", parsed

    if isinstance(parsed, dict) and "data" in parsed and isinstance(parsed["data"], list):
        if not all(isinstance(i, dict) for i in parsed["data"]):
            die("JSON dataset wrapper {'data': [...]} must contain objects.")
        return "json", parsed["data"]

    die("Unsupported dataset structure (expected JSONL or JSON list of objects).")


def detect_dataset_style(records: List[DatasetRecord]) -> str:
    """
    Returns: 'alpaca' | 'sharegpt' | 'unknown'
    """
    if not records:
        return "unknown"
    sample = records[: min(50, len(records))]

    # ShareGPT: messages or conversations
    for r in sample:
        if isinstance(r.get("messages"), list) and r["messages"] and all(isinstance(m, dict) for m in r["messages"][:3]):
            if all(("role" in m and "content" in m) for m in r["messages"][:3]):
                return "sharegpt"
        if isinstance(r.get("conversations"), list) and r["conversations"] and all(isinstance(m, dict) for m in r["conversations"][:3]):
            m0 = r["conversations"][0]
            if ("from" in m0 and "value" in m0) or ("role" in m0 and "content" in m0):
                return "sharegpt"

    # Alpaca-ish: instruction/output or common key pairs
    pairs = [
        ("instruction", "output"),
        ("instruction", "response"),
        ("Instruction", "Response"),
        ("prompt", "completion"),
        ("question", "answer"),
    ]
    for r in sample:
        for a, b in pairs:
            if a in r and b in r and isinstance(r[a], str) and isinstance(r[b], str):
                return "alpaca"

    return "unknown"


def normalize_to_alpaca(records: List[DatasetRecord]) -> Tuple[Optional[List[DatasetRecord]], Optional[str]]:
    """
    Best-effort convert to Alpaca JSONL: {instruction, input?, output}
    """
    out: List[DatasetRecord] = []

    for i, r in enumerate(records):
        # already alpaca-ish
        if "instruction" in r and ("output" in r or "response" in r):
            inst = r.get("instruction")
            outp = r.get("output", r.get("response"))
            if isinstance(inst, str) and isinstance(outp, str):
                rec = {"instruction": inst, "output": outp}
                if isinstance(r.get("input"), str):
                    rec["input"] = r["input"]
                out.append(rec)
                continue

        # blog schema
        if isinstance(r.get("Instruction"), str) and isinstance(r.get("Response"), str):
            out.append({"instruction": r["Instruction"], "output": r["Response"]})
            continue

        # prompt/completion
        if isinstance(r.get("prompt"), str) and isinstance(r.get("completion"), str):
            out.append({"instruction": r["prompt"], "output": r["completion"]})
            continue

        # question/answer
        if isinstance(r.get("question"), str) and isinstance(r.get("answer"), str):
            out.append({"instruction": r["question"], "output": r["answer"]})
            continue

        # ShareGPT -> alpaca (collapse user turns -> instruction, last assistant -> output)
        msgs = None
        if isinstance(r.get("messages"), list):
            msgs = r["messages"]
        elif isinstance(r.get("conversations"), list):
            # normalize conversations into role/content first
            msgs = []
            for m in r["conversations"]:
                if not isinstance(m, dict):
                    continue
                if "role" in m and "content" in m:
                    msgs.append({"role": m["role"], "content": m["content"]})
                elif "from" in m and "value" in m:
                    fr = m["from"]
                    role = "user" if fr in ("human", "user") else ("assistant" if fr in ("gpt", "assistant") else fr)
                    msgs.append({"role": role, "content": m["value"]})

        if isinstance(msgs, list) and msgs:
            user_parts: List[str] = []
            assistant_parts: List[str] = []
            ok = True
            for m in msgs:
                if not isinstance(m, dict):
                    ok = False
                    break
                role = m.get("role")
                content = m.get("content")
                if not isinstance(role, str) or not isinstance(content, str):
                    ok = False
                    break
                if role == "user":
                    user_parts.append(content)
                elif role == "assistant":
                    assistant_parts.append(content)
            if ok and user_parts and assistant_parts:
                out.append({"instruction": "\n\n".join(user_parts), "output": assistant_parts[-1]})
                continue

        return None, f"Record {i} cannot be converted to Alpaca (missing recognizable fields)."

    return out, None


def normalize_to_sharegpt(records: List[DatasetRecord]) -> Tuple[Optional[List[DatasetRecord]], Optional[str]]:
    """
    Best-effort convert to ShareGPT messages JSONL:
      {"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
    """
    out: List[DatasetRecord] = []

    for i, r in enumerate(records):
        # already messages
        if isinstance(r.get("messages"), list) and r["messages"]:
            msgs = []
            ok = True
            for m in r["messages"]:
                if not isinstance(m, dict):
                    ok = False
                    break
                role = m.get("role")
                content = m.get("content")
                if not isinstance(role, str) or not isinstance(content, str):
                    ok = False
                    break
                msgs.append({"role": role, "content": content})
            if ok and msgs:
                out.append({"messages": msgs})
                continue

        # conversations -> messages
        if isinstance(r.get("conversations"), list) and r["conversations"]:
            msgs = []
            ok = True
            for m in r["conversations"]:
                if not isinstance(m, dict):
                    ok = False
                    break
                if "role" in m and "content" in m and isinstance(m["role"], str) and isinstance(m["content"], str):
                    msgs.append({"role": m["role"], "content": m["content"]})
                elif "from" in m and "value" in m and isinstance(m["from"], str) and isinstance(m["value"], str):
                    fr = m["from"]
                    role = "user" if fr in ("human", "user") else ("assistant" if fr in ("gpt", "assistant") else fr)
                    msgs.append({"role": role, "content": m["value"]})
                else:
                    ok = False
                    break
            if ok and msgs:
                out.append({"messages": msgs})
                continue

        # alpaca-ish -> sharegpt
        if isinstance(r.get("instruction"), str) and (isinstance(r.get("output"), str) or isinstance(r.get("response"), str)):
            outp = r.get("output", r.get("response"))
            out.append({"messages": [{"role": "user", "content": r["instruction"]}, {"role": "assistant", "content": outp}]})
            continue

        if isinstance(r.get("Instruction"), str) and isinstance(r.get("Response"), str):
            out.append({"messages": [{"role": "user", "content": r["Instruction"]}, {"role": "assistant", "content": r["Response"]}]})
            continue

        if isinstance(r.get("prompt"), str) and isinstance(r.get("completion"), str):
            out.append({"messages": [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["completion"]}]})
            continue

        return None, f"Record {i} cannot be converted to ShareGPT (missing recognizable fields)."

    return out, None


def save_dataset(records: List[DatasetRecord], out_path: Path, as_jsonl: bool, overwrite: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        die(f"Refusing to overwrite existing file: {out_path}")
    if as_jsonl:
        with out_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def dataset_guidance() -> Panel:
    msg = (
        "[bold]Dataset formatting help[/bold]\n\n"
        "[bold cyan]Alpaca[/bold cyan] (instruction/output)\n"
        "JSONL:\n"
        "  {\"instruction\":\"...\", \"output\":\"...\"}\n"
        "Optional:\n"
        "  {\"instruction\":\"...\", \"input\":\"...\", \"output\":\"...\"}\n\n"
        "[bold magenta]ShareGPT[/bold magenta] (messages)\n"
        "JSONL:\n"
        "  {\"messages\":[\n"
        "     {\"role\":\"user\",\"content\":\"...\"},\n"
        "     {\"role\":\"assistant\",\"content\":\"...\"}\n"
        "  ]}\n\n"
        "If your keys are different (e.g. Instruction/Response), Lazyllama can usually convert.\n"
        "If you have multi-turn chats, ShareGPT is the right choice."
    )
    return Panel(msg, border_style="red")


# ------------------------------ Profile + State ------------------------------

@dataclass
class Profile:
    workdir: str
    venv_dir: Optional[str]
    repo_dir: Optional[str]

    model: str
    template: str

    dataset_name: str
    dataset_file: str
    dataset_src: Optional[str]

    auto_format: bool
    prefer_format: str

    cutoff_len: int
    max_samples: int
    preproc_workers: int
    dataloader_workers: int

    lora_rank: int
    batch_size: int
    grad_accum: int
    lr: str
    epochs: str
    lr_sched: str
    warmup_ratio: str
    bf16: bool

    train_outdir: str
    merged_outdir: str

    export_size: int
    export_device: str
    export_legacy: bool

    overwrite: bool
    yes: bool
    smoketest: bool

    @staticmethod
    def from_defaults() -> "Profile":
        return Profile(**DEFAULTS)  # type: ignore[arg-type]


def profile_path(cwd: Path) -> Path:
    return cwd / PROFILE_NAME


def load_profile(path: Path) -> Profile:
    data = load_json(path)
    # tolerate missing keys by merging defaults
    merged = dict(DEFAULTS)
    merged.update(data)
    return Profile(**merged)  # type: ignore[arg-type]


def save_profile(path: Path, prof: Profile) -> None:
    save_json(path, asdict(prof))


def resolve_paths(prof: Profile) -> Tuple[Path, Path, Path]:
    workdir = Path(prof.workdir).expanduser().resolve()
    venv_dir = Path(prof.venv_dir).expanduser().resolve() if prof.venv_dir else (workdir / "venv")
    repo_dir = Path(prof.repo_dir).expanduser().resolve() if prof.repo_dir else (workdir / "LLaMA-Factory")
    return workdir, venv_dir, repo_dir


def state_root(workdir: Path) -> Path:
    return workdir / ".lazyllama"


def state_path(workdir: Path) -> Path:
    return state_root(workdir) / "state.json"


def runs_dir(workdir: Path) -> Path:
    return state_root(workdir) / "runs"


def load_state(workdir: Path) -> Dict[str, Any]:
    sp = state_path(workdir)
    if sp.exists():
        return load_json(sp)
    return {
        "current_run_id": None,
        "runs": {},  # run_id -> {steps:{step:{status,ts,meta}}}
    }


def save_state(workdir: Path, data: Dict[str, Any]) -> None:
    save_json(state_path(workdir), data)


def step_mark(workdir: Path, run_id: str, step: str, status: str, meta: Optional[Dict[str, Any]] = None) -> None:
    st = load_state(workdir)
    st["current_run_id"] = run_id
    st.setdefault("runs", {})
    st["runs"].setdefault(run_id, {"steps": {}, "created": dt.datetime.now().isoformat()})
    st["runs"][run_id]["steps"][step] = {
        "status": status,
        "ts": dt.datetime.now().isoformat(),
        "meta": meta or {},
    }
    save_state(workdir, st)


def step_status(workdir: Path, run_id: str, step: str) -> Optional[str]:
    st = load_state(workdir)
    r = st.get("runs", {}).get(run_id, {})
    s = r.get("steps", {}).get(step, {})
    return s.get("status")


# ------------------------------ Doctor ------------------------------

def doctor(prof: Profile) -> int:
    workdir, venv_dir, repo_dir = resolve_paths(prof)
    llama_identity(risk=None)

    t = Table(title="Doctor report", box=box.ROUNDED, border_style="blue")
    t.add_column("Check", style="bold")
    t.add_column("Result")
    t.add_column("Notes", overflow="fold")

    # basic commands
    t.add_row("python", "[green]OK[/green]" if sys.version_info >= (3, 10) else "[red]FAIL[/red]",
              f"{sys.version.split()[0]}")
    t.add_row("git", "[green]OK[/green]" if which("git") else "[red]FAIL[/red]",
              "required to clone LLaMA-Factory" if not which("git") else "present")

    # nvidia-smi
    if which("nvidia-smi"):
        t.add_row("nvidia-smi", "[green]OK[/green]", "GPU visible (good sign)")
    else:
        t.add_row("nvidia-smi", "[yellow]WARN[/yellow]", "GPU not detected (CPU-only is possible but slow)")

    # disk space
    try:
        usage = shutil.disk_usage(str(workdir))
        free = usage.free
        note = f"free {bytes_to_human(free)}"
        if free < 10 * 1024**3:
            t.add_row("disk space", "[yellow]WARN[/yellow]", note + " (might be tight)")
        else:
            t.add_row("disk space", "[green]OK[/green]", note)
    except Exception:
        t.add_row("disk space", "[yellow]WARN[/yellow]", "could not check")

    # venv presence
    t.add_row("venv exists", "[green]OK[/green]" if venv_dir.exists() else "[yellow]WARN[/yellow]",
              str(venv_dir))

    # repo presence
    t.add_row("repo exists", "[green]OK[/green]" if repo_dir.exists() else "[yellow]WARN[/yellow]",
              str(repo_dir))

    # llamafactory-cli presence
    cli = venv_exe(venv_dir, "llamafactory-cli")
    t.add_row("llamafactory-cli", "[green]OK[/green]" if cli.exists() else "[yellow]WARN[/yellow]",
              str(cli))

    # torch CUDA test (inside venv if possible)
    torch_note = "skipped"
    torch_res = "[yellow]WARN[/yellow]"
    if venv_python(venv_dir).exists():
        try:
            cp = subprocess.run(
                [str(venv_python(venv_dir)), "-c", "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available());"],
                check=False,
                text=True,
                capture_output=True,
            )
            out = (cp.stdout or "").strip().replace("\n", " | ")
            if "cuda True" in out:
                torch_res = "[green]OK[/green]"
            else:
                torch_res = "[yellow]WARN[/yellow]"
            torch_note = out or (cp.stderr or "no output")
        except Exception as e:
            torch_note = f"failed: {e}"
            torch_res = "[yellow]WARN[/yellow]"
    t.add_row("torch cuda", torch_res, torch_note)

    console.print(t)
    console.print(Panel.fit(
        "If doctor shows WARN/FAIL:\n"
        "- install GPU drivers/WSL2 CUDA integration\n"
        "- ensure venv + LLaMA-Factory deps are installed\n"
        "- ensure enough disk space\n",
        border_style="cyan",
    ))
    return 0


# ------------------------------ LLaMA-Factory steps ------------------------------

def ensure_venv(workdir: Path, venv_dir: Path, overwrite: bool) -> None:
    if not venv_dir.exists():
        console.print(Panel.fit(f"Creating venv at [bold]{venv_dir}[/bold]", border_style="cyan"))
        workdir.mkdir(parents=True, exist_ok=True)
        run_cmd([sys.executable, "-m", "venv", str(venv_dir)])
    py = venv_python(venv_dir)
    pip = venv_pip(venv_dir)
    run_cmd([str(py), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])
    # install rich into venv too (handy if you later use venv python)
    req = Path(__file__).parent / "requirements.txt"
    if req.exists():
        run_cmd([str(pip), "install", "-r", str(req.resolve())], check=False)


def ensure_repo(workdir: Path, repo_dir: Path) -> None:
    if not which("git"):
        die("git not found in PATH.")
    if (repo_dir / ".git").exists():
        return
    console.print(Panel.fit("Cloning LLaMA-Factory…", border_style="cyan"))
    workdir.mkdir(parents=True, exist_ok=True)
    run_cmd(["git", "clone", "https://github.com/hiyouga/LLaMA-Factory.git", str(repo_dir)])


def install_llamafactory(venv_dir: Path, repo_dir: Path) -> None:
    pip = venv_pip(venv_dir)
    cli = venv_exe(venv_dir, "llamafactory-cli")
    console.print(Panel.fit("Installing LLaMA-Factory (editable) + metrics deps…", border_style="cyan"))
    run_cmd([str(pip), "install", "-e", "."], cwd=repo_dir)
    run_cmd([str(pip), "install", "-r", "requirements/metrics.txt"], cwd=repo_dir)
    if cli.exists():
        run_cmd([str(cli), "version"], cwd=repo_dir, check=False)
        run_cmd([str(cli), "env"], cwd=repo_dir, check=False)


def dataset_prepare_and_maybe_convert(
    prof: Profile,
    repo_dir: Path,
    overwrite: bool,
    convert_if_needed: bool,
    requested_target: Optional[str],  # alpaca|sharegpt|None
) -> Tuple[str, str]:
    """
    Improved convert flow:
    1) identify container format (json/jsonl) + style (alpaca/sharegpt/unknown)
    2) if requested_target is set:
         - if already that style: keep
         - else: convert
       else:
         - if unknown and prof.auto_format: convert to prof.prefer_format
         - otherwise keep
    Returns (dataset_filename_in_repo, detected_or_final_style)
    """
    data_dir = repo_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    target_path = data_dir / prof.dataset_file

    # copy/create dataset
    if prof.dataset_src:
        src = Path(prof.dataset_src).expanduser().resolve()
        if not src.exists():
            die(f"Dataset source not found: {src}")
        shutil.copyfile(src, target_path)
    else:
        # create example if missing
        if (not target_path.exists()) or overwrite:
            ex = (
                '{"Instruction":"Example: What is 1+1?","Response":"2"}\n'
                '{"Instruction":"Example: Explain LoRA in one sentence.","Response":"LoRA trains small low-rank adapter matrices instead of full model weights."}\n'
            )
            target_path.write_text(ex, encoding="utf-8")

    container_fmt, records = load_dataset_any(target_path)
    style = detect_dataset_style(records)

    # decide conversion target
    target = None
    if requested_target:
        target = requested_target.lower()
    else:
        if style == "unknown" and convert_if_needed and prof.auto_format:
            target = prof.prefer_format.lower()

    # if no conversion requested/needed
    if not target:
        return prof.dataset_file, style

    if target not in ("alpaca", "sharegpt"):
        die("Conversion target must be alpaca or sharegpt.")

    # already desired style
    if style == target:
        return prof.dataset_file, style

    console.print(Panel.fit(
        f"Dataset convert: [bold]{style}[/bold] → [bold]{target}[/bold] "
        f"(container={container_fmt})",
        border_style="magenta",
    ))

    if target == "alpaca":
        converted, err = normalize_to_alpaca(records)
        if converted is None:
            console.print(dataset_guidance())
            die(f"Convert to Alpaca failed: {err}")
        out_name = target_path.stem + "_alpaca.jsonl"
        out_path = data_dir / out_name
        save_dataset(converted, out_path, as_jsonl=True, overwrite=overwrite)
        return out_name, "alpaca"

    converted, err = normalize_to_sharegpt(records)
    if converted is None:
        console.print(dataset_guidance())
        die(f"Convert to ShareGPT failed: {err}")
    out_name = target_path.stem + "_sharegpt.jsonl"
    out_path = data_dir / out_name
    save_dataset(converted, out_path, as_jsonl=True, overwrite=overwrite)
    return out_name, "sharegpt"


def register_dataset_info(repo_dir: Path, dataset_name: str, dataset_file: str, style: str) -> None:
    info_path = repo_dir / "data" / "dataset_info.json"
    if not info_path.exists():
        die(f"Missing dataset_info.json at: {info_path}")

    data = load_json(info_path)
    if style == "sharegpt":
        data[dataset_name] = {
            "file_name": dataset_file,
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
        }
    else:
        # alpaca normalized: instruction/output
        data[dataset_name] = {
            "file_name": dataset_file,
            "columns": {"instruction": "instruction", "response": "output"},
        }
    save_json(info_path, data)
    console.print(Panel.fit(f"Registered dataset [bold]{dataset_name}[/bold] as [bold]{style}[/bold]", border_style="green"))


def write_yaml(path: Path, lines: List[str], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def generate_train_yaml(prof: Profile, repo_dir: Path) -> Path:
    p = repo_dir / "qwen3_lora_sft.yaml"
    lines = [
        "### model",
        f"model_name_or_path: {prof.model}",
        "trust_remote_code: true",
        "",
        "### method",
        "stage: sft",
        "do_train: true",
        "finetuning_type: lora",
        f"lora_rank: {prof.lora_rank}",
        "lora_target: all",
        "",
        "### dataset",
        f"dataset: {prof.dataset_name}",
        f"template: {prof.template}",
        f"cutoff_len: {prof.cutoff_len}",
        f"max_samples: {prof.max_samples}",
        f"preprocessing_num_workers: {prof.preproc_workers}",
        f"dataloader_num_workers: {prof.dataloader_workers}",
        "",
        "### output",
        f"output_dir: {prof.train_outdir}",
        "logging_steps: 10",
        "save_steps: 500",
        "plot_loss: true",
        "overwrite_output_dir: true",
        "save_only_model: false",
        "report_to: none",
        "",
        "### train",
        f"per_device_train_batch_size: {prof.batch_size}",
        f"gradient_accumulation_steps: {prof.grad_accum}",
        f"learning_rate: {prof.lr}",
        f"num_train_epochs: {prof.epochs}",
        f"lr_scheduler_type: {prof.lr_sched}",
        f"warmup_ratio: {prof.warmup_ratio}",
        f"bf16: {'true' if prof.bf16 else 'false'}",
        "ddp_timeout: 180000000",
        "resume_from_checkpoint: null",
    ]
    write_yaml(p, lines, overwrite=prof.overwrite)
    return p


def generate_merge_yaml(prof: Profile, repo_dir: Path) -> Path:
    p = repo_dir / "merge_config.yaml"
    lines = [
        "### Note: DO NOT use quantized model or quantization_bit when merging lora adapters",
        "",
        "### model",
        f"model_name_or_path: {prof.model}",
        f"adapter_name_or_path: {prof.train_outdir}",
        f"template: {prof.template}",
        "trust_remote_code: true",
        "",
        "### export",
        f"export_dir: {prof.merged_outdir}",
        f"export_size: {prof.export_size}",
        f"export_device: {prof.export_device}",
        f"export_legacy_format: {'true' if prof.export_legacy else 'false'}",
    ]
    write_yaml(p, lines, overwrite=prof.overwrite)
    return p


def run_train(venv_dir: Path, repo_dir: Path, train_yaml: Path) -> None:
    cli = venv_exe(venv_dir, "llamafactory-cli")
    if not cli.exists():
        die(f"llamafactory-cli not found: {cli}")
    console.print(Panel.fit("Training (SFT LoRA)…", border_style="magenta"))
    run_cmd([str(cli), "train", train_yaml.name], cwd=repo_dir)


def run_merge(venv_dir: Path, repo_dir: Path, merge_yaml: Path) -> None:
    cli = venv_exe(venv_dir, "llamafactory-cli")
    if not cli.exists():
        die(f"llamafactory-cli not found: {cli}")
    console.print(Panel.fit("Merging adapters (export)…", border_style="magenta"))
    run_cmd([str(cli), "export", merge_yaml.name], cwd=repo_dir)


def smoke_test(venv_dir: Path, repo_dir: Path, merged_outdir: str) -> None:
    """
    Minimal “does it load?” test without entering interactive chat:
    We attempt a tiny one-shot by invoking chat and immediately sending EOF is tricky;
    so we do a lightweight existence check + optional python import check.
    """
    merged_dir = repo_dir / merged_outdir
    if not merged_dir.exists():
        die(f"Smoketest failed: merged dir not found: {merged_dir}")

    required = ["model.safetensors", "config.json", "tokenizer.json"]
    missing = [f for f in required if not (merged_dir / f).exists()]
    if missing:
        die(f"Smoketest failed: missing files in merged dir: {missing}")

    console.print(Panel.fit("[bold green]Smoketest OK[/bold green] (merged artifacts present)", border_style="green"))


# ------------------------------ Risk heuristic ------------------------------

def risk_badge(prof: Profile) -> str:
    """
    Very rough heuristics (not perfect):
    - exporting on CPU is safer for small VRAM
    - bf16 True can be risky on some GPUs
    - cutoff_len large increases memory
    """
    risk = 0
    if prof.cutoff_len >= 2048:
        risk += 1
    if prof.lora_rank >= 16:
        risk += 1
    if prof.batch_size >= 2:
        risk += 1
    if prof.export_device != "cpu":
        risk += 1
    if prof.bf16:
        risk += 0  # could be safe; not scoring it by default

    if risk <= 1:
        return "SAFE"
    if risk <= 3:
        return "WARN"
    return "RISK"


# ------------------------------ Commands: init/edit/run/state/rerun/clean ------------------------------

def wizard_profile(existing: Optional[Profile] = None) -> Profile:
    base = existing or Profile.from_defaults()
    console.print(Panel("Press Enter to accept defaults. This writes a reproducible profile.", border_style="blue"))

    base.workdir = Prompt.ask("workdir", default=base.workdir)

    wd = Path(base.workdir).expanduser().resolve()
    base.venv_dir = Prompt.ask("venv_dir (blank => <workdir>/venv)", default=base.venv_dir or "")
    base.venv_dir = base.venv_dir or None
    base.repo_dir = Prompt.ask("repo_dir (blank => <workdir>/LLaMA-Factory)", default=base.repo_dir or "")
    base.repo_dir = base.repo_dir or None

    base.model = Prompt.ask("model", default=base.model)
    base.template = Prompt.ask("template", default=base.template)

    base.dataset_name = Prompt.ask("dataset_name", default=base.dataset_name)
    base.dataset_file = Prompt.ask("dataset_file", default=base.dataset_file)
    base.dataset_src = Prompt.ask("dataset_src (optional)", default=base.dataset_src or "") or None

    base.auto_format = Confirm.ask("auto_format dataset if needed?", default=base.auto_format)
    base.prefer_format = Prompt.ask("prefer_format (alpaca/sharegpt)", default=base.prefer_format)

    base.cutoff_len = int(Prompt.ask("cutoff_len", default=str(base.cutoff_len)))
    base.max_samples = int(Prompt.ask("max_samples", default=str(base.max_samples)))
    base.lora_rank = int(Prompt.ask("lora_rank", default=str(base.lora_rank)))
    base.grad_accum = int(Prompt.ask("grad_accum", default=str(base.grad_accum)))
    base.bf16 = Confirm.ask("bf16", default=base.bf16)

    base.train_outdir = Prompt.ask("train_outdir", default=base.train_outdir)
    base.merged_outdir = Prompt.ask("merged_outdir", default=base.merged_outdir)

    base.export_device = Prompt.ask("export_device (cpu/auto)", default=base.export_device)
    base.export_size = int(Prompt.ask("export_size", default=str(base.export_size)))
    base.export_legacy = Confirm.ask("export_legacy", default=base.export_legacy)

    base.overwrite = Confirm.ask("overwrite generated configs / converted datasets?", default=base.overwrite)
    base.smoketest = Confirm.ask("smoketest after merge?", default=base.smoketest)
    base.yes = False  # don't store "yes" as persistent behavior
    return base


def cmd_init(args: argparse.Namespace) -> int:
    llama_identity()
    cwd = Path.cwd()
    pp = profile_path(cwd)
    if pp.exists() and not args.force:
        die(f"{PROFILE_NAME} already exists here. Use --force to overwrite.")
    prof = wizard_profile()
    save_profile(pp, prof)
    console.print(Panel.fit(f"Wrote profile: [bold]{pp}[/bold]", border_style="green"))
    return 0


def cmd_edit(args: argparse.Namespace) -> int:
    cwd = Path.cwd()
    pp = profile_path(cwd)
    if not pp.exists():
        die(f"No {PROFILE_NAME} found in {cwd}. Run `python3 lazyllama.py init` first.")
    prof = load_profile(pp)

    if args.editor:
        ed = os.environ.get("EDITOR") or os.environ.get("VISUAL")
        if not ed:
            die("No $EDITOR/$VISUAL set. Use --wizard instead or set EDITOR.")
        run_cmd([ed, str(pp)], check=False)
        console.print(Panel.fit("Edited profile (external editor).", border_style="green"))
        return 0

    # default: wizard edit
    llama_identity()
    prof2 = wizard_profile(existing=prof)
    save_profile(pp, prof2)
    console.print(Panel.fit(f"Updated profile: [bold]{pp}[/bold]", border_style="green"))
    return 0


def cmd_state(args: argparse.Namespace) -> int:
    cwd = Path.cwd()
    pp = profile_path(cwd)
    if not pp.exists():
        die(f"No {PROFILE_NAME} found in {cwd}.")
    prof = load_profile(pp)
    workdir, _, _ = resolve_paths(prof)

    st = load_state(workdir)
    run_id = st.get("current_run_id")
    llama_identity(run_id=run_id, risk=risk_badge(prof))

    if not run_id or run_id not in st.get("runs", {}):
        console.print(Panel.fit("No runs recorded yet.", border_style="yellow"))
        return 0

    steps = st["runs"][run_id]["steps"]
    t = Table(title=f"State for run {run_id}", box=box.ROUNDED, border_style="blue")
    t.add_column("Step", style="bold")
    t.add_column("Status")
    t.add_column("Time", overflow="fold")

    for s in STEPS_ORDER:
        rec = steps.get(s)
        if not rec:
            t.add_row(s, "[dim]-[/dim]", "[dim]-[/dim]")
        else:
            status = rec.get("status", "-")
            color = {"done": "green", "skipped": "yellow", "failed": "red", "running": "cyan"}.get(status, "white")
            t.add_row(s, f"[{color}]{status}[/{color}]", rec.get("ts", "-"))
    console.print(t)
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    cwd = Path.cwd()
    pp = profile_path(cwd)
    if not pp.exists():
        die(f"No {PROFILE_NAME} found in {cwd}.")
    prof = load_profile(pp)
    workdir, _, repo_dir = resolve_paths(prof)

    llama_identity(risk=risk_badge(prof))
    console.print(Panel(
        "Clean options:\n"
        "- configs: remove generated YAMLs in repo\n"
        "- runs: remove Lazyllama state + run logs\n"
        "- outputs: remove train_outdir / merged_outdir directories\n",
        border_style="blue",
    ))

    do_configs = args.configs or Confirm.ask("Remove generated configs (qwen3_lora_sft.yaml, merge_config.yaml)?", default=False)
    do_runs = args.runs or Confirm.ask("Remove Lazyllama state/run logs?", default=False)
    do_outputs = args.outputs or Confirm.ask("Remove training/merged outputs (DANGEROUS)?", default=False)

    if not Confirm.ask("Proceed with clean?", default=False):
        console.print(Panel.fit("Aborted.", border_style="yellow"))
        return 0

    if do_configs:
        for fn in ["qwen3_lora_sft.yaml", "merge_config.yaml"]:
            p = repo_dir / fn
            if p.exists():
                p.unlink()
        console.print(Panel.fit("Removed generated configs.", border_style="green"))

    if do_outputs:
        # remove directories relative to repo
        for rel in [prof.train_outdir, prof.merged_outdir]:
            p = repo_dir / rel
            if p.exists():
                shutil.rmtree(p)
        console.print(Panel.fit("Removed outputs.", border_style="green"))

    if do_runs:
        sr = state_root(workdir)
        if sr.exists():
            shutil.rmtree(sr)
        console.print(Panel.fit("Removed Lazyllama run state/logs.", border_style="green"))

    return 0


def cmd_rerun(args: argparse.Namespace) -> int:
    """
    Force rerun from a specific step onward, using the latest run_id or a provided one.
    """
    cwd = Path.cwd()
    pp = profile_path(cwd)
    if not pp.exists():
        die(f"No {PROFILE_NAME} found in {cwd}.")
    prof = load_profile(pp)

    workdir, _, _ = resolve_paths(prof)
    st = load_state(workdir)
    run_id = args.run_id or st.get("current_run_id") or now_run_id()

    if args.step not in STEPS_ORDER:
        die(f"Invalid step '{args.step}'. Valid: {', '.join(STEPS_ORDER)}")

    # mark steps from step onward as not-done by deleting them
    st.setdefault("runs", {})
    st["runs"].setdefault(run_id, {"steps": {}, "created": dt.datetime.now().isoformat()})
    steps = st["runs"][run_id]["steps"]
    start_idx = STEPS_ORDER.index(args.step)
    for s in STEPS_ORDER[start_idx:]:
        if s in steps:
            del steps[s]
    st["current_run_id"] = run_id
    save_state(workdir, st)

    console.print(Panel.fit(f"Rerun prepared: run={run_id}, from step={args.step}", border_style="green"))
    # Then call run with that run_id
    return cmd_run(args, forced_run_id=run_id)


def cmd_run(args: argparse.Namespace, forced_run_id: Optional[str] = None) -> int:
    cwd = Path.cwd()
    pp = profile_path(cwd)
    if not pp.exists():
        die(f"No {PROFILE_NAME} found in {cwd}. Run `python3 lazyllama.py init` first.")
    prof = load_profile(pp)

    # apply runtime overrides
    if args.yes:
        prof.yes = True
    if args.overwrite:
        prof.overwrite = True
    if args.dataset_src:
        prof.dataset_src = args.dataset_src
    if args.convert_to:
        # conversion request for this run only
        pass

    workdir, venv_dir, repo_dir = resolve_paths(prof)
    run_id = forced_run_id or now_run_id()
    risk = risk_badge(prof)
    llama_identity(run_id=run_id, risk=risk)

    # show quick plan
    t = Table(title="Run plan", box=box.ROUNDED, border_style="blue")
    t.add_column("Item", style="bold")
    t.add_column("Value", overflow="fold")
    t.add_row("profile", str(pp))
    t.add_row("workdir", str(workdir))
    t.add_row("venv", str(venv_dir))
    t.add_row("repo", str(repo_dir))
    t.add_row("model", prof.model)
    t.add_row("dataset", f"{prof.dataset_name} -> {prof.dataset_file} (src={prof.dataset_src or 'none'})")
    t.add_row("risk", risk)
    console.print(t)

    if not prof.yes and not Confirm.ask("Proceed?", default=True):
        console.print(Panel.fit("Aborted.", border_style="yellow"))
        return 0

    # ensure state directories
    runs_dir(workdir).mkdir(parents=True, exist_ok=True)
    run_folder = runs_dir(workdir) / run_id
    run_folder.mkdir(parents=True, exist_ok=True)
    # snapshot profile
    save_profile(run_folder / PROFILE_NAME, prof)

    def should_run(step: str) -> bool:
        if args.force:
            return True
        s = step_status(workdir, run_id, step)
        return s != "done"

    # ---------------- Step: venv
    if should_run(STEP_VENV):
        step_mark(workdir, run_id, STEP_VENV, "running")
        try:
            with Status("[bold blue]Step: venv[/bold blue]", console=console):
                ensure_venv(workdir, venv_dir, overwrite=prof.overwrite)
            step_mark(workdir, run_id, STEP_VENV, "done")
        except Exception as e:
            step_mark(workdir, run_id, STEP_VENV, "failed", {"error": str(e)})
            raise

    # ---------------- Step: clone
    if should_run(STEP_CLONE):
        step_mark(workdir, run_id, STEP_CLONE, "running")
        try:
            with Status("[bold blue]Step: clone[/bold blue]", console=console):
                ensure_repo(workdir, repo_dir)
            step_mark(workdir, run_id, STEP_CLONE, "done")
        except Exception as e:
            step_mark(workdir, run_id, STEP_CLONE, "failed", {"error": str(e)})
            raise

    # ---------------- Step: install
    if should_run(STEP_INSTALL):
        step_mark(workdir, run_id, STEP_INSTALL, "running")
        try:
            with Status("[bold blue]Step: install[/bold blue]", console=console):
                install_llamafactory(venv_dir, repo_dir)
            step_mark(workdir, run_id, STEP_INSTALL, "done")
        except Exception as e:
            step_mark(workdir, run_id, STEP_INSTALL, "failed", {"error": str(e)})
            raise

    # ---------------- Step: dataset (prepare + convert if needed/requested)
    if should_run(STEP_DATASET):
        step_mark(workdir, run_id, STEP_DATASET, "running")
        try:
            with Status("[bold blue]Step: dataset[/bold blue]", console=console):
                dataset_file, final_style = dataset_prepare_and_maybe_convert(
                    prof=prof,
                    repo_dir=repo_dir,
                    overwrite=prof.overwrite,
                    convert_if_needed=True,
                    requested_target=args.convert_to,
                )
                prof.dataset_file = dataset_file  # update for this run
                step_mark(workdir, run_id, STEP_DATASET, "done", {"file": dataset_file, "style": final_style})
        except Exception as e:
            step_mark(workdir, run_id, STEP_DATASET, "failed", {"error": str(e)})
            console.print(dataset_guidance())
            raise

    # ---------------- Step: register
    if should_run(STEP_REGISTER):
        step_mark(workdir, run_id, STEP_REGISTER, "running")
        try:
            with Status("[bold blue]Step: register[/bold blue]", console=console):
                dataset_path = repo_dir / "data" / prof.dataset_file
                _, recs = load_dataset_any(dataset_path)
                style = detect_dataset_style(recs)
                if style == "unknown":
                    console.print(dataset_guidance())
                    die("Dataset style unknown even after conversion attempt.")
                register_dataset_info(repo_dir, prof.dataset_name, prof.dataset_file, style)
            step_mark(workdir, run_id, STEP_REGISTER, "done", {"style": style})
        except Exception as e:
            step_mark(workdir, run_id, STEP_REGISTER, "failed", {"error": str(e)})
            raise

    # ---------------- Step: configs
    if should_run(STEP_CONFIGS):
        step_mark(workdir, run_id, STEP_CONFIGS, "running")
        try:
            with Status("[bold blue]Step: configs[/bold blue]", console=console):
                train_yaml = generate_train_yaml(prof, repo_dir)
                merge_yaml = generate_merge_yaml(prof, repo_dir)
                # copy into run folder for reproducibility
                shutil.copyfile(train_yaml, run_folder / train_yaml.name)
                shutil.copyfile(merge_yaml, run_folder / merge_yaml.name)
            step_mark(workdir, run_id, STEP_CONFIGS, "done", {"train_yaml": "qwen3_lora_sft.yaml", "merge_yaml": "merge_config.yaml"})
        except Exception as e:
            step_mark(workdir, run_id, STEP_CONFIGS, "failed", {"error": str(e)})
            raise

    # ---------------- Step: train
    if should_run(STEP_TRAIN):
        step_mark(workdir, run_id, STEP_TRAIN, "running")
        try:
            with Status("[bold blue]Step: train[/bold blue]", console=console):
                run_train(venv_dir, repo_dir, repo_dir / "qwen3_lora_sft.yaml")
            step_mark(workdir, run_id, STEP_TRAIN, "done")
        except Exception as e:
            step_mark(workdir, run_id, STEP_TRAIN, "failed", {"error": str(e)})
            raise

    # ---------------- Step: merge
    if should_run(STEP_MERGE):
        step_mark(workdir, run_id, STEP_MERGE, "running")
        try:
            with Status("[bold blue]Step: merge[/bold blue]", console=console):
                run_merge(venv_dir, repo_dir, repo_dir / "merge_config.yaml")
            step_mark(workdir, run_id, STEP_MERGE, "done")
        except Exception as e:
            step_mark(workdir, run_id, STEP_MERGE, "failed", {"error": str(e)})
            raise

    # ---------------- Step: smoketest
    if prof.smoketest and should_run(STEP_SMOKETEST):
        step_mark(workdir, run_id, STEP_SMOKETEST, "running")
        try:
            with Status("[bold blue]Step: smoketest[/bold blue]", console=console):
                smoke_test(venv_dir, repo_dir, prof.merged_outdir)
            step_mark(workdir, run_id, STEP_SMOKETEST, "done")
        except Exception as e:
            step_mark(workdir, run_id, STEP_SMOKETEST, "failed", {"error": str(e)})
            raise
    elif not prof.smoketest:
        step_mark(workdir, run_id, STEP_SMOKETEST, "skipped")

    console.print(Panel.fit(
        f"[bold green]Run complete[/bold green]\n"
        f"run_id: [bold]{run_id}[/bold]\n"
        f"merged model: [bold]{(repo_dir / prof.merged_outdir)}[/bold]\n"
        f"repro artifacts: [bold]{run_folder}[/bold]",
        border_style="green",
    ))
    return 0


# ------------------------------ Dataset commands ------------------------------

def cmd_dataset_inspect(args: argparse.Namespace) -> int:
    path = Path(args.path).expanduser().resolve()
    fmt, records = load_dataset_any(path)
    style = detect_dataset_style(records)

    t = Table(title="Dataset inspection", box=box.ROUNDED, border_style="blue")
    t.add_column("Field", style="bold")
    t.add_column("Value", overflow="fold")
    t.add_row("path", str(path))
    t.add_row("container", fmt)
    t.add_row("records", str(len(records)))
    t.add_row("style", style)

    keys = sorted(set().union(*(r.keys() for r in records[: min(100, len(records))])))
    t.add_row("sample keys", ", ".join(keys[:40]) + (" …" if len(keys) > 40 else ""))
    console.print(t)
    return 0


def cmd_dataset_preview(args: argparse.Namespace) -> int:
    path = Path(args.path).expanduser().resolve()
    _, records = load_dataset_any(path)
    style = detect_dataset_style(records)

    n = min(args.n, len(records))
    if n <= 0:
        die("No records to preview.")

    picks = random.sample(records, k=n) if len(records) >= n else records
    t = Table(title=f"Dataset preview (style={style})", box=box.ROUNDED, border_style="magenta")
    t.add_column("#", style="bold")
    t.add_column("Snippet", overflow="fold")

    for i, r in enumerate(picks, start=1):
        snippet = ""
        if style == "alpaca":
            # try common fields
            inst = r.get("instruction") or r.get("Instruction") or r.get("prompt") or r.get("question")
            outp = r.get("output") or r.get("response") or r.get("Response") or r.get("completion") or r.get("answer")
            if isinstance(inst, str) and isinstance(outp, str):
                snippet = f"[bold]instruction[/bold]: {inst[:220]}\n[bold]output[/bold]: {outp[:220]}"
            else:
                snippet = json.dumps(r, ensure_ascii=False)[:450]
        elif style == "sharegpt":
            msgs = r.get("messages") or r.get("conversations")
            if isinstance(msgs, list) and msgs:
                snippet = json.dumps(msgs[: min(4, len(msgs))], ensure_ascii=False)[:450]
            else:
                snippet = json.dumps(r, ensure_ascii=False)[:450]
        else:
            snippet = json.dumps(r, ensure_ascii=False)[:450]

        t.add_row(str(i), snippet)
    console.print(t)
    return 0


def cmd_dataset_validate(args: argparse.Namespace) -> int:
    path = Path(args.path).expanduser().resolve()
    _, records = load_dataset_any(path)
    style = detect_dataset_style(records)

    issues: List[str] = []
    if not records:
        issues.append("dataset is empty")

    if style == "alpaca":
        for i, r in enumerate(records[: min(200, len(records))]):
            ok = False
            # allow multiple schemas
            pairs = [
                ("instruction", "output"),
                ("instruction", "response"),
                ("Instruction", "Response"),
                ("prompt", "completion"),
                ("question", "answer"),
            ]
            for a, b in pairs:
                if isinstance(r.get(a), str) and isinstance(r.get(b), str):
                    ok = True
                    break
            if not ok:
                issues.append(f"record {i} missing instruction/output-like fields")
                break

    elif style == "sharegpt":
        for i, r in enumerate(records[: min(200, len(records))]):
            msgs = r.get("messages") or r.get("conversations")
            if not isinstance(msgs, list) or not msgs:
                issues.append(f"record {i} missing messages/conversations list")
                break

    else:
        issues.append("style unknown (not Alpaca/ShareGPT)")

    if issues:
        console.print(Panel.fit("[bold red]Validation failed[/bold red]\n- " + "\n- ".join(issues), border_style="red"))
        console.print(dataset_guidance())
        return 2

    console.print(Panel.fit(f"[bold green]Validation OK[/bold green] (style={style})", border_style="green"))
    return 0


def cmd_dataset_convert(args: argparse.Namespace) -> int:
    inp = Path(args.path).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    target = args.to.lower()

    fmt, records = load_dataset_any(inp)
    style = detect_dataset_style(records)

    llama_identity(risk=None)
    console.print(Panel.fit(f"Convert request: style=[bold]{style}[/bold], container=[bold]{fmt}[/bold] → [bold]{target}[/bold]", border_style="magenta"))

    if style == target and not args.force:
        console.print(Panel.fit("Already in requested style. Use --force to rewrite/normalize.", border_style="yellow"))
        return 0

    if target == "alpaca":
        converted, err = normalize_to_alpaca(records)
        if converted is None:
            console.print(dataset_guidance())
            die(f"Convert failed: {err}")
        # write JSONL by default
        save_dataset(converted, out, as_jsonl=True, overwrite=args.overwrite)
        console.print(Panel.fit(f"[bold green]Converted to Alpaca[/bold green] → {out}", border_style="green"))
        return 0

    if target == "sharegpt":
        converted, err = normalize_to_sharegpt(records)
        if converted is None:
            console.print(dataset_guidance())
            die(f"Convert failed: {err}")
        save_dataset(converted, out, as_jsonl=True, overwrite=args.overwrite)
        console.print(Panel.fit(f"[bold green]Converted to ShareGPT[/bold green] → {out}", border_style="green"))
        return 0

    die("Target must be alpaca or sharegpt.")


# ------------------------------ CLI ------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="lazyllama",
        description="Lazyllama — reproducible LoRA SFT tool for LLaMA-Factory (with dataset tools + doctor + state).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # init / edit
    s_init = sub.add_parser("init", help=f"Create {PROFILE_NAME} (reproducible profile).")
    s_init.add_argument("--force", action="store_true", help="Overwrite existing profile.")

    s_edit = sub.add_parser("edit", help=f"Edit {PROFILE_NAME}.")
    s_edit.add_argument("--editor", action="store_true", help="Open with $EDITOR instead of wizard.")

    # doctor
    sub.add_parser("doctor", help="Preflight checks (git, gpu, venv, disk, torch cuda).")

    # run
    s_run = sub.add_parser("run", help="Run the pipeline using lazyllama.json (stateful).")
    s_run.add_argument("-y", "--yes", action="store_true", help="Assume yes for prompts.")
    s_run.add_argument("--overwrite", action="store_true", help="Overwrite generated configs / converted datasets.")
    s_run.add_argument("--force", action="store_true", help="Force run all steps even if state says done.")
    s_run.add_argument("--dataset-src", default=None, help="Override dataset source for this run.")
    s_run.add_argument("--convert-to", choices=["alpaca", "sharegpt"], default=None,
                       help="Force dataset conversion target for this run (even if already valid).")

    # state
    sub.add_parser("state", help="Show state of last run.")

    # rerun
    s_rerun = sub.add_parser("rerun", help="Rerun from a given step onward (uses state machine).")
    s_rerun.add_argument("step", choices=STEPS_ORDER, help="Step to rerun from.")
    s_rerun.add_argument("--run-id", default=None, help="Run id to modify; default = latest.")
    s_rerun.add_argument("-y", "--yes", action="store_true", help="Assume yes for prompts.")
    s_rerun.add_argument("--overwrite", action="store_true", help="Overwrite generated configs / converted datasets.")
    s_rerun.add_argument("--force", action="store_true", help="Force run all steps even if state says done.")
    s_rerun.add_argument("--dataset-src", default=None, help="Override dataset source for this run.")
    s_rerun.add_argument("--convert-to", choices=["alpaca", "sharegpt"], default=None,
                         help="Force dataset conversion target for this run.")

    # clean
    s_clean = sub.add_parser("clean", help="Clean configs / run state / outputs.")
    s_clean.add_argument("--configs", action="store_true", help="Remove generated YAML configs.")
    s_clean.add_argument("--runs", action="store_true", help="Remove Lazyllama state + run logs.")
    s_clean.add_argument("--outputs", action="store_true", help="Remove training + merged outputs (dangerous).")

    # dataset suite
    s_ds = sub.add_parser("dataset", help="Dataset tools: inspect/preview/validate/convert.")
    ds_sub = s_ds.add_subparsers(dest="ds_cmd", required=True)

    dsi = ds_sub.add_parser("inspect", help="Detect dataset container (json/jsonl) + style (alpaca/sharegpt).")
    dsi.add_argument("path")

    dsp = ds_sub.add_parser("preview", help="Preview random samples.")
    dsp.add_argument("path")
    dsp.add_argument("-n", type=int, default=3)

    dsv = ds_sub.add_parser("validate", help="Validate a dataset matches Alpaca or ShareGPT shape.")
    dsv.add_argument("path")

    dsc = ds_sub.add_parser("convert", help="Convert dataset to Alpaca or ShareGPT (JSONL output).")
    dsc.add_argument("path")
    dsc.add_argument("--to", choices=["alpaca", "sharegpt"], required=True)
    dsc.add_argument("--out", required=True)
    dsc.add_argument("--overwrite", action="store_true")
    dsc.add_argument("--force", action="store_true", help="Rewrite even if already target style.")
    return p.parse_args()


def main() -> int:
    ensure_python_version()
    args = parse_args()

    try:
        if args.cmd == "init":
            return cmd_init(args)
        if args.cmd == "edit":
            return cmd_edit(args)
        if args.cmd == "doctor":
            pp = profile_path(Path.cwd())
            if not pp.exists():
                die(f"No {PROFILE_NAME} found. Run `python3 lazyllama.py init` first.")
            prof = load_profile(pp)
            return doctor(prof)
        if args.cmd == "run":
            return cmd_run(args)
        if args.cmd == "state":
            return cmd_state(args)
        if args.cmd == "rerun":
            return cmd_rerun(args)
        if args.cmd == "clean":
            return cmd_clean(args)
        if args.cmd == "dataset":
            if args.ds_cmd == "inspect":
                return cmd_dataset_inspect(args)
            if args.ds_cmd == "preview":
                return cmd_dataset_preview(args)
            if args.ds_cmd == "validate":
                return cmd_dataset_validate(args)
            if args.ds_cmd == "convert":
                return cmd_dataset_convert(args)
            die("Unknown dataset command.")
        die("Unknown command.")

    except LazyllamaError as e:
        console.print(Panel.fit(f"[bold red]Error:[/bold red] {e}", border_style="red"))
        return 2
    except subprocess.CalledProcessError as e:
        console.print(Panel.fit(f"[bold red]Command failed[/bold red] (exit {e.returncode}).", border_style="red"))
        return e.returncode


if __name__ == "__main__":
    raise SystemExit(main())