import os, sys, subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def run_qwen():
    mprocs_bin = os.getenv("MPROCS_BIN", "mprocs")
    return subprocess.call(
        [
            mprocs_bin,
            "--names",
            "vllm-qwen,caddy",
            f"python -m vllm.entrypoints.openai.api_server --model {os.getenv('QWEN_MODEL')} --served-model-name qwen2.5-7b-instruct --dtype {os.getenv('DTYPE_QWEN', 'bfloat16')} --host 0.0.0.0 --port {os.getenv('QWEN_PORT')} --enable-auto-tool-choice --tool-call-parser hermes",
            f"caddy run --config src/vllms/Caddyfile --adapter caddyfile",
        ]
    )


def run_mistral():
    mprocs_bin = os.getenv("MPROCS_BIN", "mprocs")
    return subprocess.call(
        [
            mprocs_bin,
            "--names",
            "vllm-mistral,caddy",
            f"python -m vllm.entrypoints.openai.api_server --model {os.getenv('MISTRAL_MODEL')} --served-model-name mistral-7b-instruct --dtype {os.getenv('DTYPE_MISTRAL', 'bfloat16')} --host 0.0.0.0 --port {os.getenv('MISTRAL_PORT')} --enable-auto-tool-choice --tool-call-parser mistral",
            f"caddy run --config src/vllms/Caddyfile --adapter caddyfile",
        ]
    )


def run_all():
    mprocs_bin = os.getenv("MPROCS_BIN", "mprocs")
    return subprocess.call(
        [
            mprocs_bin,
            "--names",
            "vllm-qwen,vllm-mistral,vllm-gptoss,caddy",
            f"python -m vllm.entrypoints.openai.api_server --model {os.getenv('QWEN_MODEL')} --served-model-name qwen2.5-7b-instruct --dtype {os.getenv('DTYPE_QWEN', 'bfloat16')} --host 0.0.0.0 --port {os.getenv('QWEN_PORT')} --enable-auto-tool-choice --tool-call-parser hermes",
            f"python -m vllm.entrypoints.openai.api_server --model {os.getenv('MISTRAL_MODEL')} --served-model-name mistral-7b-instruct --dtype {os.getenv('DTYPE_MISTRAL', 'bfloat16')} --host 0.0.0.0 --port {os.getenv('MISTRAL_PORT')} --enable-auto-tool-choice --tool-call-parser mistral",
            f"caddy run --config src/vllms/Caddyfile --adapter caddyfile",
        ]
    )
