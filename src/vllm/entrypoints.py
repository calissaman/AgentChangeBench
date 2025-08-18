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
            f"python -m vllm.entrypoints.openai.api_server --model {os.getenv('QWEN_MODEL')} --served-model-name qwen2-7b-instruct --dtype {os.getenv('DTYPE_QWEN', 'bfloat16')} --host 0.0.0.0 --port {os.getenv('QWEN_PORT')}",
            f"caddy run --config src/vllm/Caddyfile --adapter caddyfile",
        ]
    )


def run_mistral():
    mprocs_bin = os.getenv("MPROCS_BIN", "mprocs")
    return subprocess.call(
        [
            mprocs_bin,
            "--names",
            "vllm-mistral,caddy",
            f"python -m vllm.entrypoints.openai.api_server --model {os.getenv('MISTRAL_MODEL')} --served-model-name mistral-7b-instruct --dtype {os.getenv('DTYPE_MISTRAL', 'bfloat16')} --host 0.0.0.0 --port {os.getenv('MISTRAL_PORT')}",
            f"caddy run --config src/vllm/Caddyfile --adapter caddyfile",
        ]
    )


def run_gptoss():
    mprocs_bin = os.getenv("MPROCS_BIN", "mprocs")
    return subprocess.call(
        [
            mprocs_bin,
            "--names",
            "vllm-gptoss,caddy",
            f"python -m vllm.entrypoints.openai.api_server --model {os.getenv('GPTOSS_MODEL')} --served-model-name gpt-oss-20b --host 0.0.0.0 --port {os.getenv('GPTOSS_PORT')}",
            f"caddy run --config src/vllm/Caddyfile --adapter caddyfile",
        ]
    )


def run_all():
    mprocs_bin = os.getenv("MPROCS_BIN", "mprocs")
    return subprocess.call(
        [
            mprocs_bin,
            "--names",
            "vllm-qwen,vllm-mistral,vllm-gptoss,caddy",
            f"python -m vllm.entrypoints.openai.api_server --model {os.getenv('QWEN_MODEL')} --served-model-name qwen2-7b-instruct --dtype {os.getenv('DTYPE_QWEN', 'bfloat16')} --host 0.0.0.0 --port {os.getenv('QWEN_PORT')}",
            f"python -m vllm.entrypoints.openai.api_server --model {os.getenv('MISTRAL_MODEL')} --served-model-name mistral-7b-instruct --dtype {os.getenv('DTYPE_MISTRAL', 'bfloat16')} --host 0.0.0.0 --port {os.getenv('MISTRAL_PORT')}",
            f"python -m vllm.entrypoints.openai.api_server --model {os.getenv('GPTOSS_MODEL')} --served-model-name gpt-oss-20b --host 0.0.0.0 --port {os.getenv('GPTOSS_PORT')}",
            f"caddy run --config src/vllm/Caddyfile --adapter caddyfile",
        ]
    )
