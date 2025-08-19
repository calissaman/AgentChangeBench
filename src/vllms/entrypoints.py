import os, subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def get_served_model_name(model_path):
    if not model_path:
        return "unknown-model"
    
    model_name = model_path.split('/')[-1].lower()
    
    model_name = model_name.replace('-instruct', '-instruct')
    model_name = model_name.replace('-fp8', '')
    model_name = model_name.replace('-2507', '')
    
    return model_name


def setup_vllms():
    script_path = Path(__file__).parent.parent.parent / "scripts" / "setup_vllms.sh"
    
    if not script_path.exists():
        print(f"Error: Setup script not found at {script_path}")
        return 1
    
    print(f"Running setup script: {script_path}")
    return subprocess.call(["bash", str(script_path)])


def run_qwen():
    mprocs_bin = os.getenv("MPROCS_BIN", "mprocs")
    qwen_model = os.getenv('QWEN_MODEL')
    served_name = get_served_model_name(qwen_model)
    
    return subprocess.call(
        [
            mprocs_bin,
            "--names",
            "vllm-qwen,caddy",
            f"python -m vllm.entrypoints.openai.api_server --model {qwen_model} --served-model-name {served_name} --dtype {os.getenv('DTYPE_QWEN', 'bfloat16')} --host 0.0.0.0 --port {os.getenv('QWEN_PORT')} --enable-auto-tool-choice --tool-call-parser hermes",
            f"caddy run --config src/vllms/Caddyfile --adapter caddyfile",
        ]
    )


def run_mistral():
    mprocs_bin = os.getenv("MPROCS_BIN", "mprocs")
    mistral_model = os.getenv('MISTRAL_MODEL')
    served_name = get_served_model_name(mistral_model)
    
    return subprocess.call(
        [
            mprocs_bin,
            "--names",
            "vllm-mistral,caddy",
            f"python -m vllm.entrypoints.openai.api_server --model {mistral_model} --served-model-name {served_name} --dtype {os.getenv('DTYPE_MISTRAL', 'bfloat16')} --host 0.0.0.0 --port {os.getenv('MISTRAL_PORT')} --enable-auto-tool-choice --tool-call-parser mistral",
            f"caddy run --config src/vllms/Caddyfile --adapter caddyfile",
        ]
    )


def run_all():
    mprocs_bin = os.getenv("MPROCS_BIN", "mprocs")
    qwen_model = os.getenv('QWEN_MODEL')
    mistral_model = os.getenv('MISTRAL_MODEL')
    qwen_served_name = get_served_model_name(qwen_model)
    mistral_served_name = get_served_model_name(mistral_model)
    
    return subprocess.call(
        [
            mprocs_bin,
            "--names",
            "vllm-qwen,vllm-mistral,vllm-gptoss,caddy",
            f"python -m vllm.entrypoints.openai.api_server --model {qwen_model} --served-model-name {qwen_served_name} --dtype {os.getenv('DTYPE_QWEN', 'bfloat16')} --host 0.0.0.0 --port {os.getenv('QWEN_PORT')} --enable-auto-tool-choice --tool-call-parser hermes",
            f"python -m vllm.entrypoints.openai.api_server --model {mistral_model} --served-model-name {mistral_served_name} --dtype {os.getenv('DTYPE_MISTRAL', 'bfloat16')} --host 0.0.0.0 --port {os.getenv('MISTRAL_PORT')} --enable-auto-tool-choice --tool-call-parser mistral",
            f"caddy run --config src/vllms/Caddyfile --adapter caddyfile",
        ]
    )
