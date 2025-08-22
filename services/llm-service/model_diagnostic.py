#!/usr/bin/env python3
"""
Model Path Diagnostic Script
Helps diagnose model loading issues for LLM service
"""

import os
import sys
from pathlib import Path
import json
from typing import List, Dict, Any


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_section(title, icon="📋"):
    print(f"\n{Colors.CYAN}{Colors.BOLD}{icon} {title}{Colors.END}")
    print("=" * 60)


def print_success(message, icon="✅"):
    print(f"   {Colors.GREEN}{icon} {message}{Colors.END}")


def print_error(message, icon="❌"):
    print(f"   {Colors.RED}{icon} {message}{Colors.END}")


def print_warning(message, icon="⚠️"):
    print(f"   {Colors.YELLOW}{icon} {message}{Colors.END}")


def print_info(message, icon="ℹ️"):
    print(f"   {Colors.BLUE}{icon} {message}{Colors.END}")


def check_environment_variables():
    """Check relevant environment variables"""
    print_section("Environment Variables", "🌍")

    env_vars = {
        "MODEL_PATH": "/app/models/llm",
        "MODEL_NAME": "gpt2-fa",
        "CUDA_VISIBLE_DEVICES": None,
        "LLM_GPU_MEMORY_GB": "3.0",
        "GPU_COORDINATOR_URL": "http://gpu-coordinator:8080",
        "PYTHONPATH": None,
        "HF_HOME": None,
        "TRANSFORMERS_CACHE": None,
    }

    for var, default in env_vars.items():
        value = os.getenv(var, default)
        if value:
            print_info(f"{var}: {value}")
        else:
            print_warning(f"{var}: Not set")

    return {var: os.getenv(var, default) for var, default in env_vars.items()}


def generate_possible_paths(model_path: str, model_name: str) -> List[Path]:
    """Generate all possible model paths"""
    paths = []

    # Base paths to try
    base_paths = [
        model_path,
        "/app/models/llm",
        "/app/models",
        "./models",
        "/models",
        "/data/models",
        os.path.expanduser("~/models"),
    ]

    # Model names to try
    model_names = [model_name, "gpt2-fa", "persian-gpt2", "gpt2", "distilgpt2"]

    # Generate combinations
    for base_path in base_paths:
        base = Path(base_path)

        # Try base path directly
        paths.append(base)

        # Try with model names
        for name in model_names:
            paths.append(base / name)

    # Add specific known paths
    specific_paths = [
        Path("/app/models/llm/gpt2-fa"),
        Path("/app/models/persian-llm"),
    ]

    paths.extend(specific_paths)

    # Remove duplicates
    unique_paths = []
    seen = set()
    for path in paths:
        path_str = str(path.resolve())
        if path_str not in seen:
            unique_paths.append(path)
            seen.add(path_str)

    return unique_paths


def check_file_requirements(path: Path) -> Dict[str, Any]:
    """Check if a path meets model file requirements"""

    required_files = ["config.json"]
    model_files = [
        "pytorch_model.bin",
        "model.safetensors",
        "model.bin",
        "pytorch_model-00001-of-00001.bin",
    ]
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
    ]

    result = {
        "has_required": [],
        "missing_required": [],
        "has_model_files": [],
        "has_tokenizer_files": [],
        "is_valid": False,
        "contents": [],
        "file_sizes": {},
    }

    if not path.exists() or not path.is_dir():
        return result

    try:
        contents = list(path.iterdir())
        result["contents"] = [f.name for f in contents]

        # Check file sizes
        for item in contents:
            if item.is_file():
                try:
                    size_mb = item.stat().st_size / (1024 * 1024)
                    result["file_sizes"][item.name] = f"{size_mb:.1f}MB"
                except:
                    result["file_sizes"][item.name] = "Unknown"

    except Exception as e:
        result["error"] = str(e)
        return result

    # Check required files
    for req_file in required_files:
        if (path / req_file).exists():
            result["has_required"].append(req_file)
        else:
            result["missing_required"].append(req_file)

    # Check model files
    for model_file in model_files:
        if (path / model_file).exists():
            result["has_model_files"].append(model_file)

    # Check tokenizer files
    for tok_file in tokenizer_files:
        if (path / tok_file).exists():
            result["has_tokenizer_files"].append(tok_file)

    # Determine if valid
    has_config = len(result["has_required"]) == len(required_files)
    has_model = len(result["has_model_files"]) > 0
    has_tokenizer = len(result["has_tokenizer_files"]) > 0

    result["is_valid"] = has_config and has_model

    return result


def check_model_paths(env_vars: Dict):
    """Check all possible model paths"""
    print_section("Model Path Analysis", "📁")

    model_path = env_vars.get("MODEL_PATH", "/app/models/llm")
    model_name = env_vars.get("MODEL_NAME", "gpt2-fa")

    print_info(f"Base MODEL_PATH: {model_path}")
    print_info(f"Target MODEL_NAME: {model_name}")

    possible_paths = generate_possible_paths(model_path, model_name)

    print_info(f"Checking {len(possible_paths)} possible locations...")

    valid_paths = []
    partially_valid_paths = []

    for i, path in enumerate(possible_paths):
        print(f"\n{Colors.CYAN}📍 [{i + 1}/{len(possible_paths)}] {path}{Colors.END}")

        if not path.exists():
            print_error("Path does not exist")
            continue

        if not path.is_dir():
            print_error("Path exists but is not a directory")
            continue

        # Check file requirements
        file_check = check_file_requirements(path)

        if "error" in file_check:
            print_error(f"Cannot read directory: {file_check['error']}")
            continue

        # Print directory info
        print_info(f"Directory contains {len(file_check['contents'])} items")

        if file_check["contents"]:
            # Show first few files
            sample_files = file_check["contents"][:5]
            print_info(f"Sample contents: {', '.join(sample_files)}")
            if len(file_check["contents"]) > 5:
                print_info(f"... and {len(file_check['contents']) - 5} more items")

        # Check requirements
        if file_check["has_required"]:
            print_success(f"Required files: {', '.join(file_check['has_required'])}")
        if file_check["missing_required"]:
            print_warning(
                f"Missing required: {', '.join(file_check['missing_required'])}"
            )

        if file_check["has_model_files"]:
            print_success(f"Model files: {', '.join(file_check['has_model_files'])}")
            # Show file sizes for model files
            for model_file in file_check["has_model_files"]:
                if model_file in file_check["file_sizes"]:
                    print_info(
                        f"  {model_file}: {file_check['file_sizes'][model_file]}"
                    )
        else:
            print_warning("No model files found")

        if file_check["has_tokenizer_files"]:
            print_success(
                f"Tokenizer files: {', '.join(file_check['has_tokenizer_files'])}"
            )
        else:
            print_warning("No tokenizer files found")

        # Overall assessment
        if file_check["is_valid"]:
            print_success("✨ VALID MODEL DIRECTORY")
            valid_paths.append(path)
        elif file_check["has_required"]:
            print_warning("⚠️ Partially valid (has config but missing model files)")
            partially_valid_paths.append(path)
        else:
            print_error("❌ Invalid model directory")

    return valid_paths, partially_valid_paths


def check_config_files(valid_paths: List[Path]):
    """Examine config files in valid paths"""
    if not valid_paths:
        return

    print_section("Model Configuration Analysis", "⚙️")

    for path in valid_paths[:3]:  # Check first 3 valid paths
        print(f"\n{Colors.PURPLE}📝 Analyzing config in: {path}{Colors.END}")

        config_file = path / "config.json"
        if not config_file.exists():
            print_error("config.json not found")
            continue

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)

            # Extract key information
            model_type = config.get("model_type", "unknown")
            architectures = config.get("architectures", [])
            vocab_size = config.get("vocab_size", 0)
            hidden_size = config.get("hidden_size", 0)
            num_layers = config.get("num_hidden_layers", config.get("n_layer", 0))
            num_heads = config.get("num_attention_heads", config.get("n_head", 0))

            print_success(f"Model type: {model_type}")
            if architectures:
                print_info(f"Architecture: {', '.join(architectures)}")
            print_info(f"Vocabulary size: {vocab_size:,}")
            print_info(f"Hidden size: {hidden_size}")
            print_info(f"Layers: {num_layers}")
            print_info(f"Attention heads: {num_heads}")

            # Estimate model size
            if vocab_size and hidden_size and num_layers:
                # Rough parameter estimation
                embedding_params = vocab_size * hidden_size
                transformer_params = num_layers * (
                    12 * hidden_size * hidden_size
                )  # Rough estimate
                total_params = embedding_params + transformer_params
                print_info(
                    f"Estimated parameters: ~{total_params:,} ({total_params / 1e6:.1f}M)"
                )

        except Exception as e:
            print_error(f"Error reading config: {e}")


def check_python_environment():
    """Check Python environment and dependencies"""
    print_section("Python Environment", "🐍")

    print_info(f"Python version: {sys.version}")
    print_info(f"Python executable: {sys.executable}")
    print_info(f"Current working directory: {os.getcwd()}")

    # Check key libraries
    libraries = [
        ("torch", "PyTorch for model loading"),
        ("transformers", "Hugging Face transformers"),
        ("tokenizers", "Fast tokenizers"),
        ("safetensors", "SafeTensors format support"),
        ("accelerate", "Model acceleration"),
        ("bitsandbytes", "Quantization support"),
    ]

    print_info("Library availability:")
    for lib_name, description in libraries:
        try:
            lib = __import__(lib_name)
            version = getattr(lib, "__version__", "unknown")
            print_success(f"  {lib_name} {version} - {description}")
        except ImportError:
            print_warning(f"  {lib_name} - Not available - {description}")

    # Check CUDA availability
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print_success(f"CUDA available: {torch.version.cuda}")
            print_info(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print_info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print_warning("CUDA not available - using CPU")
    except ImportError:
        print_error("PyTorch not available")


def generate_recommendations(
    valid_paths: List[Path], partially_valid_paths: List[Path], env_vars: Dict
):
    """Generate recommendations for fixing issues"""
    print_section("Recommendations", "💡")

    if valid_paths:
        print_success(f"Found {len(valid_paths)} valid model path(s)!")
        print_info("Recommended actions:")
        print_info(f"1. Use this MODEL_PATH: {valid_paths[0]}")
        print_info("2. Update your environment variable or Docker configuration")
        print_info("3. Restart the LLM service")

        # Generate Docker command
        recommended_path = valid_paths[0]
        print_info("\nDocker environment suggestion:")
        print(f"   {Colors.CYAN}MODEL_PATH={recommended_path}{Colors.END}")

    elif partially_valid_paths:
        print_warning("Found partially valid paths but no complete models")
        print_info("Issues to resolve:")
        print_info("1. Model files may be missing or corrupted")
        print_info("2. Download or copy complete model files")
        print_info("3. Check if model download was interrupted")

        for path in partially_valid_paths[:2]:
            print_info(f"\nPartially valid path: {path}")
            print_info("Try downloading the model again to this location")

    else:
        print_error("No valid model paths found!")
        print_info("Solutions:")
        print_info("1. Download a model to one of these locations:")

        base_paths = ["/app/models/llm/gpt2-fa", "/app/models/llm", "/app/models"]

        for path in base_paths:
            print_info(f"   {path}")

        print_info("2. Or use a fallback model download:")
        print_info("   The service will try to download gpt2 as fallback")

        print_info("3. Check Docker volume mounts:")
        print_info("   Ensure your model directory is properly mounted")

        print_info("4. Example model download (if you have internet):")
        print(
            f"   {Colors.CYAN}huggingface-cli download gpt2 --local-dir /app/models/llm/gpt2{Colors.END}"
        )


def create_test_model_structure(base_path: str = "/tmp/test_model"):
    """Create a test model structure for validation"""
    print_section("Test Model Structure", "🧪")

    try:
        test_path = Path(base_path)
        test_path.mkdir(parents=True, exist_ok=True)

        # Create minimal config.json
        config = {
            "model_type": "gpt2",
            "architectures": ["GPT2LMHeadModel"],
            "vocab_size": 50257,
            "n_positions": 1024,
            "n_embd": 768,
            "n_layer": 12,
            "n_head": 12,
        }

        with open(test_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Create empty model file (placeholder)
        (test_path / "pytorch_model.bin").touch()

        # Create tokenizer config
        tokenizer_config = {
            "model_max_length": 1024,
            "padding_side": "right",
            "tokenizer_class": "GPT2Tokenizer",
        }

        with open(test_path / "tokenizer_config.json", "w") as f:
            json.dump(tokenizer_config, f, indent=2)

        print_success(f"Created test model structure at: {test_path}")
        print_warning("Note: This is just a structure test, not a working model")

        return test_path

    except Exception as e:
        print_error(f"Failed to create test structure: {e}")
        return None


def main():
    """Main diagnostic function"""
    print(f"{Colors.BOLD}{Colors.PURPLE}🔍 Model Path Diagnostic Tool{Colors.END}")
    print("=" * 60)

    # Step 1: Check environment
    env_vars = check_environment_variables()

    # Step 2: Check Python environment
    check_python_environment()

    # Step 3: Check model paths
    valid_paths, partially_valid_paths = check_model_paths(env_vars)

    # Step 4: Analyze configurations
    if valid_paths:
        check_config_files(valid_paths)

    # Step 5: Generate recommendations
    generate_recommendations(valid_paths, partially_valid_paths, env_vars)

    # Step 6: Summary
    print_section("Summary", "📋")

    if valid_paths:
        print_success(f"✅ Found {len(valid_paths)} valid model path(s)")
        print_info("Your LLM service should be able to load a model")
    elif partially_valid_paths:
        print_warning(f"⚠️ Found {len(partially_valid_paths)} partially valid path(s)")
        print_info("Models may need to be re-downloaded or fixed")
    else:
        print_error("❌ No valid model paths found")
        print_info("You need to provide a model or allow fallback downloads")

    print(f"\n{Colors.BOLD}{Colors.GREEN}🏁 Diagnostic completed!{Colors.END}")


if __name__ == "__main__":
    main()
