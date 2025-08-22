# ===============================
# services/llm-service/models/loader.py - ENHANCED VERSION
# ===============================
# Enhanced model loader with better path detection and fallback mechanisms

import os
import torch
import logging
from typing import Optional, Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self):
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.model_name: str = os.getenv("MODEL_NAME", "gpt2-fa")
        self.model_path: str = os.getenv("MODEL_PATH", "/app/models/llm")
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded: bool = False

        logger.info(f"🔧 ModelLoader initialized:")
        logger.info(f"   📁 Base path: {self.model_path}")
        logger.info(f"   🏷️  Model name: {self.model_name}")
        logger.info(f"   🎯 Target device: {self.device}")

    async def initialize(self):
        """Enhanced initialization with comprehensive path checking"""
        try:
            # Step 1: Find the model directory
            model_dir = await self._find_model_directory()
            if not model_dir:
                logger.error("❌ Could not find model directory")
                await self._try_download_fallback()
                return

            logger.info(f"🎯 Using model directory: {model_dir}")

            # Step 2: Load tokenizer
            await self._load_tokenizer(model_dir)

            # Step 3: Load model
            await self._load_model(model_dir)

            # Step 4: Final setup
            if self.model:
                self.model.eval()
                self.is_loaded = True
                logger.info(
                    f"✅ Model {self.model_name} loaded successfully on {self.device}"
                )

                # Log model info
                await self._log_model_info()

        except Exception as e:
            logger.error(f"❌ Failed to initialize model: {e}")
            await self._try_download_fallback()

    async def _find_model_directory(self) -> Optional[Path]:
        """Enhanced model directory detection"""

        # Generate comprehensive list of possible paths
        possible_paths = self._generate_possible_paths()

        logger.info(
            f"🔍 Searching for model in {len(possible_paths)} possible locations..."
        )

        for i, path in enumerate(possible_paths):
            logger.info(f"📁 [{i + 1}/{len(possible_paths)}] Checking: {path}")

            if not path.exists():
                logger.debug(f"   ❌ Path does not exist")
                continue

            if not path.is_dir():
                logger.debug(f"   ❌ Path is not a directory")
                continue

            # Log directory contents for debugging
            try:
                contents = [f.name for f in path.iterdir()]
                logger.info(
                    f"   📋 Contents: {contents[:10]}{'...' if len(contents) > 10 else ''}"
                )
            except Exception as e:
                logger.warning(f"   ⚠️ Could not read directory: {e}")
                continue

            # Check if this is a valid model directory
            if self._is_model_directory(path):
                logger.info(f"   ✅ Valid model directory found!")
                return path
            else:
                logger.debug(f"   ❌ Not a valid model directory")

        logger.error("❌ No valid model directory found in any location")
        return None

    def _generate_possible_paths(self) -> List[Path]:
        """Generate comprehensive list of possible model paths"""
        paths = []

        # Base paths to try
        base_paths = [
            self.model_path,
            "/app/models/llm",
            "/app/models",
            "./models",
            os.path.expanduser("~/models"),
            "/models",
            "/data/models",
        ]

        # Model names to try
        model_names = [self.model_name, "gpt2-fa", "persian-gpt2", "gpt2", "distilgpt2"]

        # Generate combinations
        for base_path in base_paths:
            base = Path(base_path)

            # Try base path directly
            paths.append(base)

            # Try with model names
            for model_name in model_names:
                paths.append(base / model_name)

        # Add some specific known paths
        specific_paths = [
            Path("/app/models/llm/gpt2-fa"),
            Path("/app/models/persian-llm"),
            Path("/app/models/llm/persian-gpt2"),
        ]

        paths.extend(specific_paths)

        # Remove duplicates while preserving order
        unique_paths = []
        seen = set()
        for path in paths:
            path_str = str(path.resolve())
            if path_str not in seen:
                unique_paths.append(path)
                seen.add(path_str)

        return unique_paths

    def _is_model_directory(self, path: Path) -> bool:
        """Enhanced model directory validation"""

        # Required files for a Hugging Face model
        required_files = ["config.json"]

        # At least one of these model files should exist
        model_files = [
            "pytorch_model.bin",
            "model.safetensors",
            "model.bin",
            "pytorch_model-00001-of-00001.bin",
        ]

        # At least one of these tokenizer files should exist
        tokenizer_files = [
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "vocab.json",
            "merges.txt",
            "tokenizer.model",
        ]

        logger.debug(f"🔍 Validating model directory: {path}")

        # Check required files
        for req_file in required_files:
            file_path = path / req_file
            if not file_path.exists():
                logger.debug(f"   ❌ Missing required file: {req_file}")
                return False
            logger.debug(f"   ✅ Found required file: {req_file}")

        # Check for at least one model file
        has_model_file = False
        for model_file in model_files:
            if (path / model_file).exists():
                logger.debug(f"   ✅ Found model file: {model_file}")
                has_model_file = True
                break

        if not has_model_file:
            logger.debug(f"   ❌ No model files found")
            return False

        # Check for at least one tokenizer file
        has_tokenizer_file = False
        for tok_file in tokenizer_files:
            if (path / tok_file).exists():
                logger.debug(f"   ✅ Found tokenizer file: {tok_file}")
                has_tokenizer_file = True
                break

        if not has_tokenizer_file:
            logger.debug(f"   ⚠️ No tokenizer files found, but continuing...")
            # Some models might not have separate tokenizer files

        logger.debug(f"   ✅ Directory validation passed")
        return True

    async def _load_tokenizer(self, model_dir: Path):
        """Load tokenizer with enhanced error handling"""
        logger.info("📝 Loading tokenizer...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir),
                local_files_only=True,
                trust_remote_code=False,
                use_fast=False,  # Use slow tokenizer for better compatibility
            )

            logger.info("✅ Tokenizer loaded successfully")

            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    logger.info("🔧 Set pad_token to eos_token")
                else:
                    # Fallback
                    self.tokenizer.pad_token = "[PAD]"
                    logger.info("🔧 Set pad_token to [PAD]")

            logger.info(f"📊 Tokenizer info:")
            logger.info(f"   Vocab size: {len(self.tokenizer)}")
            logger.info(f"   Pad token: {self.tokenizer.pad_token}")
            logger.info(f"   EOS token: {self.tokenizer.eos_token}")
            logger.info(f"   UNK token: {self.tokenizer.unk_token}")

        except Exception as e:
            logger.error(f"❌ Failed to load tokenizer: {e}")
            raise

    async def _load_model(self, model_dir: Path):
        """Load model with enhanced configuration"""
        logger.info("🤖 Loading model...")

        try:
            # Determine optimal settings
            if torch.cuda.is_available():
                torch_dtype = torch.float16
                device_map = "auto"
                logger.info("🎮 Using CUDA with float16")
            else:
                torch_dtype = torch.float32
                device_map = None
                logger.info("🖥️ Using CPU with float32")

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                local_files_only=True,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
                # Add more compatibility options
                ignore_mismatched_sizes=True,  # In case of size mismatches
            )

            # Move to device if needed
            if device_map is None:
                self.model = self.model.to(self.device)

            logger.info("✅ Model loaded successfully")

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")

            # Try with reduced precision or compatibility mode
            if torch.cuda.is_available():
                try:
                    logger.info("🔄 Retrying with CPU and float32...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        str(model_dir),
                        local_files_only=True,
                        torch_dtype=torch.float32,
                        device_map=None,
                        trust_remote_code=False,
                        low_cpu_mem_usage=True,
                        ignore_mismatched_sizes=True,
                    )
                    self.model = self.model.to("cpu")
                    self.device = "cpu"
                    logger.info("✅ Model loaded on CPU as fallback")
                except Exception as e2:
                    logger.error(f"❌ CPU fallback also failed: {e2}")
                    raise e2
            else:
                raise e

    async def _log_model_info(self):
        """Log detailed model information"""
        try:
            if self.model:
                num_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(
                    p.numel() for p in self.model.parameters() if p.requires_grad
                )

                logger.info(f"📊 Model Information:")
                logger.info(f"   Model type: {type(self.model).__name__}")
                logger.info(f"   Total parameters: {num_params:,}")
                logger.info(f"   Trainable parameters: {trainable_params:,}")
                logger.info(f"   Device: {self.device}")

                # Memory usage if CUDA
                if torch.cuda.is_available() and self.device == "cuda":
                    current_memory = torch.cuda.memory_allocated() / 1024**3
                    max_memory = torch.cuda.max_memory_allocated() / 1024**3
                    logger.info(
                        f"   GPU memory: {current_memory:.2f}GB / {max_memory:.2f}GB peak"
                    )

        except Exception as e:
            logger.warning(f"⚠️ Could not log model info: {e}")

    async def _try_download_fallback(self):
        """Try to download a fallback model if no local model found"""
        logger.info("🌐 Attempting to download fallback model...")

        fallback_models = ["gpt2", "distilgpt2"]

        for model_name in fallback_models:
            try:
                logger.info(f"📥 Trying to download {model_name}...")

                # Try to download and cache
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                )

                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                self.model = self.model.to(self.device)
                self.model.eval()
                self.is_loaded = True
                self.model_name = f"{model_name}-fallback"

                logger.info(f"✅ Downloaded and loaded {model_name} as fallback")
                await self._log_model_info()
                return

            except Exception as e:
                logger.error(f"❌ Failed to download {model_name}: {e}")
                continue

        logger.error("❌ All fallback download attempts failed")
        self.is_loaded = False

    async def reload(self):
        """Reload the model"""
        logger.info("🔄 Reloading model...")
        await self.cleanup()
        await self.initialize()

    async def cleanup(self):
        """Clean up model and free memory"""
        logger.info("🧹 Cleaning up model...")

        if self.model:
            del self.model
            self.model = None

        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.is_loaded = False
        logger.info("✅ Model cleanup completed")

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        if not self.is_loaded:
            return {
                "total_parameters": 0,
                "trainable_parameters": 0,
                "device": self.device,
                "model_type": "Unknown",
                "vocab_size": 0,
                "status": "not_loaded",
                "model_name": self.model_name,
                "model_path": self.model_path,
            }

        try:
            num_params = (
                sum(p.numel() for p in self.model.parameters()) if self.model else 0
            )
            trainable_params = (
                sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                if self.model
                else 0
            )

            # Additional info
            info = {
                "total_parameters": num_params,
                "trainable_parameters": trainable_params,
                "device": str(self.device),
                "model_type": type(self.model).__name__ if self.model else "Unknown",
                "vocab_size": len(self.tokenizer) if self.tokenizer else 0,
                "status": "loaded",
                "model_name": self.model_name,
                "model_path": self.model_path,
                "torch_dtype": str(self.model.dtype) if self.model else "Unknown",
            }

            # GPU memory info if available
            if torch.cuda.is_available() and self.device == "cuda":
                info["gpu_memory"] = {
                    "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                    "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                    "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                }

            return info

        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {
                "total_parameters": 0,
                "trainable_parameters": 0,
                "device": str(self.device),
                "model_type": "Unknown",
                "vocab_size": 0,
                "status": "error",
                "error": str(e),
                "model_name": self.model_name,
                "model_path": self.model_path,
            }

    def set_model_path(self, path: str):
        """Set model path - useful for testing or different model locations"""
        self.model_path = path
        logger.info(f"📁 Model path updated to: {path}")

    def get_model(self):
        """Get the loaded model"""
        return self.model

    def get_tokenizer(self):
        """Get the loaded tokenizer"""
        return self.tokenizer

    def is_model_loaded(self) -> bool:
        """Check if model is properly loaded"""
        return self.is_loaded and self.model is not None and self.tokenizer is not None

    async def validate_model(self) -> Dict[str, Any]:
        """Validate model functionality with test generation"""
        if not self.is_model_loaded():
            return {
                "valid": False,
                "error": "Model not loaded",
                "tests_passed": 0,
                "total_tests": 0,
            }

        try:
            logger.info("🧪 Validating model with test generation...")

            test_prompts = [
                "Hello",
                "Test",
                "سلام",  # Persian greeting
            ]

            tests_passed = 0
            total_tests = len(test_prompts)
            test_results = []

            for prompt in test_prompts:
                try:
                    # Simple test generation
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=50,
                    )

                    # Move to device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_length=inputs["input_ids"].shape[1] + 10,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )

                    # Decode
                    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
                    generated_text = self.tokenizer.decode(
                        generated_tokens, skip_special_tokens=True
                    )

                    test_results.append(
                        {
                            "prompt": prompt,
                            "generated": generated_text[:50],  # Truncate for logging
                            "success": True,
                        }
                    )
                    tests_passed += 1

                except Exception as e:
                    test_results.append(
                        {"prompt": prompt, "error": str(e), "success": False}
                    )
                    logger.warning(f"⚠️ Test failed for prompt '{prompt}': {e}")

            success_rate = tests_passed / total_tests
            is_valid = success_rate >= 0.5  # At least 50% success rate

            logger.info(
                f"🧪 Model validation: {tests_passed}/{total_tests} tests passed ({success_rate:.1%})"
            )

            return {
                "valid": is_valid,
                "tests_passed": tests_passed,
                "total_tests": total_tests,
                "success_rate": success_rate,
                "test_results": test_results,
            }

        except Exception as e:
            logger.error(f"❌ Model validation failed: {e}")
            return {
                "valid": False,
                "error": str(e),
                "tests_passed": 0,
                "total_tests": len(test_prompts),
            }
