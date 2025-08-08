#!/usr/bin/env python3
"""
STT Service Configuration Module
ماژول تنظیمات سرویس تبدیل گفتار به متن
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch

logger = logging.getLogger(__name__)


class STTConfig:
    """Configuration manager for STT Service"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize STT configuration

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "stt_config.yaml"
        )
        self.config_data = {}

        # Load configuration
        self._load_config()
        self._load_environment_overrides()
        self._validate_config()

        logger.info(f"✅ STT Configuration loaded from {self.config_path}")

    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.config_data = yaml.safe_load(f) or {}
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                logger.warning(
                    f"Config file not found: {self.config_path}, using defaults"
                )
                self.config_data = self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load config: {e}, using defaults")
            self.config_data = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "stt_service": {
                "model": {
                    "name": "whisper",
                    "size": "base",
                    "device": "auto",
                    "download_root": "/app/models/stt",
                },
                "audio": {
                    "max_file_size": 25,
                    "max_duration": 600,
                    "supported_formats": [
                        ".wav",
                        ".mp3",
                        ".m4a",
                        ".flac",
                        ".ogg",
                        ".webm",
                    ],
                    "sample_rate": 16000,
                },
                "languages": {
                    "supported": ["fa", "en"],
                    "auto_detect": True,
                    "default": "auto",
                },
                "processing": {
                    "normalize_audio": True,
                    "remove_silence": True,
                    "enhance_audio": True,
                    "chunk_duration": 30,
                    "overlap_duration": 1,
                },
                "performance": {
                    "batch_size": 1,
                    "num_workers": 1,
                    "max_concurrent_requests": 5,
                    "timeout": 300,
                },
                "cache": {"enabled": True, "ttl": 3600, "max_size": 100},
                "logging": {
                    "level": "INFO",
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "file": "/app/logs/stt.log",
                },
            }
        }

    def _load_environment_overrides(self):
        """Load configuration overrides from environment variables"""

        # Model configuration
        if os.getenv("WHISPER_MODEL_SIZE"):
            self._set_nested_value(
                "stt_service.model.size", os.getenv("WHISPER_MODEL_SIZE")
            )

        if os.getenv("WHISPER_MODEL_DEVICE"):
            self._set_nested_value(
                "stt_service.model.device", os.getenv("WHISPER_MODEL_DEVICE")
            )

        if os.getenv("WHISPER_MODEL_PATH"):
            self._set_nested_value(
                "stt_service.model.download_root", os.getenv("WHISPER_MODEL_PATH")
            )

        # Audio configuration
        if os.getenv("MAX_FILE_SIZE_MB"):
            try:
                size = int(os.getenv("MAX_FILE_SIZE_MB"))
                self._set_nested_value("stt_service.audio.max_file_size", size)
            except ValueError:
                logger.warning("Invalid MAX_FILE_SIZE_MB value")

        if os.getenv("MAX_AUDIO_DURATION"):
            try:
                duration = int(os.getenv("MAX_AUDIO_DURATION"))
                self._set_nested_value("stt_service.audio.max_duration", duration)
            except ValueError:
                logger.warning("Invalid MAX_AUDIO_DURATION value")

        # Performance configuration
        if os.getenv("MAX_CONCURRENT_REQUESTS"):
            try:
                max_requests = int(os.getenv("MAX_CONCURRENT_REQUESTS"))
                self._set_nested_value(
                    "stt_service.performance.max_concurrent_requests", max_requests
                )
            except ValueError:
                logger.warning("Invalid MAX_CONCURRENT_REQUESTS value")

        # Cache configuration
        if os.getenv("CACHE_ENABLED"):
            enabled = os.getenv("CACHE_ENABLED").lower() in ["true", "1", "yes", "on"]
            self._set_nested_value("stt_service.cache.enabled", enabled)

        if os.getenv("CACHE_TTL"):
            try:
                ttl = int(os.getenv("CACHE_TTL"))
                self._set_nested_value("stt_service.cache.ttl", ttl)
            except ValueError:
                logger.warning("Invalid CACHE_TTL value")

        # Logging
        if os.getenv("LOG_LEVEL"):
            self._set_nested_value("stt_service.logging.level", os.getenv("LOG_LEVEL"))

    def _set_nested_value(self, key_path: str, value: Any):
        """Set nested dictionary value using dot notation"""
        keys = key_path.split(".")
        current = self.config_data

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _get_nested_value(self, key_path: str, default: Any = None) -> Any:
        """Get nested dictionary value using dot notation"""
        keys = key_path.split(".")
        current = self.config_data

        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default

    def _validate_config(self):
        """Validate configuration values"""

        # Validate model size
        valid_model_sizes = ["tiny", "base", "small", "medium", "large"]
        model_size = self.get_model_size()
        if model_size not in valid_model_sizes:
            logger.warning(f"Invalid model size '{model_size}', using 'base'")
            self._set_nested_value("stt_service.model.size", "base")

        # Validate device
        device = self.get_device()
        if device == "auto":
            optimal_device = self._get_optimal_device()
            self._set_nested_value("stt_service.model.device", optimal_device)
            logger.info(f"Auto-selected device: {optimal_device}")

        # Validate file size limits
        max_file_size = self.get_max_file_size()
        if max_file_size <= 0 or max_file_size > 100:
            logger.warning(f"Invalid max_file_size {max_file_size}MB, using 25MB")
            self._set_nested_value("stt_service.audio.max_file_size", 25)

        # Validate concurrent requests
        max_requests = self.get_max_concurrent_requests()
        if max_requests <= 0 or max_requests > 20:
            logger.warning(f"Invalid max_concurrent_requests {max_requests}, using 5")
            self._set_nested_value("stt_service.performance.max_concurrent_requests", 5)

    def _get_optimal_device(self) -> str:
        """Determine optimal device"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        return "cpu"

    # Configuration property getters
    def get_model_name(self) -> str:
        return self._get_nested_value("stt_service.model.name", "whisper")

    def get_model_size(self) -> str:
        return self._get_nested_value("stt_service.model.size", "base")

    def get_device(self) -> str:
        return self._get_nested_value("stt_service.model.device", "auto")

    def get_download_root(self) -> str:
        return self._get_nested_value(
            "stt_service.model.download_root", "/app/models/stt"
        )

    def get_max_file_size(self) -> int:
        """Get max file size in MB"""
        return self._get_nested_value("stt_service.audio.max_file_size", 25)

    def get_max_duration(self) -> int:
        """Get max duration in seconds"""
        return self._get_nested_value("stt_service.audio.max_duration", 600)

    def get_supported_formats(self) -> List[str]:
        return self._get_nested_value(
            "stt_service.audio.supported_formats",
            [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"],
        )

    def get_sample_rate(self) -> int:
        return self._get_nested_value("stt_service.audio.sample_rate", 16000)

    def get_supported_languages(self) -> List[str]:
        return self._get_nested_value("stt_service.languages.supported", ["fa", "en"])

    def get_auto_detect(self) -> bool:
        return self._get_nested_value("stt_service.languages.auto_detect", True)

    def get_default_language(self) -> str:
        return self._get_nested_value("stt_service.languages.default", "auto")

    def get_normalize_audio(self) -> bool:
        return self._get_nested_value("stt_service.processing.normalize_audio", True)

    def get_remove_silence(self) -> bool:
        return self._get_nested_value("stt_service.processing.remove_silence", True)

    def get_enhance_audio(self) -> bool:
        return self._get_nested_value("stt_service.processing.enhance_audio", True)

    def get_chunk_duration(self) -> float:
        return self._get_nested_value("stt_service.processing.chunk_duration", 30.0)

    def get_overlap_duration(self) -> float:
        return self._get_nested_value("stt_service.processing.overlap_duration", 1.0)

    def get_batch_size(self) -> int:
        return self._get_nested_value("stt_service.performance.batch_size", 1)

    def get_num_workers(self) -> int:
        return self._get_nested_value("stt_service.performance.num_workers", 1)

    def get_max_concurrent_requests(self) -> int:
        return self._get_nested_value(
            "stt_service.performance.max_concurrent_requests", 5
        )

    def get_timeout(self) -> int:
        return self._get_nested_value("stt_service.performance.timeout", 300)

    def get_cache_enabled(self) -> bool:
        return self._get_nested_value("stt_service.cache.enabled", True)

    def get_cache_ttl(self) -> int:
        return self._get_nested_value("stt_service.cache.ttl", 3600)

    def get_cache_max_size(self) -> int:
        return self._get_nested_value("stt_service.cache.max_size", 100)

    def get_log_level(self) -> str:
        return self._get_nested_value("stt_service.logging.level", "INFO")

    def get_log_format(self) -> str:
        return self._get_nested_value(
            "stt_service.logging.format",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def get_log_file(self) -> str:
        return self._get_nested_value("stt_service.logging.file", "/app/logs/stt.log")

    def to_dict(self) -> Dict[str, Any]:
        """Get complete configuration as dictionary"""
        return self.config_data

    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary for logging/debugging"""
        return {
            "model": {
                "name": self.get_model_name(),
                "size": self.get_model_size(),
                "device": self.get_device(),
            },
            "audio": {
                "max_file_size_mb": self.get_max_file_size(),
                "max_duration_seconds": self.get_max_duration(),
                "supported_formats": len(self.get_supported_formats()),
                "sample_rate": self.get_sample_rate(),
            },
            "languages": {
                "supported": self.get_supported_languages(),
                "auto_detect": self.get_auto_detect(),
            },
            "performance": {
                "max_concurrent_requests": self.get_max_concurrent_requests(),
                "timeout": self.get_timeout(),
            },
            "cache": {"enabled": self.get_cache_enabled(), "ttl": self.get_cache_ttl()},
            "processing": {
                "normalize_audio": self.get_normalize_audio(),
                "remove_silence": self.get_remove_silence(),
                "enhance_audio": self.get_enhance_audio(),
            },
        }
