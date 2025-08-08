# me
# services/audio-service/stt/utils/audio_processor.py
#!/usr/bin/env python3
"""
Enhanced Audio Processor for STT Service
پردازشگر صوتی پیشرفته برای سرویس تبدیل گفتار به متن
"""

import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import tempfile
import os
import logging
from typing import Tuple, Optional, List
from scipy import signal
from scipy.signal import butter, filtfilt

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Enhanced audio processing utilities for STT service"""

    def __init__(self):
        self.target_sr = 16000  # Whisper's preferred sample rate
        self.supported_formats = [
            ".wav",
            ".mp3",
            ".m4a",
            ".flac",
            ".ogg",
            ".webm",
            ".aac",
        ]

    def load_and_preprocess(
        self,
        file_path: str,
        target_sr: int = 16000,
        normalize: bool = True,
        mono: bool = True,
    ) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess audio file for optimal STT performance

        Args:
            file_path: Path to audio file
            target_sr: Target sample rate (default: 16000 for Whisper)
            normalize: Whether to normalize audio amplitude
            mono: Convert to mono if True

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            logger.info(f"🔊 Loading audio: {file_path}")

            # Load audio file
            audio, sr = librosa.load(
                file_path, sr=target_sr, mono=mono, dtype=np.float32
            )

            # Validate audio
            if len(audio) == 0:
                raise ValueError("Empty audio file")

            if np.all(audio == 0):
                raise ValueError("Audio contains only silence")

            # Apply preprocessing steps
            audio = self._preprocess_audio(audio, sr, normalize)

            logger.info(f"✅ Audio loaded: {len(audio) / sr:.2f}s, {sr}Hz")

            return audio, sr

        except Exception as e:
            logger.error(f"❌ Failed to load audio {file_path}: {e}")
            raise

    def _preprocess_audio(
        self, audio: np.ndarray, sr: int, normalize: bool
    ) -> np.ndarray:
        """Apply preprocessing steps to audio"""

        # 1. Trim silence from beginning and end
        audio = self.trim_silence(audio, sr)

        # 2. Apply pre-emphasis filter
        audio = self.apply_preemphasis(audio)

        # 3. Normalize audio if requested
        if normalize:
            audio = self.normalize_audio(audio)

        # 4. Apply noise reduction
        audio = self.reduce_noise_spectral_subtraction(audio, sr)

        # 5. Apply bandpass filter for speech frequencies
        audio = self.apply_speech_filter(audio, sr)

        return audio

    def trim_silence(
        self, audio: np.ndarray, sr: int, threshold: float = 0.01, margin: float = 0.1
    ) -> np.ndarray:
        """
        Trim silence from the beginning and end of audio

        Args:
            audio: Audio signal
            sr: Sample rate
            threshold: RMS threshold for silence detection
            margin: Margin to keep around speech (in seconds)

        Returns:
            Trimmed audio
        """
        try:
            # Use librosa's built-in function
            audio_trimmed, _ = librosa.effects.trim(
                audio,
                top_db=int(-20 * np.log10(threshold)),  # Convert to dB
                frame_length=2048,
                hop_length=512,
            )

            # Add margin back
            margin_samples = int(margin * sr)
            start_idx = max(0, len(audio) - len(audio_trimmed) - margin_samples)
            end_idx = min(
                len(audio), start_idx + len(audio_trimmed) + 2 * margin_samples
            )

            return audio[start_idx:end_idx] if end_idx > start_idx else audio_trimmed

        except Exception as e:
            logger.warning(f"Trimming failed, using original: {e}")
            return audio

    def remove_silence(
        self,
        audio: np.ndarray,
        sr: int,
        threshold: float = 0.01,
        frame_length: int = 2048,
        hop_length: int = 512,
    ) -> np.ndarray:
        """
        Remove silent segments from audio (more aggressive than trimming)

        Args:
            audio: Audio signal
            sr: Sample rate
            threshold: Energy threshold for silence detection
            frame_length: Frame size for analysis
            hop_length: Hop size between frames

        Returns:
            Audio with silence removed
        """
        try:
            # Calculate RMS energy for each frame
            rms = librosa.feature.rms(
                y=audio, frame_length=frame_length, hop_length=hop_length
            )[0]

            # Find frames above threshold
            active_frames = rms > threshold

            # Convert frame indices to sample indices
            frame_times = librosa.frames_to_time(
                np.arange(len(active_frames)), sr=sr, hop_length=hop_length
            )

            # Create segments of continuous active frames
            segments = []
            start = None

            for i, active in enumerate(active_frames):
                if active and start is None:
                    start = frame_times[i]
                elif not active and start is not None:
                    segments.append((start, frame_times[i]))
                    start = None

            # Handle case where audio ends while active
            if start is not None:
                segments.append((start, frame_times[-1]))

            # Extract active segments
            if segments:
                active_audio = []
                for start_time, end_time in segments:
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)
                    active_audio.append(audio[start_sample:end_sample])

                return np.concatenate(active_audio) if active_audio else audio
            else:
                return audio

        except Exception as e:
            logger.warning(f"Silence removal failed, using original: {e}")
            return audio

    def apply_preemphasis(self, audio: np.ndarray, alpha: float = 0.97) -> np.ndarray:
        """
        Apply pre-emphasis filter to enhance high frequencies

        Args:
            audio: Input audio signal
            alpha: Pre-emphasis coefficient

        Returns:
            Pre-emphasized audio
        """
        return np.append(audio[0], audio[1:] - alpha * audio[:-1])

    def normalize_audio(self, audio: np.ndarray, method: str = "peak") -> np.ndarray:
        """
        Normalize audio amplitude

        Args:
            audio: Input audio signal
            method: Normalization method ('peak' or 'rms')

        Returns:
            Normalized audio
        """
        if method == "peak":
            # Peak normalization
            peak = np.max(np.abs(audio))
            if peak > 0:
                return audio / peak
        elif method == "rms":
            # RMS normalization
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                return audio / rms * 0.1  # Target RMS of 0.1

        return audio

    def reduce_noise_spectral_subtraction(
        self,
        audio: np.ndarray,
        sr: int,
        noise_duration: float = 0.5,
        alpha: float = 2.0,
    ) -> np.ndarray:
        """
        Reduce noise using spectral subtraction method

        Args:
            audio: Input audio signal
            sr: Sample rate
            noise_duration: Duration to estimate noise from (seconds)
            alpha: Over-subtraction factor

        Returns:
            Noise-reduced audio
        """
        try:
            # Compute STFT
            stft = librosa.stft(audio, hop_length=512, n_fft=2048)
            magnitude = np.abs(stft)
            phase = np.angle(stft)

            # Estimate noise spectrum from first few frames
            noise_frames = int(noise_duration * sr / 512)  # hop_length = 512
            noise_frames = min(
                noise_frames, magnitude.shape[1] // 4
            )  # Max 25% of audio

            if noise_frames > 0:
                noise_magnitude = np.mean(
                    magnitude[:, :noise_frames], axis=1, keepdims=True
                )

                # Spectral subtraction
                enhanced_magnitude = magnitude - alpha * noise_magnitude

                # Ensure magnitude doesn't go too low
                enhanced_magnitude = np.maximum(
                    enhanced_magnitude,
                    0.1 * magnitude,  # Floor at 10% of original
                )

                # Reconstruct audio
                enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
                enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)

                return enhanced_audio

        except Exception as e:
            logger.warning(f"Noise reduction failed, using original: {e}")

        return audio

    def apply_speech_filter(
        self, audio: np.ndarray, sr: int, low_freq: int = 300, high_freq: int = 3400
    ) -> np.ndarray:
        """
        Apply bandpass filter for speech frequencies

        Args:
            audio: Input audio signal
            sr: Sample rate
            low_freq: Lower cutoff frequency (Hz)
            high_freq: Upper cutoff frequency (Hz)

        Returns:
            Filtered audio
        """
        try:
            # Design bandpass filter
            nyquist = sr / 2
            low = low_freq / nyquist
            high = min(high_freq / nyquist, 0.95)  # Avoid aliasing

            if low >= high:
                return audio

            # Butterworth bandpass filter
            b, a = butter(4, [low, high], btype="band")

            # Apply filter
            filtered_audio = filtfilt(b, a, audio)

            return filtered_audio.astype(np.float32)

        except Exception as e:
            logger.warning(f"Speech filtering failed, using original: {e}")
            return audio

    def enhance_audio(
        self,
        audio: np.ndarray,
        sr: int,
        enhance_speech: bool = True,
        reduce_noise: bool = True,
    ) -> np.ndarray:
        """
        Comprehensive audio enhancement for STT

        Args:
            audio: Input audio signal
            sr: Sample rate
            enhance_speech: Apply speech enhancement
            reduce_noise: Apply noise reduction

        Returns:
            Enhanced audio
        """
        enhanced_audio = audio.copy()

        if reduce_noise:
            enhanced_audio = self.reduce_noise_spectral_subtraction(enhanced_audio, sr)

        if enhance_speech:
            enhanced_audio = self.apply_speech_filter(enhanced_audio, sr)

        # Final normalization
        enhanced_audio = self.normalize_audio(enhanced_audio, method="peak")

        return enhanced_audio

    def split_audio_chunks(
        self,
        audio: np.ndarray,
        sr: int,
        chunk_duration: float = 30.0,
        overlap_duration: float = 1.0,
    ) -> List[np.ndarray]:
        """
        Split long audio into overlapping chunks for processing

        Args:
            audio: Input audio signal
            sr: Sample rate
            chunk_duration: Duration of each chunk (seconds)
            overlap_duration: Overlap between chunks (seconds)

        Returns:
            List of audio chunks
        """
        chunk_samples = int(chunk_duration * sr)
        overlap_samples = int(overlap_duration * sr)

        if len(audio) <= chunk_samples:
            return [audio]

        chunks = []
        start = 0

        while start < len(audio):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]

            if len(chunk) > 0:
                chunks.append(chunk)

            # Move start with overlap
            start = end - overlap_samples

            # Break if remaining audio is too short
            if start >= len(audio) or end == len(audio):
                break

        return chunks

    def convert_format(
        self, input_path: str, output_format: str = "wav", target_sr: int = 16000
    ) -> str:
        """
        Convert audio file to specified format

        Args:
            input_path: Path to input audio file
            output_format: Target format ('wav', 'mp3', etc.)
            target_sr: Target sample rate

        Returns:
            Path to converted file
        """
        try:
            # Load audio
            audio, sr = librosa.load(input_path, sr=target_sr, mono=True)

            # Generate output path
            output_path = tempfile.mktemp(suffix=f".{output_format}")

            # Save in target format
            if output_format.lower() == "wav":
                sf.write(output_path, audio, sr, format="WAV")
            else:
                # For other formats, use pydub if available
                try:
                    from pydub import AudioSegment

                    # Save as wav first
                    temp_wav = tempfile.mktemp(suffix=".wav")
                    sf.write(temp_wav, audio, sr, format="WAV")

                    # Convert using pydub
                    audio_segment = AudioSegment.from_wav(temp_wav)
                    audio_segment.export(output_path, format=output_format)

                    # Clean up temp wav
                    os.unlink(temp_wav)

                except ImportError:
                    logger.warning(f"pydub not available, saving as WAV instead")
                    output_path = output_path.replace(f".{output_format}", ".wav")
                    sf.write(output_path, audio, sr, format="WAV")

            return output_path

        except Exception as e:
            logger.error(f"Format conversion failed: {e}")
            raise

    def validate_audio_file(self, file_path: str) -> bool:
        """
        Validate if file is a valid audio file

        Args:
            file_path: Path to audio file

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check file extension
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.supported_formats:
                return False

            # Try to load a small portion
            audio, sr = librosa.load(file_path, duration=1.0, sr=None)

            # Basic validation
            return (
                len(audio) > 0
                and sr > 0
                and not np.all(audio == 0)
                and not np.any(np.isnan(audio))
                and not np.any(np.isinf(audio))
            )

        except Exception as e:
            logger.warning(f"Audio validation failed for {file_path}: {e}")
            return False

    def get_audio_info(self, file_path: str) -> dict:
        """
        Get information about audio file

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary with audio information
        """
        try:
            # Load audio to get info
            audio, sr = librosa.load(file_path, sr=None)

            return {
                "duration": len(audio) / sr,
                "sample_rate": sr,
                "channels": 1,  # librosa loads as mono by default
                "samples": len(audio),
                "format": Path(file_path).suffix.lower(),
                "rms_energy": float(np.sqrt(np.mean(audio**2))),
                "peak_amplitude": float(np.max(np.abs(audio))),
                "zero_crossing_rate": float(
                    np.mean(librosa.feature.zero_crossing_rate(audio))
                ),
                "spectral_centroid": float(
                    np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
                ),
            }

        except Exception as e:
            logger.error(f"Failed to get audio info for {file_path}: {e}")
            return {}

    def detect_silence_segments(
        self,
        audio: np.ndarray,
        sr: int,
        threshold: float = 0.01,
        min_duration: float = 0.5,
    ) -> List[Tuple[float, float]]:
        """
        Detect silent segments in audio

        Args:
            audio: Audio signal
            sr: Sample rate
            threshold: Energy threshold for silence
            min_duration: Minimum duration for a silent segment (seconds)

        Returns:
            List of (start_time, end_time) tuples for silent segments
        """
        try:
            # Calculate RMS energy
            frame_length = 2048
            hop_length = 512

            rms = librosa.feature.rms(
                y=audio, frame_length=frame_length, hop_length=hop_length
            )[0]

            # Find silent frames
            silent_frames = rms < threshold

            # Convert to time
            frame_times = librosa.frames_to_time(
                np.arange(len(silent_frames)), sr=sr, hop_length=hop_length
            )

            # Group consecutive silent frames
            silent_segments = []
            start_time = None

            for i, is_silent in enumerate(silent_frames):
                if is_silent and start_time is None:
                    start_time = frame_times[i]
                elif not is_silent and start_time is not None:
                    end_time = frame_times[i]
                    duration = end_time - start_time

                    if duration >= min_duration:
                        silent_segments.append((start_time, end_time))

                    start_time = None

            # Handle case where audio ends in silence
            if start_time is not None:
                end_time = frame_times[-1]
                duration = end_time - start_time
                if duration >= min_duration:
                    silent_segments.append((start_time, end_time))

            return silent_segments

        except Exception as e:
            logger.error(f"Silence detection failed: {e}")
            return []
