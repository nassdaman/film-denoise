"""
Encoding pipeline: VapourSynth → vspipe → ffmpeg libsvtav1 → final MKV.
Handles progress tracking, temp file management, error handling, and cleanup.
"""

import os
import re
import sys
import time
import atexit
import signal
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Callable
from presets import GrainSource, GrainPreset, DEFAULT_ENCODE_PARAMS

# Rich imports — fail gracefully if not installed
try:
    from rich.progress import (
        Progress, BarColumn, TextColumn, TimeRemainingColumn,  # type: ignore[no-redef]
        TaskProgressColumn, SpinnerColumn,
    )
    from rich.console import Console
    from rich.table import Table
    _RICH_OK = True
except ImportError:
    Progress = None  # type: ignore
    Console = None   # type: ignore
    Table = None     # type: ignore
    _RICH_OK = False


# ── Paths ──────────────────────────────────────────────────────────────────

_VS_LIB = os.path.expanduser("~/.local/lib/x86_64-linux-gnu")
_LSMASH_LIB = os.path.expanduser("~/.local/lsmash/lib")
_VPYTHON = os.path.expanduser("~/.local/lib/python3/dist-packages")
_VS_BIN = os.path.expanduser("~/.local/bin")

_VSPIPE = shutil.which("vspipe", path=_VS_BIN) or "vspipe"
_FFMPEG = "ffmpeg"
_FFPROBE = "ffprobe"


def _setup_env():
    """Ensure VapourSynth and lsmas libs are on LD_LIBRARY_PATH."""
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    needed = [_VS_LIB, _LSMASH_LIB]
    parts = ld.split(":") if ld else []
    for n in needed:
        if n not in parts:
            parts.insert(0, n)
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
    py = os.environ.get("PYTHONPATH", "")
    if _VPYTHON not in py:
        os.environ["PYTHONPATH"] = f"{_VPYTHON}:{py}" if py else _VPYTHON


_setup_env()

# ── Grav1synth detection ─────────────────────────────────────────────────
_GRAV1SYNTH = shutil.which("grav1synth") or os.path.expanduser("~/.cargo/bin/grav1synth")
_GRAV1SYNTH_OK = os.path.isfile(_GRAV1SYNTH) and os.access(_GRAV1SYNTH, os.X_OK)

# ── Enums and Dataclasses ──────────────────────────────────────────────────

class AudioAction(Enum):
    PASSTHROUGH = "passthrough"
    ENCODE = "encode"

class AudioCodec(Enum):
    OPUS = "opus"

@dataclass
class AudioTrack:
    """Represents an audio track with user-selected options."""
    index: int               # absolute ffprobe stream index
    codec: str               # e.g. "ac3", "dts", "opus"
    channels: int            # 2, 6, 8
    channel_layout: str      # e.g. "5.1", "stereo"
    sample_rate: int         # 44100, 48000
    lang: str                # ISO 639-2, e.g. "eng", "jpn"
    title: str               # Original track title (empty if none)
    bitrate: int | None      # bps, None for VBR
    default: bool            # Is this the source's default track?
    # Derived:
    audio_index: int = 0     # 0-based index among audio streams only
    # User choices:
    selected: bool = True
    action: AudioAction = AudioAction.PASSTHROUGH
    encode_codec: AudioCodec = AudioCodec.OPUS
    encode_bitrate: int = 128   # kbps
    custom_title: str = ""      # Override track name
    # Internal (populated after re-encode):
    _encoded_path: str = field(default="", init=False, repr=False)

@dataclass
class SubtitleTrack:
    """Represents a subtitle track with user-selected options."""
    index: int
    codec: str              # e.g. "subrip", "hdmv_pgs_subtitle"
    lang: str
    title: str
    default: bool
    forced: bool
    # User choices:
    selected: bool = True


# Temp file registry for cleanup
_temp_files: list[str] = []


def _register_temp(path: str):
    _temp_files.append(path)


def get_opus_bitrate(channels: int) -> int:
    """Return recommended Opus bitrate (kbps) based on source channel count.

    Mono/stereo (≤2ch): 128 kbps
    Surround up to 6ch: 192 kbps
    7ch+ (7.1 etc.):   256 kbps (compensates for downmix quality loss)
    """
    if channels <= 2:
        return 128
    if channels <= 6:
        return 192
    return 256


def _cleanup_temps():
    for p in _temp_files:
        try:
            if os.path.isfile(p):
                os.unlink(p)
            elif os.path.isdir(p):
                shutil.rmtree(p)
        except OSError:
            pass


def _on_signal(signum, frame):
    _cleanup_temps()
    sys.exit(1)


atexit.register(_cleanup_temps)
signal.signal(signal.SIGINT, _on_signal)
signal.signal(signal.SIGTERM, _on_signal)


# ── Probe ──────────────────────────────────────────────────────────────────

def probe(source: str) -> dict:
    """Extract media info from a video file using ffprobe."""
    cmd = [
        _FFPROBE, "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", source,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {source}: {result.stderr.strip()}")

    import json
    info = json.loads(result.stdout)

    v_stream = None
    a_streams = []
    s_streams = []
    for s in info.get("streams", []):
        if s["codec_type"] == "video" and v_stream is None:
            v_stream = s
        elif s["codec_type"] == "audio":
            a_streams.append(s)
        elif s["codec_type"] == "subtitle":
            s_streams.append(s)

    if v_stream is None:
        raise RuntimeError(f"No video stream found in {source}")

    # Parse frame rate
    fps_str = v_stream.get("r_frame_rate", "25/1")
    num, den = fps_str.split("/")
    fps = float(num) / float(den) if den != "0" else 25.0

    # Frame count
    nb_frames = v_stream.get("nb_frames")
    if nb_frames is None:
        dur = float(info.get("format", {}).get("duration", 0))
        nb_frames = int(dur * fps) if dur > 0 else 0
    else:
        nb_frames = int(nb_frames)

    # Bit depth
    bits = v_stream.get("bits_per_raw_sample")
    if bits is None:
        pix = v_stream.get("pix_fmt", "yuv420p")
        if "10" in pix or "p010" in pix or "p210" in pix:
            bits = 10
        elif "12" in pix:
            bits = 12
        else:
            bits = 8

    return {
        "width": v_stream.get("width", 0),
        "height": v_stream.get("height", 0),
        "fps": fps,
        "frames": nb_frames,
        "duration": float(info.get("format", {}).get("duration", 0)),
        "codec": v_stream.get("codec_name", "unknown"),
        "pix_fmt": v_stream.get("pix_fmt", "yuv420p"),
        "bit_depth": int(bits),
        "color_primaries": v_stream.get("color_primaries", "bt709"),
        "color_transfer": v_stream.get("color_transfer", "bt709"),
        "color_space": v_stream.get("color_space", "bt709"),
        "color_range": v_stream.get("color_range", "tv"),
        "chroma_location": v_stream.get("chroma_location", "left"),
        "audio_streams": a_streams,
        "subtitle_streams": s_streams,
        "path": source,
        "basename": os.path.splitext(os.path.basename(source))[0],
        "video_start_time": float(v_stream.get("start_time", "0") or "0"),
        "audio_start_time": 0.0,  # probe() is video-only; placeholder for API parity
    }


def _safe_int(v):
    """Parse int safely; returns None on non-numeric strings like ffprobe's 'N/A'."""
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def probe_rich(source: str) -> dict:
    """Extended probe — returns full track metadata lists."""
    import json
    cmd = [
        _FFPROBE, "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", source,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {source}: {result.stderr.strip()}")

    info = json.loads(result.stdout)

    # Video
    v_stream = None
    a_streams = []
    s_streams = []
    video_count = 0
    for s in info.get("streams", []):
        if s["codec_type"] == "video":
            video_count += 1
            if v_stream is None:
                v_stream = s
        elif s["codec_type"] == "audio":
            a_streams.append(s)
        elif s["codec_type"] == "subtitle":
            s_streams.append(s)

    if v_stream is None:
        raise RuntimeError(f"No video stream found in {source}")

    fps_str = v_stream.get("r_frame_rate", "25/1")
    num, den = fps_str.split("/")
    fps = float(num) / float(den) if den != "0" else 25.0

    nb_frames = v_stream.get("nb_frames")
    if nb_frames is None:
        dur = float(info.get("format", {}).get("duration", 0))
        nb_frames = int(dur * fps) if dur > 0 else 0
    else:
        nb_frames = int(nb_frames)

    bits = v_stream.get("bits_per_raw_sample")
    if bits is None:
        pix = v_stream.get("pix_fmt", "yuv420p")
        if "10" in pix or "p010" in pix or "p210" in pix:
            bits = 10
        elif "12" in pix:
            bits = 12
        else:
            bits = 8

    # Parse audio tracks
    audio_tracks = []
    audio_idx = 0
    for a in a_streams:
        disp = a.get("disposition", {})
        tags = a.get("tags", {})
        audio_tracks.append(AudioTrack(
            index=a["index"],
            audio_index=audio_idx,
            codec=a.get("codec_name", "unknown"),
            channels=a.get("channels", 2),
            channel_layout=a.get("channel_layout", "stereo"),
            sample_rate=int(a.get("sample_rate", 48000)),
            lang=tags.get("language", "und"),
            title=tags.get("title", ""),
            bitrate=_safe_int(a.get("bit_rate")) if a.get("bit_rate") else None,
            default=disp.get("default", 0) == 1,
        ))
        audio_idx += 1

    # Compute earliest audio start_time across all tracks
    audio_start_times = [float(a.get("start_time", "0") or "0") for a in a_streams]
    audio_start_time = min(audio_start_times) if audio_start_times else 0.0

    # Parse subtitle tracks
    subtitles = []
    for sub in s_streams:
        disp = sub.get("disposition", {})
        tags = sub.get("tags", {})
        subtitles.append(SubtitleTrack(
            index=sub["index"],
            codec=sub.get("codec_name", "unknown"),
            lang=tags.get("language", "und"),
            title=tags.get("title", ""),
            default=disp.get("default", 0) == 1,
            forced=disp.get("forced", 0) == 1,
        ))

    return {
        "width": v_stream.get("width", 0),
        "height": v_stream.get("height", 0),
        "fps": fps,
        "frames": nb_frames,
        "duration": float(info.get("format", {}).get("duration", 0)),
        "codec": v_stream.get("codec_name", "unknown"),
        "pix_fmt": v_stream.get("pix_fmt", "yuv420p"),
        "bit_depth": int(bits),
        "video_count": video_count,
        "color_primaries": v_stream.get("color_primaries", "bt709"),
        "color_transfer": v_stream.get("color_transfer", "bt709"),
        "color_space": v_stream.get("color_space", "bt709"),
        "color_range": v_stream.get("color_range", "tv"),
        "chroma_location": v_stream.get("chroma_location", "left"),
        "audio_tracks": audio_tracks,
        "subtitle_tracks": subtitles,
        "grav1synth_available": _GRAV1SYNTH_OK,
        "video_start_time": float(v_stream.get("start_time", "0") or "0"),
        "audio_start_time": audio_start_time,
        "path": source,
        "basename": os.path.splitext(os.path.basename(source))[0],
    }


# ── Progress parsing ───────────────────────────────────────────────────────

_FRAME_RE = re.compile(r"frame=\s*(\d+)")
_FPS_RE = re.compile(r"fps=\s*([\d.]+)")
_SIZE_RE = re.compile(r"size=\s*(\d+)([kKmMgG]?)")


def _format_size(size_bytes: int) -> str:
    """Format byte count with binary prefixes (KiB, MiB, GiB)."""
    if size_bytes >= 1_073_741_824:
        return f"{size_bytes / 1_073_741_824:.1f} GiB"
    if size_bytes >= 1_048_576:
        return f"{size_bytes / 1_048_576:.1f} MiB"
    if size_bytes >= 1_024:
        return f"{size_bytes / 1_024:.1f} KiB"
    return f"{size_bytes} B"


# ── Encoding ───────────────────────────────────────────────────────────────

class EncodeJob:
    """Represents a pending or completed encode job."""

    def __init__(
        self, source: str, vpy_path: str, output: str, *,
        svt_preset: int = 4, crf: int = 20,
        tune: str = "vq", extra_params: str = "",
        grain_preset: GrainPreset | None = None,
        keyint: int = 0, total_frames: int = 0,
        bit_depth: int = 10,
        fps: float = 24.0,
        video_start_time: float = 0.0,
        audio_start_time: float = 0.0,
        color_primaries: str = "bt709", color_transfer: str = "bt709",
        color_space: str = "bt709", color_range: str = "tv",
        chroma_location: str = "left",
        pass_audio: bool = True, pass_subs: bool = True, pass_chapters: bool = True,
        audio_tracks: list[AudioTrack] | None = None,
        sub_tracks: list[SubtitleTrack] | None = None,
        progress_callback: Callable | None = None,
    ):
        self.source = source
        self.vpy_path = vpy_path
        self.output = output
        self.svt_preset = svt_preset
        self.crf = crf
        self.tune = tune
        self.extra_params = extra_params
        self.grain_preset = grain_preset
        self.keyint = keyint
        self.total_frames = total_frames
        self.bit_depth = bit_depth
        self.fps = fps
        self.video_start_time = video_start_time
        self.audio_start_time = audio_start_time
        self.color_primaries = color_primaries
        self.color_transfer = color_transfer
        self.color_space = color_space
        self.color_range = color_range
        self.chroma_location = chroma_location
        self.pass_audio = pass_audio
        self.pass_subs = pass_subs
        self.pass_chapters = pass_chapters
        self.audio_tracks = audio_tracks or []
        self.sub_tracks = sub_tracks or []
        self.progress_callback = progress_callback

        # Stats populated after encode
        self.start_time = 0.0
        self.end_time = 0.0
        self.output_size = 0
        self.avg_fps = 0.0
        self.frames_encoded = 0
        self.success = False

        # Internal
        self._temp_video = ""
        self._temp_dir = ""

    @property
    def elapsed(self) -> float:
        return self.end_time - self.start_time if self.end_time else 0

    def _build_video_cmd(self) -> list[str]:
        """Build the ffmpeg encode command (video-only, to temp IVF). Still used for grain path."""
        pix_fmt = {
            8: "yuv420p",
            10: "yuv420p10le",
            12: "yuv420p12le",
        }.get(self.bit_depth, "yuv420p10le")

        cmd = [
            _FFMPEG, "-y",
            "-f", "yuv4mpegpipe",
            "-i", "-",
            "-c:v", "libsvtav1",
            "-preset", str(self.svt_preset),
            "-crf", str(self.crf),
            "-pix_fmt", pix_fmt,
            # Frame rate from Y4M header (exact rational) — no -r flag.
            # Output -r would force frame rate conversion, silently
            # dropping/duplicating frames when float ≠ pipe rational.
        ]
        # Build SVT-AV1 params from defaults + user overrides
        tune_int = 0 if self.tune == "vq" else 1
        params_parts = [DEFAULT_ENCODE_PARAMS.to_svtav1_params(), f"tune={tune_int}"]
        # Double-grain prevention: force film-grain=0 when Grav1synth is active
        if self.grain_preset and self.grain_preset.source != GrainSource.NONE:
            params_parts.append("film-grain=0")
        # Append user custom params (last-value-wins for duplicates)
        if self.extra_params:
            params_parts.append(self.extra_params)
        svtav1_params = ":".join(params_parts)
        cmd += ["-svtav1-params", svtav1_params]
        if self.keyint > 0:
            cmd += ["-g", str(self.keyint)]

        # Color metadata
        cmd += [
            "-color_primaries", self.color_primaries,
            "-color_trc", self.color_transfer,
            "-colorspace", self.color_space,
            "-color_range", "limited" if self.color_range in ("tv", "mpeg", "limited") else "full",
            "-chroma_sample_location", self.chroma_location,
        ]

        # Map only video (no audio/subs; we mux those later)
        cmd += ["-an", "-sn"]
        # Progress output to stderr
        cmd += ["-progress", "pipe:2", "-nostats"]

        # Output to temp IVF
        tf = tempfile.NamedTemporaryFile(suffix=".ivf", prefix="fden_", delete=False)
        tf.close()
        self._temp_video = tf.name
        _register_temp(self._temp_video)
        cmd.append(self._temp_video)
        return cmd

    def _build_combined_cmd(self, video_input: str) -> list[str]:
        """Build combined ffmpeg command: video + source audio → final MKV.
        
        Args:
            video_input: "-" for stdin (non-grain) or IVF file path (grain path).
        """
        pix_fmt = {
            8: "yuv420p", 10: "yuv420p10le", 12: "yuv420p12le",
        }.get(self.bit_depth, "yuv420p10le")

        is_stdin = (video_input == "-")

        cmd = [_FFMPEG, "-y"]

        # Input 0: video (stdin or IVF file)
        if is_stdin:
            cmd += ["-f", "yuv4mpegpipe", "-i", "-"]
        else:
            cmd += ["-i", video_input]

        # Input 1: source audio (normalized timestamps)
        # Apply audio offset: if source video has a start_pts offset, seek audio input
        # to skip pre-roll content that VapourSynth doesn't emit through the Y4M pipe.
        # Compute offset as difference between video and audio start times.
        offset = max(0.0, self.video_start_time - self.audio_start_time)
        if offset > 0.001:  # 1ms threshold — ignore negligible offsets
            cmd += ["-ss", str(offset)]
        # -vn -dn: skip video/data streams — we only need audio & subs from this input.
        # Without this, ffmpeg initializes all decoders (DV enhancement layer, etc.)
        # causing unusual failures (exit code 218) on DV/HDR10+ sources.
        cmd += ["-vn", "-dn", "-fflags", "+genpts+igndts", "-i", self.source]

        # Video output
        cmd += ["-map", "0:v:0"]
        if is_stdin:
            # Encoding from vspipe
            cmd += ["-c:v", "libsvtav1", "-preset", str(self.svt_preset),
                    "-crf", str(self.crf), "-pix_fmt", pix_fmt]
            # SVT-AV1 params (copy from _build_video_cmd lines 632-644)
            tune_int = 0 if self.tune == "vq" else 1
            params_parts = [DEFAULT_ENCODE_PARAMS.to_svtav1_params(), f"tune={tune_int}"]
            if self.grain_preset and self.grain_preset.source != GrainSource.NONE:
                params_parts.append("film-grain=0")
            if self.extra_params:
                params_parts.append(self.extra_params)
            cmd += ["-svtav1-params", ":".join(params_parts)]
            if self.keyint > 0:
                cmd += ["-g", str(self.keyint)]
            # Color metadata
            cmd += [
                "-color_primaries", self.color_primaries,
                "-color_trc", self.color_transfer,
                "-colorspace", self.color_space,
                "-color_range", "limited" if self.color_range in ("tv", "mpeg", "limited") else "full",
                "-chroma_sample_location", self.chroma_location,
            ]
        else:
            # Grain path: video already encoded, just copy
            cmd += ["-c:v", "copy"]

        # Audio outputs (from input 1)
        for ai, at in enumerate(self.audio_tracks):
            if not at.selected:
                continue
            if at.action == AudioAction.ENCODE:
                cmd += ["-map", f"1:a:{ai}"]
                cmd += ["-c:a", "libopus", "-b:a", f"{at.encode_bitrate}k"]
                if at.channels > 2:
                    # Multichannel: mapping_family 255 for discrete channels
                    # Avoids FFmpeg ticket #5718 (5.1(side) layout validation bug)
                    # without needing audio filter remapping
                    cmd += ["-mapping_family", "255"]
                # For stereo/mono (≤2ch): no -ac, no mapping_family needed
                # ffmpeg default (-1) handles stereo/mono correctly
            else:  # PASSTHROUGH
                # NOTE: Passthrough + audio offset may have ±1 frame imprecision
                # (~10-35ms for most codecs) because ffmpeg seeks on audio frame
                # boundaries. Opus passthrough with offset is untested — re-encode
                # to Opus is safer when offset > 0.
                cmd += ["-map", f"1:a:{ai}", "-c:a", "copy"]

        # Subtitle passthrough
        for si, st in enumerate(self.sub_tracks):
            if st.selected:
                cmd += ["-map", f"1:s:{si}", "-c:s", "copy"]

        # Progress
        cmd += ["-progress", "pipe:2", "-nostats"]

        # Ensure output timestamps start at 0 regardless of input offsets
        cmd += ["-avoid_negative_ts", "make_zero"]

        cmd.append(self.output)
        return cmd

def _cancelled_job(job: EncodeJob) -> EncodeJob:
    """Return a failed EncodeJob signalling cancellation."""
    job.end_time = time.time()
    job.success = False
    job.output_size = 0
    job.avg_fps = 0
    job.frames_encoded = 0
    return job


def run_encode(
    job: EncodeJob,
    *,
    quiet: bool = False,
    cancel_event: threading.Event | None = None,
) -> EncodeJob:
    """Run the full encode pipeline: vspipe → ffmpeg → final MKV.

    For non-grain encodes, video and audio are encoded together in a single
    ffmpeg pass (ensuring A/V sync). For grain encodes, video is encoded to
    a temp IVF first (grav1synth needs it), then muxed with audio using ffmpeg.

    Args:
        job: EncodeJob with all parameters set.
        quiet: If True, suppress progress bars and only show final stats.

    Returns:
        The same EncodeJob with success, stats, and output_size populated.
    """
    console: "Console | None" = None
    if _RICH_OK:
        console = Console(stderr=True)
    job.start_time = time.time()

    # ── Choose encoding path ───────────────────────────────────────────
    has_grain = bool(job.grain_preset and job.grain_preset.source != GrainSource.NONE)

    # ── Audio offset warning ───────────────────────────────────────────────
    if job.audio_start_time > job.video_start_time + 0.001:
        msg = (
            f"Warning: Audio starts {job.audio_start_time - job.video_start_time:.3f}s "
            f"after video in source. Audio offset cannot be negative — output may "
            f"have silence at the start."
        )
        if not quiet and console:
            console.print(f"[yellow]{msg}[/]")
        if job.progress_callback:
            try:
                job.progress_callback("warn", 0, 0, 0, 0, msg)
            except Exception:
                pass

    # Non-grain: combined ffmpeg (vspipe video + source audio → final MKV)
    # Grain: video to IVF first, then combined ffmpeg after grav1synth
    if has_grain:
        ffmpeg_cmd = job._build_video_cmd()  # video-only to IVF
    else:
        ffmpeg_cmd = job._build_combined_cmd("-")  # video+audio in one pass
    vspipe_cmd = [_VSPIPE, "-c", "y4m", job.vpy_path, "-"]

    # Launch vspipe pipe to ffmpeg
    # Use DEVNULL for vspipe stderr to avoid pipe buffer deadlock.
    # If debugging, set export FDEN_DEBUG=1 to capture vspipe output.
    _vs_stderr = subprocess.PIPE if os.environ.get("FDEN_DEBUG") else subprocess.DEVNULL
    try:
        vspipe_proc = subprocess.Popen(
            vspipe_cmd, stdout=subprocess.PIPE, stderr=_vs_stderr,
            env=os.environ,
        )
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd, stdin=vspipe_proc.stdout, stderr=subprocess.PIPE,
            text=True, env=os.environ,
        )
        if vspipe_proc.stdout is not None:
            vspipe_proc.stdout.close()  # allow SIGPIPE if ffmpeg dies
    except Exception as e:
        raise RuntimeError(f"Failed to start encode pipeline: {e}")

    # ── Progress tracking state ───────────────────────────────────────────
    total = job.total_frames
    if total <= 0:
        total = None
    last_update = time.time()
    _text_last = last_update
    _text_interval = 2.0
    _last_cb_update = last_update

    if not quiet and _RICH_OK and total:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TextColumn("[dim]•[/dim]"),
            TextColumn("{task.fields[fps]}"),
            TextColumn("[dim]•[/dim]"),
            TimeRemainingColumn(),
            console=console,
        )
        task = progress.add_task(
            f"Encoding {os.path.basename(job.source)}",
            total=total,
            fps="0 fps",
        )
        progress.start()
    else:
        progress = None
        task = None

    # ── Parse ffmpeg stderr for progress ──────────────────────────────────
    current_frame = 0
    current_fps = 0.0
    last_frame_report = 0

    try:
        for line in ffmpeg_proc.stderr:
            m_frame = _FRAME_RE.search(line)
            if m_frame:
                current_frame = int(m_frame.group(1))
            m_fps = _FPS_RE.search(line)
            if m_fps:
                current_fps = float(m_fps.group(1))

            now = time.time()

            # Rich progress bar
            if progress and task is not None:
                if now - last_update >= 0.25:
                    progress.update(
                        task,
                        completed=current_frame,
                        fps=f"{current_fps:.1f} fps",
                    )
                    last_update = now

            # Text-only fallback
            elif not quiet and total:
                pct = (current_frame / total * 100) if total > 0 else 0
                elapsed = now - job.start_time
                eta = ((total - current_frame) / current_fps) if current_fps > 0 else 0
                if now - _text_last >= _text_interval:
                    sys.stderr.write(
                        f"\r  [{pct:5.1f}%]  frame {current_frame}/{total}  "
                        f"{current_fps:.1f} fps  ETA {eta:.0f}s"
                    )
                    sys.stderr.flush()
                    _text_last = now

            # Cancel check
            if cancel_event and cancel_event.is_set():
                ffmpeg_proc.terminate()
                vspipe_proc.terminate()
                _cleanup_temps()
                return _cancelled_job(job)

            # Progress callback for GUI
            if job.progress_callback and (now - _last_cb_update >= 0.25):
                _last_cb_update = now
                try:
                    eta = 0.0
                    if current_fps > 0 and total:
                        eta = max(0, (total - current_frame) / current_fps)
                    should_continue = job.progress_callback(
                        "video", current_frame, total or 0,
                        current_fps, eta,
                        f"Encoding... {os.path.basename(job.source)}"
                    )
                    if not should_continue and cancel_event:
                        cancel_event.set()
                except Exception:
                    pass  # Don't let GUI callback crash the encode

            # Track last frame report for fallback ETA
            if "frame=" in line:
                last_frame_report = current_frame
    except Exception:
        pass
    finally:
        if progress:
            progress.stop()
        elif not quiet and total:
            sys.stderr.write("\r" + " " * 80 + "\r")  # clear text progress line
            sys.stderr.flush()

    # Wait for processes to complete
    ffmpeg_rc = ffmpeg_proc.wait()
    vspipe_proc.wait()

    job.end_time = time.time()
    job.frames_encoded = current_frame

    # ffmpeg exit code check
    if ffmpeg_rc != 0:
        _cleanup_temps()
        raise RuntimeError(
            f"ffmpeg exited with code {ffmpeg_rc}. "
            f"Check that SVT-AV1 encoder is available and the source is valid."
        )

    # vspipe exit code check
    vs_rc = vspipe_proc.returncode
    if vs_rc != 0:
        _cleanup_temps()
        raise RuntimeError(
            f"vspipe exited with code {vs_rc}. The VapourSynth script may be invalid. "
            f"Run with FDEN_DEBUG=1 to see vspipe errors."
        )

    # ── Post-encode handling ─────────────────────────────────────────────
    if has_grain:
        # Grain path: validate IVF, run grav1synth, then mux with combined cmd
        if not os.path.isfile(job._temp_video) or os.path.getsize(job._temp_video) == 0:
            _cleanup_temps()
            raise RuntimeError("Encoded video file is empty or missing")

        # Grav1synth grain synthesis
        if not _GRAV1SYNTH_OK:
            _cleanup_temps()
            raise RuntimeError(
                "grav1synth not found. Install from "
                "https://github.com/rust-av/grav1synth\n"
                f"Expected at: {_GRAV1SYNTH}"
            )

        if job.progress_callback:
            try:
                job.progress_callback("grain", 0, 1, 0, 0, "Applying film grain...")
            except Exception:
                pass

        grainy_fd, grainy_path = tempfile.mkstemp(
            suffix=".ivf", prefix="fden_grainy_"
        )
        os.close(grainy_fd)

        grain_cmd = [_GRAV1SYNTH, "apply", "-y", job._temp_video, "-o", grainy_path]
        if job.grain_preset.source == GrainSource.ISO:
            grain_cmd += ["--iso", str(job.grain_preset.iso_value)]
            if job.grain_preset.chroma:
                grain_cmd.append("--chroma")
        elif job.grain_preset.source == GrainSource.PRESET:
            grain_cmd += ["--preset", job.grain_preset.preset_name]

        if not quiet and console:
            desc = (f"ISO {job.grain_preset.iso_value}"
                    if job.grain_preset.source == GrainSource.ISO
                    else job.grain_preset.preset_name)
            console.print(f"[bold]Applying grain: {desc}...[/]")

        grain_result = subprocess.run(grain_cmd, capture_output=True, text=True)
        if grain_result.returncode != 0:
            _cleanup_temps()
            err = grain_result.stderr.strip()[-300:] if grain_result.stderr else "unknown error"
            raise RuntimeError(f"grav1synth failed: {err}")

        # Swap temp video: delete clean IVF, use grainy IVF
        try:
            os.unlink(job._temp_video)
        except OSError:
            pass
        if job._temp_video in _temp_files:
            _temp_files.remove(job._temp_video)
        job._temp_video = grainy_path
        _register_temp(grainy_path)

        if job.progress_callback:
            try:
                job.progress_callback("grain", 1, 1, 0, 0, "Grain applied")
            except Exception:
                pass

        if cancel_event and cancel_event.is_set():
            _cleanup_temps()
            return _cancelled_job(job)

        # After grav1synth: mux with combined ffmpeg cmd
        if job.progress_callback:
            try:
                job.progress_callback("mux", 0, 0, 0, 0, "Muxing tracks...")
            except Exception:
                pass
        if not quiet and console:
            console.print("[bold]Muxing audio/subtitles with ffmpeg...[/]")
        mux_cmd = job._build_combined_cmd(job._temp_video)
        mux_result = subprocess.run(mux_cmd, capture_output=True, text=True)
        if mux_result.returncode != 0:
            _cleanup_temps()
            raise RuntimeError(f"ffmpeg mux failed: {mux_result.stderr.strip()[-300:]}")
    else:
        # Non-grain: combined command already wrote to self.output
        if not os.path.isfile(job.output) or os.path.getsize(job.output) == 0:
            _cleanup_temps()
            raise RuntimeError("Encoded output file is empty or missing")

    # Stats
    job.output_size = os.path.getsize(job.output) if os.path.isfile(job.output) else 0
    job.avg_fps = job.frames_encoded / job.elapsed if job.elapsed > 0 else 0
    job.success = True

    _cleanup_temps()
    return job


# ── Stats display ──────────────────────────────────────────────────────────

def print_stats(job: EncodeJob, input_size: int = 0):
    """Print a summary table of encode results."""
    if not _RICH_OK:
        print(f"\n{'='*60}")
        print(f"Output:       {job.output}")
        print(f"Frames:       {job.frames_encoded}")
        print(f"Duration:     {job.elapsed:.1f}s")
        print(f"Avg FPS:      {job.avg_fps:.1f}")
        print(f"Output size:  {_format_size(job.output_size)}")
        if input_size > 0:
            ratio = input_size / job.output_size if job.output_size > 0 else 0
            print(f"Compression:  {ratio:.1f}:1")
        print(f"{'='*60}\n")
        return

    console = Console()
    table = Table(title="Encode Complete", title_style="bold green")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Output", job.output)
    table.add_row("Frames", str(job.frames_encoded))
    table.add_row("Duration", f"{job.elapsed:.1f}s ({job.elapsed/60:.1f}m)")
    table.add_row("Avg FPS", f"{job.avg_fps:.1f}")
    table.add_row("Output size", _format_size(job.output_size))
    if input_size > 0:
        ratio = input_size / job.output_size if job.output_size > 0 else 0
        table.add_row("Compression", f"{ratio:.1f}:1")
    table.add_row("CRF", str(job.crf))
    table.add_row("SVT preset", str(job.svt_preset))

    console.print()
    console.print(table)
    console.print()
