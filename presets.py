"""
Film denoiser presets: Neo Vague Denoiser + SVT-AV1 encoding parameters.
All Neo VD thresholds are 8-bit equivalent; the plugin auto-scales for 10-bit.
"""

from dataclasses import dataclass
from enum import Enum


# ── Grain synthesis ──────────────────────────────────────────────────────────


class GrainSource(Enum):
    """Source for film grain synthesis."""
    NONE = "none"
    ISO = "iso"
    PRESET = "preset"


@dataclass
class GrainPreset:
    """Grav1synth grain synthesis configuration."""
    source: GrainSource = GrainSource.NONE
    iso_value: int = 400        # ISO 100-6400
    preset_name: str = ""       # e.g. "16mm", "16mm-2", "Classic35"
    chroma: bool = False        # apply grain to chroma (ISO mode only)

    def __post_init__(self):
        if self.source == GrainSource.ISO:
            if not (100 <= self.iso_value <= 6400):
                raise ValueError(f"ISO must be 100-6400, got {self.iso_value}")
        if self.chroma and self.source != GrainSource.ISO:
            raise ValueError("chroma grain is only valid with source=ISO")


# Grav1synth built-in film stock presets (for UI dropdowns)
GRAIN_FILM_PRESETS = [
    "Super8",
    "MaxMid",
    "16mm", "16mm-1", "16mm-2", "16mm-3",
    "Classic35", "Classic35-1", "Classic35-2", "Classic35-3",
    "Modern35", "Modern35-1", "Modern35-2", "Modern35-3",
]


# ── Neo Vague Denoiser parameters ──────────────────────────────────────────

@dataclass(frozen=True)
class DenoisePreset:
    """Neo Vague Denoiser parameters for a preset level."""
    threshold: float      # wavelet coefficient threshold (8-bit equivalent)
    nsteps: int           # wavelet decomposition levels (1-32)
    percent: float        # blend percentage (1-100)
    method: int = 2       # 0=hard, 1=soft, 2=garrote (smooth, best for film)
    opt: int = 0          # CPU: 0=auto, 2=SSE, 3=SSE4.1, 5=AVX2
    process_chroma: bool = True  # True = process all planes, False = luma only

    @property
    def planes(self) -> list[int]:
        """Plane indices to process. [0]=luma always; chroma depends on preset."""
        if self.process_chroma:
            return [0, 1, 2]
        return [0]


# ── Film-tuned denoiser presets ────────────────────────────────────────────
# All use method=2 (garrote) — the smooth, continuous thresholding function
# that best preserves film grain texture without "plasticky" artifacts.

DENOISE_PRESETS: dict[str, DenoisePreset] = {
    "ultralight": DenoisePreset(
        threshold=0.7, nsteps=3, percent=60, process_chroma=False,
    ),
    "light": DenoisePreset(
        threshold=1.5, nsteps=4, percent=75,
    ),
    "medium": DenoisePreset(
        threshold=3.0, nsteps=5, percent=85,
    ),
    "strong": DenoisePreset(
        threshold=6.0, nsteps=6, percent=100,
    ),
}

DEFAULT_DENOISE_PRESET = "medium"


# ── SVT-AV1 encoding parameters ────────────────────────────────────────────

@dataclass(frozen=True)
class EncodeParams:
    """SVT-AV1 encoding parameters (defaults — no CRF, tune, or film-grain).

    CRF, tune, and extra params are passed separately to EncodeJob.
    film-grain defaults to 0 — Grav1synth handles all grain synthesis.
    """
    svt_preset: int = 4
    keyint: int = 0
    enable_overlays: int = 1
    enable_qm: int = 1
    qm_min: int = 0
    scd: int = 1
    enable_tf: int = 1
    enable_restoration: int = 1
    enable_cdef: int = 1
    sharpness: int = 0

    def to_svtav1_params(self) -> str:
        """Build colon-delimited params string (without tune, crf, film-grain)."""
        parts = [
            f"film-grain-denoise=0",
            f"enable-overlays={self.enable_overlays}",
            f"lookahead=120",
            f"scd={self.scd}",
            f"enable-qm={self.enable_qm}",
            f"qm-min={self.qm_min}",
            f"enable-tf={self.enable_tf}",
            f"enable-restoration={self.enable_restoration}",
            f"enable-cdef={self.enable_cdef}",
            f"sharpness={self.sharpness}",
        ]
        return ":".join(parts)


# ── Default encode constants ───────────────────────────────────────────────

# A standalone EncodeParams instance with all defaults
DEFAULT_ENCODE_PARAMS = EncodeParams()

# Default encode values (not from preset)
DEFAULT_SVT_PRESET = 4
DEFAULT_TUNE = "vq"         # "vq" or "psnr"
DEFAULT_CRF_CLEAN = 20      # When no denoising applied
DEFAULT_CRF_DENOISED = 26   # When denoising is active


# ── Output colorspace presets ──────────────────────────────────────────────

# Maps user choice → (VS matrix, VS transfer, VS primaries, ff primaries, ff trc, ff space)
COLORSPACE_PRESETS: dict[str, dict[str, str]] = {
    "bt709": {
        "vs_matrix": "709", "vs_transfer": "709", "vs_primaries": "709",
        "ff_primaries": "bt709", "ff_trc": "bt709", "ff_space": "bt709",
    },
    "bt2020": {
        "vs_matrix": "2020ncl", "vs_transfer": "2020_10", "vs_primaries": "2020",
        "ff_primaries": "bt2020", "ff_trc": "bt2020-10", "ff_space": "bt2020nc",
    },
    "bt601": {
        "vs_matrix": "470bg", "vs_transfer": "601", "vs_primaries": "470bg",
        "ff_primaries": "bt470bg", "ff_trc": "smpte170m", "ff_space": "bt470bg",
    },
}


# ── VapourSynth script generation ──────────────────────────────────────────

def generate_vpy(
    source_path: str,
    denoise_preset: DenoisePreset | None = None,
    *,
    start_frame: int | None = None,
    num_frames: int | None = None,
    source_filter: str = "LWLibavSource",
    target_width: int | None = None,
    target_height: int | None = None,
    output_colorspace: str | None = None,
) -> str:
    """Generate a VapourSynth Python script as a string.

    Args:
        source_path: Path to the input media file.
        denoise_preset: Optional DenoisePreset with threshold, nsteps, method, etc.
                        When None, no denoising is applied.
        start_frame: Optional trim start (0-indexed frame number).
        num_frames: Optional number of frames to output.
        source_filter: VapourSynth source filter name.
                       'LWLibavSource' for containers (mkv/mp4),
                       'LibavSMASHSource' for raw streams (ts/m2ts).
        target_width: Optional target width for resize.
        target_height: Optional target height for resize (should be mod-2).
        output_colorspace: Optional output colorspace name key in COLORSPACE_PRESETS.

    Returns:
        The VapourSynth script content as a string, ready to write to a .vpy file.
    """
    lines = [
        "import vapoursynth as vs",
        "core = vs.core",
        "",
        f'clip = core.lsmas.{source_filter}(r"{source_path}")',
    ]

    # Normalize to YUV420P10 for internal processing (Neo VD + SVT-AV1).
    # Non-4:2:0 sources lose chroma resolution; 12-bit sources are truncated.
    resize_args = ["clip = core.resize.Bicubic(clip, format=vs.YUV420P10"]
    if target_width is not None and target_height is not None:
        resize_args.append(f", width={target_width}, height={target_height}")
    if output_colorspace:
        cs = COLORSPACE_PRESETS[output_colorspace]
        resize_args.append(
            f", matrix_s={cs['vs_matrix']}, transfer_s={cs['vs_transfer']}, "
            f"primaries_s={cs['vs_primaries']}"
        )
    resize_args.append(")")
    lines.append("".join(resize_args))

    # Trim if requested
    if start_frame is not None and num_frames is not None:
        end_frame = start_frame + num_frames
        lines.append(f"clip = clip[{start_frame}:{end_frame}]")
    elif start_frame is not None:
        lines.append(f"clip = clip[{start_frame}:]")
    elif num_frames is not None:
        lines.append(f"clip = clip[:{num_frames}]")

    # Apply Neo Vague Denoiser if a preset is provided
    if denoise_preset is not None:
        planes_str = repr(denoise_preset.planes)
        lines.append(
            f"clip = core.neo_vd.VagueDenoiser("
            f"clip, "
            f"threshold={denoise_preset.threshold}, "
            f"method={denoise_preset.method}, "
            f"nsteps={denoise_preset.nsteps}, "
            f"percent={denoise_preset.percent}, "
            f"opt={denoise_preset.opt}, "
            f"planes={planes_str}"
            f")"
        )
    lines.append("clip.set_output()")
    return "\n".join(lines) + "\n"
