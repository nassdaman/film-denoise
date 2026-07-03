#!/usr/bin/env python3
"""
Film Denoiser — VapourSynth + Neo Vague Denoiser + SVT-AV1 encoding.

Usage:
  film-denoiser encode input.mkv [--denoise-preset medium] [--output out.mkv]
  film-denoiser preview input.mkv [--denoise-preset medium] [--start 10:00] [--duration 60]
  film-denoiser info input.mkv
"""

import os
import sys
import tempfile
from dataclasses import replace

# Ensure project directory is importable
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import click

from presets import (
    DENOISE_PRESETS, DEFAULT_DENOISE_PRESET, DenoisePreset,
    generate_vpy, GrainSource, GrainPreset,
    DEFAULT_SVT_PRESET, DEFAULT_TUNE, DEFAULT_CRF_CLEAN, DEFAULT_CRF_DENOISED,
    DEFAULT_ENCODE_PARAMS, COLORSPACE_PRESETS,
)
from encoder import (
    EncodeJob, run_encode, print_stats, probe, probe_rich, _format_size,
    _setup_env, _GRAV1SYNTH, _GRAV1SYNTH_OK,
    AudioTrack, AudioAction, AudioCodec, SubtitleTrack,
    get_opus_bitrate,
)
from preview import parse_time_to_seconds


_setup_env()


# ── Shared options ─────────────────────────────────────────────────────────

def _parse_audio_tracks(
    spec: str,
    source_audio: list[AudioTrack],
    source_subs: list[SubtitleTrack],
) -> tuple[list[AudioTrack], list[SubtitleTrack]]:
    """Parse --audio-tracks spec string into configured AudioTrack/SubtitleTrack lists.

    Format: INDEX:(p|e):CODEC:BITRATE
    Examples: "0:p,1:e:opus:128"  or  "0:p,2:p"

    Unspecified audio tracks are deselected. All subtitle tracks remain selected by default.
    """

    # Deselect all by default; each spec entry re-enables the specified track
    for a in source_audio:
        a.selected = False

    if not spec.strip():
        return source_audio, source_subs

    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) < 2:
            raise click.BadParameter(
                f"Invalid audio track spec: '{item}'. "
                f"Format: INDEX:p or INDEX:e:opus:BITRATE"
            )

        try:
            track_idx = int(parts[0])
        except ValueError:
            raise click.BadParameter(
                f"Invalid track index in spec: '{item}'"
            )

        # Find the track
        track = None
        for a in source_audio:
            if a.index == track_idx:
                track = a
                break

        if track is None:
            raise click.BadParameter(
                f"Audio track {track_idx} not found in source. "
                f"Available: {[a.index for a in source_audio]}"
            )

        track.selected = True
        action = parts[1].lower()

        if action == "p":
            track.action = AudioAction.PASSTHROUGH
        elif action == "e":
            track.action = AudioAction.ENCODE
            # Parse optional codec and bitrate
            if len(parts) >= 3:
                codec_str = parts[2].lower()
                if codec_str in ("opus",):
                    track.encode_codec = AudioCodec.OPUS
                else:
                    raise click.BadParameter(
                        f"Unknown codec '{codec_str}' for track {track_idx}. "
                        f"Supported: opus"
                    )
            if len(parts) >= 4:
                try:
                    track.encode_bitrate = int(parts[3])
                except ValueError:
                    raise click.BadParameter(
                        f"Invalid bitrate '{parts[3]}' for track {track_idx}"
                    )
            else:
                track.encode_bitrate = get_opus_bitrate(track.channels)
        else:
            raise click.BadParameter(
                f"Unknown action '{action}' for track {track_idx}. "
                f"Use 'p' (passthrough) or 'e' (encode)"
            )

    return source_audio, source_subs


# ── Commands ───────────────────────────────────────────────────────────────

@click.group()
@click.version_option(version="0.1.0", prog_name="film-denoiser")
def cli():
    """Film denoiser — Neo Vague Denoiser + SVT-AV1 encoding pipeline.

    Uses VapourSynth for wavelet-based film grain denoising,
    encodes with mainline SVT-AV1 via ffmpeg's libsvtav1,
    and remuxes with mkvmerge for clean output.
    """
    pass


@cli.command()
@click.argument("source", type=click.Path(exists=True))
@click.option(
    "--denoise-preset", "-D", default=DEFAULT_DENOISE_PRESET,
    type=click.Choice(list(DENOISE_PRESETS.keys())),
    help="Denoiser preset (default: medium). Use --no-denoise to skip denoising.",
)
@click.option(
    "--no-denoise", is_flag=True, default=False,
    help="Skip denoising entirely (ignores --denoise-preset).",
)
@click.option(
    "--threshold", type=float, default=None,
    help="Override Neo VD threshold.",
)
@click.option(
    "--nsteps", type=int, default=None,
    help="Override Neo VD wavelet levels.",
)
@click.option(
    "--grain-iso", type=click.IntRange(100, 6400), default=None,
    help="Apply photon-noise grain at given ISO (100-6400) via Grav1synth.",
)
@click.option(
    "--grain-preset", type=str, default=None,
    help="Apply film stock grain preset via Grav1synth (e.g. '16mm', '16mm-1').",
)
@click.option(
    "--grain-chroma", is_flag=True, default=False,
    help="Apply grain to chroma planes (ISO mode only).",
)
@click.option(
    "--crf", type=click.IntRange(0, 63), default=DEFAULT_CRF_CLEAN,
    help=f"SVT-AV1 CRF (default: {DEFAULT_CRF_CLEAN}). Use {DEFAULT_CRF_DENOISED} with denoising.",
)
@click.option(
    "--svt-preset", type=click.IntRange(0, 13), default=DEFAULT_SVT_PRESET,
    help=f"SVT-AV1 encoder preset 0-13 (default: {DEFAULT_SVT_PRESET}).",
)
@click.option(
    "--tune", type=click.Choice(["vq", "psnr"]), default=DEFAULT_TUNE,
    help=f"SVT-AV1 tune: vq (visual quality) or psnr (default: {DEFAULT_TUNE}).",
)
@click.option(
    "--svtav1-params", type=str, default=None,
    help="Additional colon-delimited SVT-AV1 params (appended after defaults, last-value-wins).",
)
@click.option(
    "--output", "-o", type=click.Path(), default=None,
    help="Output file path (default: <source>_denoised.mkv).",
)
@click.option(
    "--quiet", "-q", is_flag=True, default=False,
    help="Suppress progress bars, show only final stats.",
)
@click.option(
    "--no-audio", is_flag=True, default=False,
    help="Strip audio from output.",
)
@click.option(
    "--no-subs", is_flag=True, default=False,
    help="Strip subtitles from output.",
)
@click.option(
    "--audio-tracks", type=str, default=None,
    help=(
        "Audio track selection: comma-separated. "
        "Format: INDEX:p (passthrough) or INDEX:e:opus[:BITRATE] "
        "(bitrate auto: 128 kbps stereo, 192 kbps 5.1+) "
        "Example: --audio-tracks 0:p,1:e:opus"
    ),
)
@click.option(
    "--keep-temps", is_flag=True, default=False,
    help="Keep temporary files (for debugging).",
)
@click.option("--scale-1080p", is_flag=True, default=False,
              help="Downscale output to 1920 width (height proportional, mod-2).")
@click.option("--colorspace", type=click.Choice(["bt709", "bt2020", "bt601"]),
              default=None, help="Output colorspace (overrides source metadata).")
def encode(
    source, denoise_preset, no_denoise, threshold, nsteps,
    grain_iso, grain_preset, grain_chroma,
    crf, svt_preset, tune, svtav1_params,
    output, quiet, no_audio, no_subs, audio_tracks, keep_temps,
    scale_1080p, colorspace,
):
    """Encode a video file with optional film denoising, grain synthesis, and SVT-AV1."""
    # ── Probe ──────────────────────────────────────────────────────────
    info = probe_rich(source)
    fps = info["fps"]
    total_frames = info["frames"]
    source_size = os.path.getsize(source)

    if not quiet:
        click.echo(f"Source:  {source}")
        click.echo(f"  Codec:    {info['codec']} {info['width']}x{info['height']} "
                    f"@ {fps:.2f}fps, {info['frames']} frames")
        click.echo(f"  Duration: {info['duration']:.0f}s "
                    f"({info['duration']/60:.1f}m)")
        if info["audio_tracks"]:
            langs = [a.lang for a in info["audio_tracks"]]
            click.echo(f"  Audio:    {len(info['audio_tracks'])} track(s) — "
                        f"{', '.join(langs)}")
        click.echo(f"  Size:     {_format_size(source_size)}")

    # ── Denoise preset ─────────────────────────────────────────────────
    if no_denoise:
        dp = None
    else:
        dp = DENOISE_PRESETS.get(denoise_preset)
        if dp is None:
            raise click.BadParameter(
                f"Unknown denoise preset '{denoise_preset}'. "
                f"Choices: {', '.join(DENOISE_PRESETS.keys())}"
            )
        # Apply overrides
        if threshold is not None:
            dp = replace(dp, threshold=threshold)
        if nsteps is not None:
            dp = replace(dp, nsteps=nsteps)

    # ── Grain preset ───────────────────────────────────────────────────
    if grain_iso and grain_preset:
        raise click.ClickException("Cannot use both --grain-iso and --grain-preset")
    if grain_chroma and not grain_iso:
        raise click.ClickException("--grain-chroma requires --grain-iso")

    gp = None
    if grain_iso:
        gp = GrainPreset(source=GrainSource.ISO, iso_value=grain_iso,
                         chroma=grain_chroma)
    elif grain_preset:
        gp = GrainPreset(source=GrainSource.PRESET, preset_name=grain_preset)

    if gp and not _GRAV1SYNTH_OK:
        raise click.ClickException(
            f"Grav1synth not found at {_GRAV1SYNTH}. "
            f"Install from https://github.com/rust-av/grav1synth "
            f"or remove --grain-iso/--grain-preset flags."
        )

    # ── Output path ────────────────────────────────────────────────────
    if output is None:
        out_name = f"{info['basename']}_denoised.mkv"
        out_dir = os.path.dirname(os.path.abspath(source)) or "."
        output = os.path.join(out_dir, out_name)

    if os.path.exists(output):
        raise click.ClickException(
            f"Output file already exists: {output}\n"
            f"  Use --output to specify a different path, or delete the existing file."
        )

    # ── Audio track selection ─────────────────────────────────────────
    if audio_tracks:
        sel_audio, sel_subs = _parse_audio_tracks(
            audio_tracks,
            info["audio_tracks"],
            info["subtitle_tracks"],
        )
    else:
        sel_audio = []
        sel_subs = []

    if audio_tracks and no_audio:
        if not quiet:
            click.echo(
                click.style("Warning: ", fg="yellow") +
                "--no-audio is ignored when --audio-tracks is specified",
                err=True,
            )

    # ── Compute target dimensions if scaling ────────────────────────────
    target_w, target_h = None, None
    if scale_1080p:
        target_w = 1920
        target_h = round(target_w * info["height"] / info["width"] / 2) * 2

    # ── Generate VapourSynth script ────────────────────────────────────
    vpy = generate_vpy(source, dp, target_width=target_w, target_height=target_h,
                       output_colorspace=colorspace)

    if not quiet:
        click.echo()
        if dp:
            click.echo(f"Denoise: {denoise_preset} "
                        f"(threshold={dp.threshold}, nsteps={dp.nsteps}, "
                        f"percent={dp.percent})")
        else:
            click.echo("Denoise: disabled (--no-denoise)")
        if gp:
            if gp.source == GrainSource.ISO:
                grain_desc = f"ISO {gp.iso_value}" + (" (chroma)" if gp.chroma else "")
            else:
                grain_desc = gp.preset_name
            click.echo(f"Grain:   {grain_desc} (Grav1synth)")
        click.echo(f"Encode:  SVT-AV1 preset {svt_preset}, "
                    f"CRF {crf}, tune={tune}")
        click.echo(f"Output:  {output}")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".vpy", prefix="fden_", delete=False
    ) as f:
        f.write(vpy)
        vpy_path = f.name

    # ── Encode ─────────────────────────────────────────────────────────
    try:
        if not quiet:
            click.echo()
            click.echo("Starting encode...")
            click.echo()

        job = EncodeJob(
            source=source,
            vpy_path=vpy_path,
            output=output,
            svt_preset=svt_preset,
            crf=crf,
            tune=tune,
            extra_params=svtav1_params or "",
            keyint=0,  # auto
            total_frames=total_frames,
            bit_depth=10,
            fps=info["fps"],
            video_start_time=info.get("video_start_time", 0.0),
            audio_start_time=info.get("audio_start_time", 0.0),
            color_primaries=COLORSPACE_PRESETS[colorspace]["ff_primaries"] if colorspace else info["color_primaries"],
            color_transfer=COLORSPACE_PRESETS[colorspace]["ff_trc"] if colorspace else info["color_transfer"],
            color_space=COLORSPACE_PRESETS[colorspace]["ff_space"] if colorspace else info["color_space"],
            color_range=info["color_range"],
            chroma_location=info["chroma_location"],
            pass_audio=not no_audio,
            pass_subs=not no_subs,
            pass_chapters=True,
            audio_tracks=sel_audio,
            sub_tracks=sel_subs,
            grain_preset=gp,
        )

        job = run_encode(job, quiet=quiet)
        print_stats(job, input_size=source_size)

        if job.success:
            click.echo(f"\n{click.style('✓', fg='green')}  Encode complete: {output}")
        else:
            click.echo(f"\n{click.style('✗', fg='red')}  Encode failed", err=True)
            sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\nInterrupted.", err=True)
        sys.exit(130)
    except Exception as e:
        click.echo(f"\n{click.style('✗', fg='red')}  Error: {e}", err=True)
        sys.exit(1)
    finally:
        if not keep_temps and os.path.exists(vpy_path):
            os.unlink(vpy_path)


@cli.command()
@click.argument("source", type=click.Path(exists=True))
@click.option(
    "--denoise-preset", "-D", default=DEFAULT_DENOISE_PRESET,
    type=click.Choice(list(DENOISE_PRESETS.keys())),
    help="Denoiser preset (default: medium).",
)
@click.option(
    "--no-denoise", is_flag=True, default=False,
    help="Skip denoising.",
)
@click.option(
    "--start", "-s", default="300",
    help="Preview start time (seconds: 300, minutes:seconds: 5:00). Default: 300s.",
)
@click.option(
    "--duration", "-d", default="60",
    help="Preview duration in seconds. Default: 60.",
)
@click.option(
    "--crf", type=click.IntRange(0, 63), default=DEFAULT_CRF_CLEAN,
    help=f"SVT-AV1 CRF (default: {DEFAULT_CRF_CLEAN}).",
)
@click.option(
    "--svt-preset", type=click.IntRange(0, 13), default=DEFAULT_SVT_PRESET,
    help=f"SVT-AV1 encoder preset (default: {DEFAULT_SVT_PRESET}).",
)
@click.option(
    "--tune", type=click.Choice(["vq", "psnr"]), default=DEFAULT_TUNE,
    help=f"SVT-AV1 tune (default: {DEFAULT_TUNE}).",
)
@click.option(
    "--svtav1-params", type=str, default=None,
    help="Additional SVT-AV1 params.",
)
@click.option(
    "--grain-iso", type=click.IntRange(100, 6400), default=None,
    help="Apply photon-noise grain at given ISO.",
)
@click.option(
    "--grain-preset", type=str, default=None,
    help="Apply film stock grain preset.",
)
@click.option(
    "--grain-chroma", is_flag=True, default=False,
    help="Apply grain to chroma planes.",
)
@click.option(
    "--output", "-o", type=click.Path(), default=None,
    help="Output preview file (default: auto-named in /tmp).",
)
@click.option(
    "--quiet", "-q", is_flag=True, default=False,
    help="Suppress progress bars.",
)
@click.option(
    "--keep-temps", is_flag=True, default=False,
    help="Keep temporary files.",
)
@click.option("--scale-1080p", is_flag=True, default=False,
              help="Downscale output to 1920 width (height proportional, mod-2).")
@click.option("--colorspace", type=click.Choice(["bt709", "bt2020", "bt601"]),
              default=None, help="Output colorspace (overrides source metadata).")
def preview(source, denoise_preset, no_denoise, start, duration,
            crf, svt_preset, tune, svtav1_params,
            grain_iso, grain_preset, grain_chroma,
            output, quiet, keep_temps,
            scale_1080p, colorspace):
    """Generate a short preview clip for evaluation before full encode.

    Encodes a denoised clip from a specific timestamp so you can check
    quality and denoising strength without committing to a full encode.
    """
    info = probe(source)
    fps = info["fps"]

    # Denoise preset
    if no_denoise:
        dp = None
    else:
        dp = DENOISE_PRESETS.get(denoise_preset)
        if dp is None:
            raise click.BadParameter(
                f"Unknown denoise preset '{denoise_preset}'."
            )

    # Grain preset
    if grain_iso and grain_preset:
        raise click.ClickException("Cannot use both --grain-iso and --grain-preset")
    if grain_chroma and not grain_iso:
        raise click.ClickException("--grain-chroma requires --grain-iso")
    gp = None
    if grain_iso:
        gp = GrainPreset(source=GrainSource.ISO, iso_value=grain_iso,
                         chroma=grain_chroma)
    elif grain_preset:
        gp = GrainPreset(source=GrainSource.PRESET, preset_name=grain_preset)

    if gp and not _GRAV1SYNTH_OK:
        raise click.ClickException(
            f"Grav1synth not found at {_GRAV1SYNTH}. "
            f"Install from https://github.com/rust-av/grav1synth "
            f"or remove --grain-iso/--grain-preset flags."
        )

    # Parse times
    start_secs = parse_time_to_seconds(start)
    dur_secs = parse_time_to_seconds(duration)
    start_frame = int(start_secs * fps)
    num_frames = int(dur_secs * fps)

    if not quiet:
        click.echo(f"Preview: {source}")
        if dp:
            click.echo(f"  Denoise:  {denoise_preset} "
                        f"(threshold={dp.threshold}, nsteps={dp.nsteps})")
        else:
            click.echo(f"  Denoise:  disabled")
        if gp:
            if gp.source == GrainSource.ISO:
                click.echo(f"  Grain:    ISO {gp.iso_value}" +
                           (" (chroma)" if gp.chroma else ""))
            else:
                click.echo(f"  Grain:    {gp.preset_name}")
        click.echo(f"  Encode:   CRF {crf}, SVT preset {svt_preset}, tune={tune}")
        click.echo(f"  Start:    {start_secs}s (frame {start_frame})")
        click.echo(f"  Duration: {dur_secs}s ({num_frames} frames)")

    # Compute target dimensions if scaling
    target_w, target_h = None, None
    if scale_1080p:
        target_w = 1920
        target_h = round(target_w * info["height"] / info["width"] / 2) * 2

    # Generate VapourSynth script
    vpy = generate_vpy(source, dp, start_frame=start_frame, num_frames=num_frames,
                       target_width=target_w, target_height=target_h,
                       output_colorspace=colorspace)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".vpy", prefix="fden_preview_", delete=False
    ) as f:
        f.write(vpy)
        vpy_path = f.name

    try:
        if output is None:
            output = f"/tmp/preview_{os.path.splitext(os.path.basename(source))[0]}.mkv"

        job = EncodeJob(
            source=source,
            vpy_path=vpy_path,
            output=output,
            svt_preset=svt_preset,
            crf=crf,
            tune=tune,
            extra_params=svtav1_params or "",
            keyint=0,
            total_frames=num_frames,
            bit_depth=10,
            fps=info["fps"],
            color_primaries=COLORSPACE_PRESETS[colorspace]["ff_primaries"] if colorspace else info["color_primaries"],
            color_transfer=COLORSPACE_PRESETS[colorspace]["ff_trc"] if colorspace else info["color_transfer"],
            color_space=COLORSPACE_PRESETS[colorspace]["ff_space"] if colorspace else info["color_space"],
            color_range=info["color_range"],
            chroma_location=info["chroma_location"],
            pass_audio=False,
            pass_subs=False,
            pass_chapters=False,
            grain_preset=gp,
        )
        job = run_encode(job, quiet=quiet)
        print_stats(job)

        if job.success:
            click.echo(f"\n{click.style('✓', fg='green')}  Preview saved: {output}")
        else:
            click.echo(f"\n{click.style('✗', fg='red')}  Preview failed", err=True)
            sys.exit(1)
    finally:
        if not keep_temps and os.path.exists(vpy_path):
            os.unlink(vpy_path)


@cli.command()
@click.argument("source", type=click.Path(exists=True))
@click.option(
    "--preset", "-p", default=DEFAULT_DENOISE_PRESET,
    type=click.Choice(list(DENOISE_PRESETS.keys())),
    help="Show denoise preset details.",
)
def info(source, preset):
    """Display media information and preset parameters."""
    probed = probe_rich(source)

    click.echo()
    click.secho("─" * 60, dim=True)
    click.secho("  Media Info", bold=True)
    click.secho("─" * 60, dim=True)
    click.echo(f"  File:        {probed['path']}")
    click.echo(f"  Codec:       {probed['codec']}")
    click.echo(f"  Resolution:  {probed['width']}x{probed['height']}")
    click.echo(f"  FPS:         {probed['fps']:.3f}")
    click.echo(f"  Frames:      {probed['frames']}")
    click.echo(f"  Duration:    {probed['duration']:.1f}s "
                f"({probed['duration']/60:.1f}m)")
    click.echo(f"  Bit Depth:   {probed['bit_depth']}-bit "
                f"({'→ 10-bit output' if probed['bit_depth'] != 10 else '10-bit native'})")
    click.echo(f"  Pix Fmt:     {probed['pix_fmt']}")
    click.echo(f"  Color:       {probed['color_primaries']} / "
                f"{probed['color_transfer']} / {probed['color_space']} "
                f"({probed['color_range']})")
    if probed.get("video_count", 1) > 1:
        click.echo(f"  {click.style('⚠ Video:', fg='yellow')}  "
                    f"{probed['video_count']} tracks (only track #0 processed)")

    if probed.get("grav1synth_available"):
        click.secho(f"  Grav1synth:  available", fg="green")
    else:
        click.secho(f"  Grav1synth:  not installed", fg="yellow")

    if probed["audio_tracks"]:
        click.echo(f"  Audio:       {len(probed['audio_tracks'])} track(s)")
        for a in probed["audio_tracks"]:
            flags = ""
            if a.default:
                flags += " [default]"
            br = f"{a.bitrate // 1000}k" if a.bitrate else "VBR"
            nm = f" — {a.title}" if a.title else ""
            click.echo(f"    #{a.index}: {a.lang} {a.codec} "
                        f"{a.channels}ch {a.channel_layout} {br}{flags}{nm}")
    if probed["subtitle_tracks"]:
        click.echo(f"  Subtitles:   {len(probed['subtitle_tracks'])} track(s)")
        for s in probed["subtitle_tracks"]:
            flags = ""
            if s.default:
                flags += " [default]"
            if s.forced:
                flags += " [forced]"
            nm = f" — {s.title}" if s.title else ""
            click.echo(f"    #{s.index}: {s.lang} {s.codec}{flags}{nm}")

    # Show preset details
    click.echo()
    click.secho("─" * 60, dim=True)
    click.secho(f"  Denoise Preset: {preset}", bold=True)
    click.secho("─" * 60, dim=True)

    dp = DENOISE_PRESETS.get(preset)
    if dp:
        click.echo()
        click.secho("  Neo Vague Denoiser", bold=True)
        click.echo(f"    threshold:      {dp.threshold}")
        click.echo(f"    nsteps:         {dp.nsteps}")
        click.echo(f"    percent:        {dp.percent}%")
        click.echo(f"    method:         {dp.method} (garrote)")
        click.echo(f"    CPU opt:        auto (AVX2)")
        click.echo(f"    chroma:         {'processed' if dp.process_chroma else 'copied (luma only)'}")
        click.echo()
        click.secho("  SVT-AV1 Encode (defaults)", bold=True)
        ep = DEFAULT_ENCODE_PARAMS
        click.echo(f"    encoder preset: {ep.svt_preset}")
        click.echo(f"    tune:           {DEFAULT_TUNE} (VQ)")
        click.echo(f"    CRF (clean):    {DEFAULT_CRF_CLEAN}")
        click.echo(f"    CRF (denoised): {DEFAULT_CRF_DENOISED}")
        click.echo(f"    lookahead:      {120}")
        click.echo(f"    keyint:         {'auto' if ep.keyint == 0 else ep.keyint}")
        click.echo(f"    overlays:       {'on' if ep.enable_overlays else 'off'}")
        click.echo(f"    QM:             {'on' if ep.enable_qm else 'off'} (qm-min={ep.qm_min})")
        click.echo(f"    SCD:            {'on' if ep.scd else 'off'}")
        click.echo(f"    temporal filt:  {'on' if ep.enable_tf else 'off'}")

    # All presets quick reference
    click.echo()
    click.secho("─" * 60, dim=True)
    click.secho("  All Denoise Presets", bold=True)
    click.secho("─" * 60, dim=True)
    for name, pr in DENOISE_PRESETS.items():
        marker = "→ " if name == preset else "  "
        click.echo(f"{marker}{name:<14}  "
                    f"threshold={pr.threshold:<4} "
                    f"nsteps={pr.nsteps}  "
                    f"percent={pr.percent}%")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
