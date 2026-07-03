#!/usr/bin/env python3
"""
Film Denoiser GUI — CustomTkinter frontend for the film denoising pipeline.
"""

import os
import subprocess
import sys
import threading
import tempfile
import uuid

# Ensure project directory is importable
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

# ── Backend imports ─────────────────────────────────────────────────────────
from encoder import (
    probe_rich, EncodeJob, AudioTrack, SubtitleTrack, AudioAction, AudioCodec,
    run_encode, _format_size, _GRAV1SYNTH_OK,
    get_opus_bitrate, _FFMPEG,
)
from presets import (
    DENOISE_PRESETS, DEFAULT_DENOISE_PRESET, DEFAULT_SVT_PRESET, DEFAULT_TUNE,
    DEFAULT_CRF_CLEAN, DEFAULT_CRF_DENOISED, GRAIN_FILM_PRESETS,
    GrainSource, GrainPreset, generate_vpy, COLORSPACE_PRESETS,
)

# ── GUI imports with graceful fallback ─────────────────────────────────────
try:
    import customtkinter as ctk
    import tkinter.filedialog as fd
    import tkinter.messagebox as mb
except ImportError:
    print("CustomTkinter is required for GUI. Install with: pip install customtkinter")
    print("The CLI is still available via: film-denoiser encode/preview/info")
    sys.exit(1)


class FilmDenoiserGUI:
    """Main GUI window for the Film Denoiser tool."""

    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Film Denoiser")
        self.root.geometry("960x720")
        self.root.minsize(800, 600)

        self.info = None
        self.encoding = False
        self.cancel_event = threading.Event()
        self.audio_track_widgets = []
        self.subtitle_track_widgets = []

        # ── Main layout ────────────────────────────────────────────────
        self.tabview = ctk.CTkTabview(self.root)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_input = self.tabview.add("Input")
        self.tab_denoise = self.tabview.add("Denoise")
        self.tab_grain = self.tabview.add("Grain Synthesis")
        self.tab_audio = self.tabview.add("Audio & Subtitles")
        self.tab_encode = self.tabview.add("Encode")

        self._build_input_tab()
        self._build_denoise_tab()
        self._build_grain_tab()
        self._build_audio_tab()
        self._build_encode_tab()

        # Close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Enter key on source entry triggers file load
        self.root.bind("<Return>", lambda e: self.load_file())

    # ── Tab 1: Input (simplified) ──────────────────────────────────────────

    def _build_input_tab(self):
        tab = self.tab_input

        # Row 1: Source file
        row1 = ctk.CTkFrame(tab, fg_color="transparent")
        row1.pack(fill="x", padx=10, pady=(10, 2))

        ctk.CTkLabel(row1, text="Source:", anchor="w", width=60).pack(side="left")
        self.source_entry = ctk.CTkEntry(row1, placeholder_text="No file selected...")
        self.source_entry.pack(side="left", fill="x", expand=True, padx=(5, 5))
        ctk.CTkButton(row1, text="Browse", command=self._browse_source, width=90).pack(side="left")

        # Row 2: Media info
        self.media_frame = ctk.CTkFrame(tab, fg_color="transparent", height=140)
        self.media_frame.pack(fill="x", padx=10, pady=5)
        self.media_frame.pack_propagate(False)

        self.media_line1 = ctk.CTkLabel(self.media_frame, text="", anchor="w", justify="left")
        self.media_line1.pack(fill="x", padx=5, pady=(5, 0))

        self.media_line2 = ctk.CTkLabel(self.media_frame, text="", anchor="w", justify="left")
        self.media_line2.pack(fill="x", padx=5, pady=0)

        self.media_line3 = ctk.CTkLabel(self.media_frame, text="", anchor="w", justify="left")
        self.media_line3.pack(fill="x", padx=5, pady=(0, 5))

        self.media_grav_label = ctk.CTkLabel(self.media_frame, text="", anchor="w", justify="left")
        self.media_grav_label.pack(fill="x", padx=5, pady=(0, 5))

        self.media_warn1 = ctk.CTkLabel(self.media_frame, text="", anchor="w", justify="left")
        self.media_warn2 = ctk.CTkLabel(self.media_frame, text="", anchor="w", justify="left")

        # Row 3: Output path
        row3 = ctk.CTkFrame(tab, fg_color="transparent")
        row3.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(row3, text="Output:", anchor="w", width=60).pack(side="left")
        self.output_entry = ctk.CTkEntry(row3, placeholder_text="(auto)")
        self.output_entry.pack(side="left", fill="x", expand=True, padx=(5, 5))
        ctk.CTkButton(row3, text="Browse", command=self._browse_output, width=90).pack(side="left")

        # stretch at bottom
        ctk.CTkLabel(tab, text="").pack(fill="both", expand=True)

    # ── Tab 2: Denoise ─────────────────────────────────────────────────────

    def _build_denoise_tab(self):
        tab = self.tab_denoise

        # Enable checkbox
        self.denoise_enabled = ctk.BooleanVar(value=True)
        self.denoise_cb = ctk.CTkCheckBox(
            tab, text="Enable Denoising", variable=self.denoise_enabled,
            command=self._on_denoise_toggle,
        )
        self.denoise_cb.pack(anchor="w", padx=10, pady=(10, 5))

        # Preset row
        preset_row = ctk.CTkFrame(tab, fg_color="transparent")
        preset_row.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(preset_row, text="Preset:", anchor="w", width=60).pack(side="left")
        self.denoise_preset_var = ctk.StringVar(value=DEFAULT_DENOISE_PRESET)
        self.denoise_preset_combo = ctk.CTkOptionMenu(
            preset_row, values=list(DENOISE_PRESETS.keys()),
            variable=self.denoise_preset_var,
            command=self._on_denoise_preset_change, width=140,
        )
        self.denoise_preset_combo.pack(side="left", padx=(5, 10))

        # Description label
        initial_dp = DENOISE_PRESETS.get(DEFAULT_DENOISE_PRESET)
        if initial_dp:
            desc = (
                f"threshold={initial_dp.threshold}, nsteps={initial_dp.nsteps}, "
                f"chroma={'yes' if initial_dp.process_chroma else 'no'}"
            )
        else:
            desc = ""
        self.denoise_desc_label = ctk.CTkLabel(
            preset_row, text=desc,
            anchor="w", justify="left", wraplength=500,
        )
        self.denoise_desc_label.pack(side="left", fill="x", expand=True)

        # stretch at bottom
        ctk.CTkLabel(tab, text="").pack(fill="both", expand=True)

    def _on_denoise_toggle(self):
        enabled = self.denoise_enabled.get()
        state = "normal" if enabled else "disabled"
        self.denoise_preset_combo.configure(state=state)
        text_color = None if enabled else "gray"
        self.denoise_desc_label.configure(text_color=text_color)

    def _on_denoise_preset_change(self, choice):
        dp = DENOISE_PRESETS.get(choice)
        if dp:
            self.denoise_desc_label.configure(
                text=f"threshold={dp.threshold}, nsteps={dp.nsteps}, "
                     f"chroma={'yes' if dp.process_chroma else 'no'}"
            )

    # ── Tab 3: Grain Synthesis ─────────────────────────────────────────────

    def _build_grain_tab(self):
        tab = self.tab_grain

        # Grav1synth availability warning
        if not _GRAV1SYNTH_OK:
            self.grav_warning = ctk.CTkLabel(
                tab,
                text="Grav1synth not installed. Install from https://github.com/rust-av/grav1synth",
                text_color="#e8c547", anchor="w", justify="left", wraplength=700,
            )
            self.grav_warning.pack(fill="x", padx=10, pady=(10, 5))
        else:
            self.grav_warning = None

        # Grain mode
        mode_frame = ctk.CTkFrame(tab, fg_color="transparent")
        mode_frame.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(mode_frame, text="Grain Mode:",
                      font=ctk.CTkFont(size=13, weight="bold"),
                      anchor="w").pack(anchor="w", pady=(0, 5))

        self.grain_mode = ctk.StringVar(value="none")

        self.grain_none_rb = ctk.CTkRadioButton(
            mode_frame, text="None", variable=self.grain_mode, value="none",
        )
        self.grain_none_rb.pack(anchor="w", padx=10, pady=2)

        self.grain_iso_rb = ctk.CTkRadioButton(
            mode_frame, text="ISO", variable=self.grain_mode, value="iso",
        )
        self.grain_iso_rb.pack(anchor="w", padx=10, pady=2)

        self.grain_preset_rb = ctk.CTkRadioButton(
            mode_frame, text="Film Stock", variable=self.grain_mode, value="preset",
        )
        self.grain_preset_rb.pack(anchor="w", padx=10, pady=2)

        # Defer trace_add until after all radio buttons exist
        self.grain_mode.trace_add("write", self._on_grain_mode_change)

        # ISO controls
        self.grain_iso_frame = ctk.CTkFrame(tab, fg_color="transparent")
        self.grain_iso_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(self.grain_iso_frame, text="ISO Value:", anchor="w", width=80).pack(side="left")
        self.grain_iso_var = ctk.StringVar(value="800")
        self.grain_iso_combo = ctk.CTkOptionMenu(
            self.grain_iso_frame,
            values=["100", "200", "400", "800", "1600", "3200", "6400"],
            variable=self.grain_iso_var, width=90,
        )
        self.grain_iso_combo.pack(side="left", padx=(5, 20))

        self.grain_chroma = ctk.BooleanVar(value=False)
        self.grain_chroma_cb = ctk.CTkCheckBox(
            self.grain_iso_frame, text="Apply to chroma planes",
            variable=self.grain_chroma,
        )
        self.grain_chroma_cb.pack(side="left", padx=5)

        # Film stock controls
        self.grain_preset_frame = ctk.CTkFrame(tab, fg_color="transparent")
        self.grain_preset_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(self.grain_preset_frame, text="Film Stock:", anchor="w", width=80).pack(side="left")
        self.grain_preset_name_var = ctk.StringVar(value="16mm")
        self.grain_preset_name_combo = ctk.CTkOptionMenu(
            self.grain_preset_frame, values=GRAIN_FILM_PRESETS,
            variable=self.grain_preset_name_var, width=140,
        )
        self.grain_preset_name_combo.pack(side="left", padx=(5, 10))

        # Apply initial state
        self._on_grain_mode_change()

        # stretch at bottom
        ctk.CTkLabel(tab, text="").pack(fill="both", expand=True)

    def _on_grain_mode_change(self, *_):
        mode = self.grain_mode.get()
        grav_ok = _GRAV1SYNTH_OK

        state_grain = "normal" if grav_ok else "disabled"
        state_iso = "normal" if (grav_ok and mode == "iso") else "disabled"
        state_preset = "normal" if (grav_ok and mode == "preset") else "disabled"

        for rb in [self.grain_none_rb, self.grain_iso_rb, self.grain_preset_rb]:
            rb.configure(state=state_grain)

        self.grain_iso_combo.configure(state=state_iso)
        self.grain_chroma_cb.configure(state=state_iso)
        self.grain_preset_name_combo.configure(state=state_preset)

    # ── Tab 4: Audio & Subtitles ────────────────────────────────────────────

    def _build_audio_tab(self):
        tab = self.tab_audio

        # Audio section
        ctk.CTkLabel(tab, text="Audio Tracks", font=ctk.CTkFont(size=14, weight="bold"),
                      anchor="w").pack(anchor="w", padx=10, pady=(10, 2))

        self.audio_scroll = ctk.CTkScrollableFrame(tab)
        self.audio_scroll.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.audio_empty_label = ctk.CTkLabel(
            self.audio_scroll, text="No audio tracks found.", anchor="w"
        )
        self.audio_empty_label.pack(anchor="w", padx=5, pady=5)

        # Subtitle section
        ctk.CTkLabel(tab, text="Subtitle Tracks", font=ctk.CTkFont(size=14, weight="bold"),
                      anchor="w").pack(anchor="w", padx=10, pady=(10, 2))

        self.sub_scroll = ctk.CTkScrollableFrame(tab)
        self.sub_scroll.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.sub_empty_label = ctk.CTkLabel(
            self.sub_scroll, text="No subtitle tracks found.", anchor="w"
        )
        self.sub_empty_label.pack(anchor="w", padx=5, pady=5)

    # ── Tab 5: Encode ───────────────────────────────────────────────────────

    def _build_encode_tab(self):
        tab = self.tab_encode

        # Centered container
        container = ctk.CTkFrame(tab, fg_color="transparent")
        container.pack(expand=True, fill="both", padx=40, pady=30)

        # ── SVT-AV1 Settings section ──
        svt_section = ctk.CTkFrame(container)
        svt_section.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(svt_section, text="SVT-AV1 Settings",
                      font=ctk.CTkFont(size=13, weight="bold"),
                      anchor="w").pack(anchor="w", padx=10, pady=(8, 2))

        # CRF row
        crf_row = ctk.CTkFrame(svt_section, fg_color="transparent")
        crf_row.pack(fill="x", padx=10, pady=(5, 2))

        ctk.CTkLabel(crf_row, text="CRF (0-63):", anchor="w", width=90).pack(side="left")
        self.crf_entry = ctk.CTkEntry(crf_row, width=70, placeholder_text=str(DEFAULT_CRF_CLEAN))
        self.crf_entry.pack(side="left", padx=(5, 20))

        # SVT Preset row
        svt_row = ctk.CTkFrame(svt_section, fg_color="transparent")
        svt_row.pack(fill="x", padx=10, pady=2)

        ctk.CTkLabel(svt_row, text="SVT Preset:", anchor="w", width=90).pack(side="left")
        self.svt_preset_var = ctk.StringVar(value=str(DEFAULT_SVT_PRESET))
        self.svt_preset_combo = ctk.CTkOptionMenu(
            svt_row,
            values=["0","1","2","3","4","5","6","7","8","9","10","11","12","13"],
            variable=self.svt_preset_var, width=80,
        )
        self.svt_preset_combo.pack(side="left", padx=(5, 20))

        # Tune row
        tune_row = ctk.CTkFrame(svt_section, fg_color="transparent")
        tune_row.pack(fill="x", padx=10, pady=2)

        ctk.CTkLabel(tune_row, text="Tune:", anchor="w", width=90).pack(side="left")
        self.tune_var = ctk.StringVar(value=DEFAULT_TUNE)
        self.tune_vq_rb = ctk.CTkRadioButton(
            tune_row, text="VQ", variable=self.tune_var, value="vq",
        )
        self.tune_vq_rb.pack(side="left", padx=(5, 10))
        self.tune_psnr_rb = ctk.CTkRadioButton(
            tune_row, text="PSNR", variable=self.tune_var, value="psnr",
        )
        self.tune_psnr_rb.pack(side="left", padx=5)

        # Custom params row
        params_row = ctk.CTkFrame(svt_section, fg_color="transparent")
        params_row.pack(fill="x", padx=10, pady=(2, 8))

        ctk.CTkLabel(params_row, text="Extra SVT-AV1 Params\n(colon-delimited):",
                      anchor="w", width=90, justify="left").pack(side="left")
        self.extra_params_entry = ctk.CTkEntry(
            params_row, placeholder_text="enable-qm=0:sharpness=2",
        )
        self.extra_params_entry.pack(side="left", fill="x", expand=True, padx=(5, 10))

        # ── Output Settings section ──
        out_section = ctk.CTkFrame(container)
        out_section.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(out_section, text="Output Settings",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     anchor="w").pack(anchor="w", padx=10, pady=(8, 2))

        # Downscale checkbox
        self.scale_1080p_var = ctk.BooleanVar(value=False)
        self.scale_1080p_cb = ctk.CTkCheckBox(
            out_section, text="Downscale to 1080p (1920 width)",
            variable=self.scale_1080p_var,
        )
        self.scale_1080p_cb.pack(anchor="w", padx=10, pady=(5, 2))

        # Colorspace dropdown
        cs_row = ctk.CTkFrame(out_section, fg_color="transparent")
        cs_row.pack(fill="x", padx=10, pady=(2, 8))
        ctk.CTkLabel(cs_row, text="Colorspace:", anchor="w").pack(side="left", padx=(0, 5))
        self.colorspace_var = ctk.StringVar(value="source")
        self.colorspace_combo = ctk.CTkOptionMenu(
            cs_row, values=["source", "bt709", "bt2020", "bt601"],
            variable=self.colorspace_var, width=100,
        )
        self.colorspace_combo.pack(side="left")

        # ── Progress section ──
        # Main progress bar
        ctk.CTkLabel(container, text="Video Progress",
                      font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", pady=(10, 2))
        self.progress_bar = ctk.CTkProgressBar(container, orientation="horizontal",
                                                mode="determinate", width=600)
        self.progress_bar.pack(fill="x", pady=(0, 10))
        self.progress_bar.set(0)

        # Sub progress bar (audio / mux)
        ctk.CTkLabel(container, text="Audio / Mux",
                      font=ctk.CTkFont(size=12)).pack(anchor="w", pady=(0, 2))
        self.sub_progress = ctk.CTkProgressBar(container, orientation="horizontal",
                                                mode="indeterminate", width=600)
        self.sub_progress.pack(fill="x", pady=(0, 10))

        # Status labels
        self.status_label = ctk.CTkLabel(container, text="Ready", anchor="w", justify="left",
                                          font=ctk.CTkFont(size=13))
        self.status_label.pack(fill="x", pady=(5, 2))

        self.stats_label = ctk.CTkLabel(container, text="", anchor="w", justify="left",
                                         font=ctk.CTkFont(size=11, family="monospace"))
        self.stats_label.pack(fill="x", pady=(0, 15))

        # ── Preview clip settings ──
        preview_row = ctk.CTkFrame(container, fg_color="transparent")
        preview_row.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(preview_row, text="Preview:",
                      font=ctk.CTkFont(size=13, weight="bold"),
                      anchor="w").pack(side="left", padx=(0, 10))

        ctk.CTkLabel(preview_row, text="Start (s):", anchor="w").pack(side="left", padx=(0, 2))
        self.preview_start_entry = ctk.CTkEntry(preview_row, width=60, placeholder_text="300")
        self.preview_start_entry.insert(0, "300")
        self.preview_start_entry.pack(side="left", padx=(0, 10))

        ctk.CTkLabel(preview_row, text="Duration (s):", anchor="w").pack(side="left", padx=(0, 2))
        self.preview_dur_entry = ctk.CTkEntry(preview_row, width=50, placeholder_text="6")
        self.preview_dur_entry.insert(0, "6")
        self.preview_dur_entry.pack(side="left")

        # Buttons
        btn_row = ctk.CTkFrame(container, fg_color="transparent")
        btn_row.pack(pady=(5, 0))

        self.preview_btn = ctk.CTkButton(btn_row, text="Preview Clip", width=120,
                                          command=self._start_preview)
        self.preview_btn.pack(side="left", padx=5)

        self.start_btn = ctk.CTkButton(btn_row, text="Start Encode", width=160,
                                        fg_color="#2d7a3a", hover_color="#23612e",
                                        command=self.start_encode)
        self.start_btn.pack(side="left", padx=5)

        self.cancel_btn = ctk.CTkButton(btn_row, text="Cancel", width=120,
                                         fg_color="#a03030", hover_color="#802525",
                                         state="disabled", command=self.cancel_encode)
        self.cancel_btn.pack(side="left", padx=5)

        # Results frame (hidden until encode done)
        self.result_frame = ctk.CTkFrame(container, fg_color="transparent")

    # ── File dialogs ────────────────────────────────────────────────────────

    def _browse_source(self):
        path = fd.askopenfilename(
            title="Select Source Video",
            filetypes=[("Video files", "*.mkv *.mp4 *.avi *.mov *.m2ts"),
                       ("All files", "*.*")]
        )
        if path:
            self.source_entry.delete(0, "end")
            self.source_entry.insert(0, path)
            self.load_file()

    def _browse_output(self):
        default_name = "output.mkv"
        if self.info:
            default_name = f"{self.info['basename']}_denoised.mkv"
        path = fd.asksaveasfilename(
            title="Save Output As",
            defaultextension=".mkv",
            filetypes=[("Matroska video", "*.mkv")],
            initialfile=default_name,
        )
        if path:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, path)

    # ── File loading ────────────────────────────────────────────────────────

    def load_file(self):
        path = self.source_entry.get().strip()
        if not path or not os.path.exists(path):
            return

        try:
            self.info = probe_rich(path)
        except Exception as e:
            mb.showerror("Probe Error", f"Failed to probe file:\n{e}")
            return

        self._update_media_info()

        # Auto-fill output
        out_dir = os.path.dirname(os.path.abspath(path))
        basename = self.info["basename"]
        default_output = os.path.join(out_dir, f"{basename}_denoised.mkv")
        self.output_entry.delete(0, "end")
        self.output_entry.insert(0, default_output)

        # Populate tracks
        self._populate_audio_tracks()
        self._populate_subtitle_tracks()

    def _update_media_info(self):
        info = self.info
        w, h = info["width"], info["height"]
        fps = info["fps"]
        codec = info["codec"]
        duration = info["duration"]
        bit_depth = info["bit_depth"]

        line1 = f"Codec: {codec.upper()}    Resolution: {w}x{h}    FPS: {fps:.3f}"
        self.media_line1.configure(text=line1)

        dur_min = duration / 60
        if bit_depth != 10:
            bd_text = f"{bit_depth}-bit (to 10-bit output)"
        else:
            bd_text = "10-bit (10-bit native)"
        line2 = f"Duration: {dur_min:.1f}m    Bit Depth: {bd_text}"
        self.media_line2.configure(text=line2)

        audio_count = len(info["audio_tracks"])
        sub_count = len(info["subtitle_tracks"])
        line3 = f"Audio: {audio_count} tracks    Subtitles: {sub_count} tracks"
        self.media_line3.configure(text=line3)

        # Grav1synth availability
        grav_avail = info.get("grav1synth_available", False)
        grav_text = "Grav1synth available" if grav_avail else "Grav1synth not installed"
        self.media_grav_label.configure(
            text=grav_text,
            text_color=("#1a7a1a", "#4caf50") if grav_avail else "#e8c547",
        )

        # Warnings
        video_count = info.get("video_count", 1)
        if video_count > 1:
            self.media_warn1.configure(
                text=f"  {video_count} video tracks (only #0 processed)",
                text_color="#e8c547"
            )
            self.media_warn1.pack(fill="x", padx=5, pady=(0, 0))
        else:
            self.media_warn1.pack_forget()

        if bit_depth == 12:
            self.media_warn2.configure(
                text="  12-bit source to 10-bit output",
                text_color="#e8c547"
            )
            self.media_warn2.pack(fill="x", padx=5, pady=(0, 0))
        else:
            self.media_warn2.pack_forget()

    # ── Audio track UI ──────────────────────────────────────────────────────

    def _populate_audio_tracks(self):
        # Clear existing
        for w in self.audio_track_widgets:
            w["frame"].destroy()
        self.audio_track_widgets.clear()
        self.audio_empty_label.pack_forget()

        tracks = self.info.get("audio_tracks", [])
        if not tracks:
            self.audio_empty_label.pack(anchor="w", padx=5, pady=5)
            return

        for at in tracks:
            self._add_audio_track_row(at)

    def _add_audio_track_row(self, track):
        row = ctk.CTkFrame(self.audio_scroll, fg_color=("gray90", "gray20"))
        row.pack(fill="x", padx=5, pady=2)

        cb_var = ctk.StringVar(value="on")
        cb = ctk.CTkCheckBox(row, text="", variable=cb_var, onvalue="on", offvalue="off",
                              width=20)

        # Info label
        br_display = f"{track.bitrate // 1000}k" if track.bitrate else "VBR"
        flags = ""
        if track.default:
            flags = "  (default)"
        title_str = f"  - {track.title}" if track.title else ""
        info_text = f"#{track.index}  {track.lang}  {track.codec}  {track.channels}ch  {br_display}{title_str}{flags}"

        info_label = ctk.CTkLabel(row, text=info_text, anchor="w", width=280)
        if track.default:
            info_label.configure(text_color=("#1a7a1a", "#4caf50"))

        action_menu = ctk.CTkOptionMenu(
            row, values=["Passthrough", "Encode to Opus"], width=160
        )
        action_menu.set("Passthrough")

        bitrate = get_opus_bitrate(track.channels)
        bitrate_label = ctk.CTkLabel(row, text=f"{bitrate} kbps", width=60, anchor="e",
                                      font=ctk.CTkFont(size=12))

        name_entry = ctk.CTkEntry(row, placeholder_text="Custom name...", width=150)

        idx = len(self.audio_track_widgets)

        def on_toggle(i=idx):
            selected = self.audio_track_widgets[i]["checkbox_var"].get() == "on"
            widgets = self.audio_track_widgets[i]
            if selected:
                widgets["action_menu"].configure(state="normal")
                widgets["name_entry"].configure(state="normal")
            else:
                widgets["action_menu"].configure(state="disabled")
                widgets["name_entry"].configure(state="disabled")

        def on_action_change(value, i=idx):
            widgets = self.audio_track_widgets[i]
            if value == "Encode to Opus":
                widgets["bitrate_label"].pack(side="left", padx=2)
            else:
                widgets["bitrate_label"].pack_forget()

        cb.configure(command=on_toggle)
        action_menu.configure(command=on_action_change)

        # Pack row
        cb.pack(side="left", padx=(5, 2))
        info_label.pack(side="left", padx=(2, 5))
        action_menu.pack(side="left", padx=2)
        # Bitrate starts hidden (passthrough is default)
        name_entry.pack(side="left", padx=(5, 5))

        self.audio_track_widgets.append({
            "track": track,
            "frame": row,
            "checkbox": cb,
            "checkbox_var": cb_var,
            "info_label": info_label,
            "action_menu": action_menu,
            "bitrate_label": bitrate_label,
            "name_entry": name_entry,
        })

    # ── Subtitle track UI ───────────────────────────────────────────────────

    def _populate_subtitle_tracks(self):
        for w in self.subtitle_track_widgets:
            w["frame"].destroy()
        self.subtitle_track_widgets.clear()
        self.sub_empty_label.pack_forget()

        tracks = self.info.get("subtitle_tracks", [])
        if not tracks:
            self.sub_empty_label.pack(anchor="w", padx=5, pady=5)
            return

        for st in tracks:
            self._add_subtitle_track_row(st)

    def _add_subtitle_track_row(self, track):
        row = ctk.CTkFrame(self.sub_scroll, fg_color=("gray90", "gray20"))
        row.pack(fill="x", padx=5, pady=2)

        cb_var = ctk.StringVar(value="on")
        cb = ctk.CTkCheckBox(row, text="", variable=cb_var, onvalue="on", offvalue="off",
                              width=20)

        info_text = f"#{track.index}  {track.lang}  {track.codec}"
        info_label = ctk.CTkLabel(row, text=info_text, anchor="w")

        cb.pack(side="left", padx=(5, 2))
        info_label.pack(side="left", padx=(2, 2))

        # Colored flag labels
        if track.default:
            flag_lbl = ctk.CTkLabel(row, text="(default)", anchor="w",
                                     text_color=("#1a7a1a", "#4caf50"))
            flag_lbl.pack(side="left", padx=(2, 2))

        if track.forced:
            flag_lbl = ctk.CTkLabel(row, text="(forced)", anchor="w",
                                     text_color=("#b0670a", "#e8a020"))
            flag_lbl.pack(side="left", padx=(2, 2))

        self.subtitle_track_widgets.append({
            "track": track,
            "frame": row,
            "checkbox": cb,
            "checkbox_var": cb_var,
            "info_label": info_label,
        })

    # ── Build track lists from UI ───────────────────────────────────────────

    def build_audio_tracks_from_ui(self):
        tracks = []
        for w in self.audio_track_widgets:
            track = w["track"]
            track.selected = (w["checkbox_var"].get() == "on")
            action_text = w["action_menu"].get()
            if action_text == "Encode to Opus":
                track.action = AudioAction.ENCODE
                track.encode_bitrate = get_opus_bitrate(track.channels)
            else:
                track.action = AudioAction.PASSTHROUGH
            track.custom_title = w["name_entry"].get()
            tracks.append(track)
        return tracks

    def build_subtitle_tracks_from_ui(self):
        tracks = []
        for w in self.subtitle_track_widgets:
            track = w["track"]
            track.selected = (w["checkbox_var"].get() == "on")
            tracks.append(track)
        return tracks

    # ── Encode ──────────────────────────────────────────────────────────────

    def _gather_encode_settings(self):
        """Gather denoise preset, grain preset, and encode params from UI."""
        # Denoise preset
        if self.denoise_enabled.get():
            dp = DENOISE_PRESETS.get(self.denoise_preset_var.get())
        else:
            dp = None

        # Grain preset
        gp = None
        mode = self.grain_mode.get()
        if mode == "iso":
            gp = GrainPreset(
                source=GrainSource.ISO,
                iso_value=int(self.grain_iso_var.get()),
                chroma=self.grain_chroma.get(),
            )
        elif mode == "preset":
            gp = GrainPreset(
                source=GrainSource.PRESET,
                preset_name=self.grain_preset_name_var.get(),
            )

        # Encode params
        try:
            crf = int(self.crf_entry.get().strip()) if self.crf_entry.get().strip() else DEFAULT_CRF_CLEAN
        except ValueError:
            crf = DEFAULT_CRF_CLEAN

        try:
            svt = int(self.svt_preset_var.get())
        except ValueError:
            svt = DEFAULT_SVT_PRESET

        tune = self.tune_var.get()
        extra_params = self.extra_params_entry.get().strip()

        # Output settings
        scale_1080p = self.scale_1080p_var.get()
        colorspace = self.colorspace_var.get() if self.colorspace_var.get() != "source" else None

        return dp, gp, crf, svt, tune, extra_params, scale_1080p, colorspace

    def start_encode(self):
        if self.encoding:
            return

        source = self.source_entry.get().strip()
        if not source or not os.path.exists(source):
            self.status_label.configure(text="No source file selected", text_color="red")
            return

        if self.info is None:
            self.status_label.configure(text="Please select a valid source file first",
                                        text_color="red")
            return

        dp, gp, crf, svt, tune, extra_params, scale_1080p, colorspace = self._gather_encode_settings()

        audio_tracks = self.build_audio_tracks_from_ui()
        sub_tracks = self.build_subtitle_tracks_from_ui()

        output = self.output_entry.get().strip()
        if not output:
            out_dir = os.path.dirname(os.path.abspath(source))
            basename = os.path.splitext(os.path.basename(source))[0]
            output = os.path.join(out_dir, f"{basename}_denoised.mkv")

        # Build output dir
        out_parent = os.path.dirname(os.path.abspath(output))
        os.makedirs(out_parent, exist_ok=True)

        # Compute target dimensions if scaling
        target_w, target_h = None, None
        if scale_1080p:
            target_w = 1920
            target_h = round(target_w * self.info["height"] / self.info["width"] / 2) * 2

        # Generate VPY
        vpy = generate_vpy(source, dp, target_width=target_w, target_height=target_h,
                           output_colorspace=colorspace)
        vpy_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".vpy", prefix="fden_", delete=False
        )
        try:
            vpy_file.write(vpy)
            vpy_path = vpy_file.name
        finally:
            vpy_file.close()

        # Disable UI
        self.start_btn.configure(state="disabled")
        self.preview_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.status_label.configure(text="Starting encode...", text_color="white")
        self.stats_label.configure(text="")
        self.progress_bar.set(0)
        self.sub_progress.set(0)

        self.encoding = True
        self.cancel_event = threading.Event()

        self._run_encode_task(
            source=source, vpy_path=vpy_path, output=output,
            dp=dp, gp=gp, crf=crf, svt_preset=svt, tune=tune,
            extra_params=extra_params,
            total_frames=self.info["frames"],
            audio_tracks=audio_tracks, sub_tracks=sub_tracks,
            pass_audio=bool(audio_tracks), pass_subs=bool(sub_tracks),
            pass_chapters=True, bit_depth=10,
            colorspace=colorspace,
        )

    def _start_preview(self):
        if self.encoding:
            return

        source = self.source_entry.get().strip()
        if not source or not os.path.exists(source):
            self.status_label.configure(text="No source file selected", text_color="red")
            return

        if self.info is None:
            self.status_label.configure(text="Please select a valid source file first",
                                        text_color="red")
            return

        dp, gp, crf, svt, tune, extra_params, scale_1080p, colorspace = self._gather_encode_settings()

        # Read preview start and duration from UI fields
        try:
            start_secs = float(self.preview_start_entry.get().strip())
        except ValueError:
            start_secs = 300.0
        try:
            dur_secs = float(self.preview_dur_entry.get().strip())
        except ValueError:
            dur_secs = 6.0

        fps = self.info["fps"]
        start_frame = int(start_secs * fps)
        num_frames = max(1, int(dur_secs * fps))

        # Compute target dimensions if scaling
        target_w, target_h = None, None
        if scale_1080p:
            target_w = 1920
            target_h = round(target_w * self.info["height"] / self.info["width"] / 2) * 2

        # Generate preview VPY
        vpy = generate_vpy(source, dp, start_frame=start_frame, num_frames=num_frames,
                           target_width=target_w, target_height=target_h,
                           output_colorspace=colorspace)
        vpy_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".vpy", prefix="fden_prev_", delete=False
        )
        try:
            vpy_file.write(vpy)
            vpy_path = vpy_file.name
        finally:
            vpy_file.close()

        # Default preview output path in /tmp
        preview_id = uuid.uuid4().hex[:8]
        output = os.path.join(tempfile.gettempdir(), f"fden_preview_{preview_id}.mkv")

        # Disable UI
        self.start_btn.configure(state="disabled")
        self.preview_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.status_label.configure(text="Starting preview...", text_color="white")
        self.stats_label.configure(text="")
        self.progress_bar.set(0)
        self.sub_progress.set(0)

        self.encoding = True
        self.cancel_event = threading.Event()

        self._run_encode_task(
            source=source, vpy_path=vpy_path, output=output,
            dp=dp, gp=gp, crf=crf, svt_preset=svt, tune=tune,
            extra_params=extra_params,
            total_frames=num_frames,
            audio_tracks=[], sub_tracks=[],
            pass_audio=False, pass_subs=False, pass_chapters=False,
            bit_depth=10,
            colorspace=colorspace,
            preview_audio_start=start_secs, preview_audio_dur=dur_secs,
        )

    def _run_encode_task(self, source, vpy_path, output,
                         dp, gp, crf, svt_preset, tune, extra_params,
                         total_frames, audio_tracks, sub_tracks,
                         pass_audio, pass_subs, pass_chapters, bit_depth,
                         colorspace: str | None = None,
                         preview_audio_start: float | None = None,
                         preview_audio_dur: float | None = None):
        """Run encoding in a background thread."""

        # Progress callback (runs in encode thread)
        def on_progress(phase, frame, total, fps_val, eta, status):
            if self.cancel_event.is_set():
                return False

            if phase == "video":
                pct = frame / total if total > 0 else 0
                f, t, fps_v, e, s = frame, total, fps_val, eta, status
                self.root.after(0, lambda: self.sub_progress.stop())
                self.root.after(0, lambda: self.sub_progress.set(0))
                self.root.after(0, lambda p=pct: self.progress_bar.set(p))
                if total:
                    self.root.after(0, lambda p=pct, s=s: self.status_label.configure(
                        text=f"{s} - {p*100:.0f}%"
                    ))
                else:
                    self.root.after(0, lambda s=s: self.status_label.configure(text=s))
                self.root.after(0, lambda f=f, t=t, fps=fps_v, e=e: self.stats_label.configure(
                    text=f"Frame {f}/{t}  {fps:.1f}fps  ETA {e:.0f}s"
                ))
            elif phase == "mux":
                s = status
                self.root.after(0, lambda s=s: self.status_label.configure(text=s))
            elif phase == "grain":
                s = status
                self.root.after(0, lambda s=s: self.status_label.configure(text=s))
            elif phase == "audio":
                self.root.after(0, lambda: self.sub_progress.start())
                self.root.after(0, lambda s=status: self.status_label.configure(text=s))

            return not self.cancel_event.is_set()

        # Build EncodeJob with new signature
        job = EncodeJob(
            source=source,
            vpy_path=vpy_path,
            output=output,
            svt_preset=svt_preset,
            crf=crf,
            tune=tune,
            extra_params=extra_params,
            grain_preset=gp,
            total_frames=total_frames,
            bit_depth=bit_depth,
            fps=self.info["fps"],
            video_start_time=self.info.get("video_start_time", 0.0),
            audio_start_time=self.info.get("audio_start_time", 0.0),
            color_primaries=COLORSPACE_PRESETS[colorspace]["ff_primaries"] if colorspace else self.info.get("color_primaries", "bt709"),
            color_transfer=COLORSPACE_PRESETS[colorspace]["ff_trc"] if colorspace else self.info.get("color_transfer", "bt709"),
            color_space=COLORSPACE_PRESETS[colorspace]["ff_space"] if colorspace else self.info.get("color_space", "bt709"),
            color_range=self.info.get("color_range", "tv"),
            chroma_location=self.info.get("chroma_location", "left"),
            pass_audio=pass_audio,
            pass_subs=pass_subs,
            pass_chapters=pass_chapters,
            audio_tracks=audio_tracks,
            sub_tracks=sub_tracks,
            progress_callback=on_progress,
        )

        # Start sub_progress (for audio phase) on the main thread
        self.sub_progress.start()

        def encode_thread():
            try:
                # Phase 2: Video encode + mux
                result = run_encode(job, quiet=False, cancel_event=self.cancel_event)

                # Phase 3: Preview audio extraction + mux (if applicable)
                if preview_audio_start is not None and result.success:
                    self._mux_preview_audio(
                        result, source, preview_audio_start, preview_audio_dur,
                    )

                # Phase 4: Done
                self.root.after(0, lambda r=result: self.on_encode_done(r, source))
            except Exception as e:
                # Capture message immediately — avoid lambda closure issues
                err_msg = f"{type(e).__name__}: {e}"
                print(f"[encode error] {err_msg}", file=sys.stderr)
                self.root.after(0, lambda m=err_msg: self.on_encode_error(m))
            finally:
                # Cleanup VPY
                if os.path.exists(vpy_path):
                    try:
                        os.unlink(vpy_path)
                    except OSError:
                        pass

        threading.Thread(target=encode_thread, daemon=True).start()

    def cancel_encode(self):
        self.cancel_event.set()
        self.status_label.configure(text="Cancelling...", text_color="#e8c547")

    def on_encode_done(self, job, source_path):
        self.cancel_btn.configure(state="disabled")
        self.start_btn.configure(state="normal")
        self.preview_btn.configure(state="normal")
        self.encoding = False
        self.sub_progress.stop()

        if job.success:
            self.progress_bar.set(1.0)
            self.status_label.configure(text="Encode complete!", text_color="green")

            source_size = os.path.getsize(source_path)
            lines = [
                f"Output: {job.output}",
                (f"Size: {_format_size(job.output_size)}  "
                 f"FPS: {job.avg_fps:.1f}  "
                 f"Duration: {job.elapsed:.0f}s"),
            ]
            if source_size > 0 and job.output_size > 0:
                ratio = source_size / job.output_size
                lines.append(f"Compression: {ratio:.1f}:1")

            self.stats_label.configure(text="\n".join(lines))

            # Show result frame
            self.result_frame.pack_forget()
            self.result_frame.pack(fill="x", pady=(15, 0))
            for w in self.result_frame.winfo_children():
                w.destroy()
            ctk.CTkLabel(
                self.result_frame,
                text=f"Output: {job.output}",
                anchor="w", justify="left",
                text_color="green",
                font=ctk.CTkFont(size=12),
            ).pack(fill="x")
        else:
            self.status_label.configure(text="Encode failed or cancelled", text_color="red")

    def on_encode_error(self, msg):
        self.cancel_btn.configure(state="disabled")
        self.start_btn.configure(state="normal")
        self.preview_btn.configure(state="normal")
        self.encoding = False
        self.sub_progress.stop()
        self.status_label.configure(text=f"Error: {msg}", text_color="red")
        self.stats_label.configure(text="")

    # ── Preview audio post-processing ───────────────────────────────────────

    def _mux_preview_audio(self, job, source, start_secs, dur_secs):
        """Extract audio from source for preview time range and mux into output.

        Falls back silently to video-only preview on any failure.
        """
        preview_id = uuid.uuid4().hex[:8]
        audio_tmp = os.path.join(
            tempfile.gettempdir(), f"fden_preva_{preview_id}.mka"
        )

        try:
            # Extract audio from source for the time range
            audio_cmd = [
                _FFMPEG, "-y",
                "-ss", str(start_secs), "-t", str(dur_secs),
                "-i", source,
                "-map", "0:a:0",
                "-c:a", "copy",
                audio_tmp,
            ]
            subprocess.run(audio_cmd, check=True, capture_output=True)

            if not os.path.isfile(audio_tmp) or os.path.getsize(audio_tmp) == 0:
                raise RuntimeError("Audio extraction produced empty output")

            # Mux video + audio with ffmpeg
            final_output = os.path.join(
                tempfile.gettempdir(), f"fden_preview_{preview_id}_av.mkv"
            )
            mux_cmd = [
                _FFMPEG, "-y",
                "-i", job.output,
                "-i", audio_tmp,
                "-c", "copy",
                final_output,
            ]
            subprocess.run(mux_cmd, check=True, capture_output=True)

            # Cleanup old video-only output
            old_output = job.output
            try:
                os.unlink(old_output)
            except OSError:
                pass

            # Update job to point to new file
            job.output = final_output
            job.output_size = os.path.getsize(final_output)

        except Exception as e:
            print(f"[preview audio] Failed to mux audio (using video-only): {e}",
                  file=sys.stderr)
        finally:
            # Cleanup temp audio
            if os.path.isfile(audio_tmp):
                try:
                    os.unlink(audio_tmp)
                except OSError:
                    pass

    # ── Window close ────────────────────────────────────────────────────────

    def on_closing(self):
        if self.encoding:
            if mb.askyesno("Confirm", "Encoding in progress. Cancel and quit?"):
                self.cancel_event.set()
                # Poll for thread to finish (up to 3 seconds) so subprocesses
                # are properly cleaned up before destroying the window
                self.root.after(100, self._check_thread_and_close)
        else:
            self.root.destroy()

    def _check_thread_and_close(self, attempts=0):
        if not self.encoding or attempts > 30:
            self.root.destroy()
        else:
            self.root.after(100, lambda: self._check_thread_and_close(attempts + 1))

    # ── Run ─────────────────────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()


def main():
    app = FilmDenoiserGUI()
    app.run()


if __name__ == "__main__":
    main()
