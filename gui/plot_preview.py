"""In-app plot preview using static PNG images (Plotly/kaleido)."""

from __future__ import annotations

import io
import os
import tempfile
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

import customtkinter as ctk
import plotly.io as pio
from PIL import Image


@dataclass
class PlotRecord:
    label: str
    html_path: str
    png_path: str
    summary: str = ""


class PlotViewer(ctk.CTkFrame):
    """Tabbed viewer with scrollable PNG previews and export/browser actions."""

    PREVIEW_WIDTH = 960
    PREVIEW_HEIGHT = 560

    def __init__(self, master: Any, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        toolbar = ctk.CTkFrame(self, fg_color="transparent")
        toolbar.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        toolbar.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(toolbar, text="No plots yet", anchor="w")
        self.status_label.grid(row=0, column=0, sticky="w")

        self.open_btn = ctk.CTkButton(toolbar, text="Open Interactive", width=140, command=self._open_current)
        self.open_btn.grid(row=0, column=1, padx=(8, 0))
        self.open_btn.configure(state="disabled")

        self.export_btn = ctk.CTkButton(toolbar, text="Export HTML", width=120, command=self._export_current)
        self.export_btn.grid(row=0, column=2, padx=(8, 0))
        self.export_btn.configure(state="disabled")

        self.export_png_btn = ctk.CTkButton(toolbar, text="Export PNG", width=110, command=self._export_png_current)
        self.export_png_btn.grid(row=0, column=3, padx=(8, 0))
        self.export_png_btn.configure(state="disabled")

        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        self._records: list[PlotRecord] = []
        self._ctk_images: list[ctk.CTkImage] = []
        self._out_dir = Path(tempfile.gettempdir()) / "md_gbsa_analysis_plots"
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def clear(self) -> None:
        for tab_name in self.tabview._tab_dict.copy():
            self.tabview.delete(tab_name)
        self._records.clear()
        self._ctk_images.clear()
        self.status_label.configure(text="No plots yet")
        self.open_btn.configure(state="disabled")
        self.export_btn.configure(state="disabled")
        self.export_png_btn.configure(state="disabled")

    def add_plot(self, label: str, figure: Any, summary: str = "") -> None:
        safe_label = label[:40]
        tab = self.tabview.add(safe_label)
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in safe_label)
        html_path = str(self._out_dir / f"{safe_name}.html")
        png_path = str(self._out_dir / f"{safe_name}.png")

        try:
            self._write_html(figure, html_path)
            self._write_png(figure, png_path)
        except Exception as exc:
            self.tabview.delete(safe_label)
            raise RuntimeError(
                f"Could not render preview for '{label}'. "
                f"Install kaleido (pip install kaleido) and retry.\nDetails: {exc}"
            ) from exc

        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.grid(row=0, column=0, sticky="nsew")
        scroll.grid_columnconfigure(0, weight=1)

        preview = self._build_preview_image(png_path)
        img_label = ctk.CTkLabel(scroll, text="", image=preview)
        img_label.grid(row=0, column=0, padx=8, pady=8)
        self._ctk_images.append(preview)

        if summary:
            summary_box = ctk.CTkTextbox(scroll, height=90, wrap="word")
            summary_box.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
            summary_box.insert("1.0", summary)
            summary_box.configure(state="disabled")

        self._records.append(PlotRecord(label=label, html_path=html_path, png_path=png_path, summary=summary))
        self.tabview.set(safe_label)
        self.status_label.configure(text=f"{len(self._records)} plot(s) ready — scroll to view full chart")
        self.open_btn.configure(state="normal")
        self.export_btn.configure(state="normal")
        self.export_png_btn.configure(state="normal")

    def _write_html(self, figure: Any, path: str) -> None:
        pio.write_html(
            figure,
            file=path,
            auto_open=False,
            include_plotlyjs=True,
            config={"responsive": True, "displayModeBar": True},
        )

    def _write_png(self, figure: Any, path: str) -> None:
        pio.write_image(
            figure,
            path,
            width=self.PREVIEW_WIDTH,
            height=self.PREVIEW_HEIGHT,
            scale=2,
        )

    def _build_preview_image(self, png_path: str) -> ctk.CTkImage:
        pil_img = Image.open(png_path)
        return ctk.CTkImage(
            light_image=pil_img,
            dark_image=pil_img,
            size=(self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT),
        )

    def _current_record(self) -> PlotRecord | None:
        if not self._records:
            return None
        current = self.tabview.get()
        for record in self._records:
            if record.label[:40] == current:
                return record
        return self._records[-1]

    def _open_current(self) -> None:
        record = self._current_record()
        if record:
            webbrowser.open(f"file:///{record.html_path.replace(os.sep, '/')}")

    def _export_current(self) -> None:
        record = self._current_record()
        if not record:
            return
        dest = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML files", "*.html")],
            initialfile=Path(record.html_path).name,
        )
        if dest:
            Path(dest).write_text(Path(record.html_path).read_text(encoding="utf-8"), encoding="utf-8")
            messagebox.showinfo("Export", f"Saved to\n{dest}")

    def _export_png_current(self) -> None:
        record = self._current_record()
        if not record:
            return
        dest = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG images", "*.png")],
            initialfile=Path(record.png_path).name,
        )
        if dest:
            Path(dest).write_bytes(Path(record.png_path).read_bytes())
            messagebox.showinfo("Export", f"Saved to\n{dest}")
