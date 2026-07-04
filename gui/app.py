"""Modern responsive GUI for MD trajectory and GBSA analysis."""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Any

import customtkinter as ctk

from analysis.gbsa_analyzer import GBSA_PLOT_TYPES, GBSADataset, run_gbsa_analysis
from analysis.md_analyzer import MD_PLOT_TYPES, TrajectoryDataset, run_md_analysis
from gui.plot_preview import PlotViewer

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

APP_TITLE = "MD & GBSA Analysis Studio"
MIN_WIDTH = 1100
MIN_HEIGHT = 720


class DatasetRow(ctk.CTkFrame):
    """One trajectory input row."""

    def __init__(self, master: Any, on_remove: Any, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self.on_remove = on_remove
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(3, weight=1)
        self.grid_columnconfigure(5, weight=1)

        ctk.CTkLabel(self, text="Label").grid(row=0, column=0, padx=(8, 4), pady=6, sticky="w")
        self.name_var = ctk.StringVar(value="Dataset 1")
        ctk.CTkEntry(self, textvariable=self.name_var, width=200).grid(row=0, column=1, padx=2, pady=6, sticky="ew")

        ctk.CTkLabel(self, text="Topology").grid(row=0, column=2, padx=(12, 4), pady=6, sticky="w")
        self.topo_var = ctk.StringVar()
        ctk.CTkEntry(self, textvariable=self.topo_var, width=100).grid(row=0, column=3, padx=4, pady=6, sticky="ew")
        ctk.CTkButton(self, text="Browse", width=70, command=self._browse_topo).grid(row=0, column=4, padx=4, pady=6)

        ctk.CTkLabel(self, text="Trajectory").grid(row=1, column=0, padx=(8, 4), pady=6, sticky="w")
        self.traj_var = ctk.StringVar()
        ctk.CTkEntry(self, textvariable=self.traj_var).grid(row=1, column=1, columnspan=3, padx=4, pady=6, sticky="ew")
        ctk.CTkButton(self, text="Browse", width=70, command=self._browse_traj).grid(row=1, column=4, padx=4, pady=6)

        ctk.CTkLabel(self, text="SASA (.xvg, optional)").grid(row=2, column=0, padx=(8, 4), pady=6, sticky="w")
        self.sasa_var = ctk.StringVar()
        ctk.CTkEntry(self, textvariable=self.sasa_var).grid(row=2, column=1, columnspan=3, padx=4, pady=6, sticky="ew")
        ctk.CTkButton(self, text="Browse", width=70, command=self._browse_sasa).grid(row=2, column=4, padx=4, pady=6)

        ctk.CTkButton(self, text="✕", width=36, fg_color="#8B0000", hover_color="#A52A2A", command=self._remove).grid(
            row=0, column=5, rowspan=3, padx=8, pady=6, sticky="ns"
        )

    def _browse_topo(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Structure files", "*.gro *.pdb *.psf *.tpr *.prmtop"), ("All files", "*.*")]
        )
        if path:
            self.topo_var.set(path)

    def _browse_traj(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Trajectory files", "*.xtc *.dcd *.trr *.nc"), ("All files", "*.*")]
        )
        if path:
            self.traj_var.set(path)

    def _browse_sasa(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("XVG files", "*.xvg"), ("All files", "*.*")])
        if path:
            self.sasa_var.set(path)

    def _remove(self) -> None:
        self.on_remove(self)

    def to_dataset(self) -> TrajectoryDataset | None:
        name = self.name_var.get().strip()
        topo = self.topo_var.get().strip()
        traj = self.traj_var.get().strip()
        if not name or not topo or not traj:
            return None
        sasa = self.sasa_var.get().strip() or None
        return TrajectoryDataset(name=name, topology=topo, trajectory=traj, sasa_file=sasa)


class GBSARow(ctk.CTkFrame):
    """One GBSA CSV input row."""

    def __init__(self, master: Any, on_remove: Any, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self.on_remove = on_remove
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(self, text="Label").grid(row=0, column=0, padx=(8, 4), pady=6, sticky="w")
        self.name_var = ctk.StringVar(value="Run 1")
        ctk.CTkEntry(self, textvariable=self.name_var, width=120).grid(row=0, column=1, padx=4, pady=6, sticky="ew")

        ctk.CTkLabel(self, text="GBSA CSV").grid(row=0, column=2, padx=(12, 4), pady=6, sticky="w")
        self.file_var = ctk.StringVar()
        ctk.CTkEntry(self, textvariable=self.file_var).grid(row=0, column=3, padx=4, pady=6, sticky="ew")
        ctk.CTkButton(self, text="Browse", width=70, command=self._browse).grid(row=0, column=4, padx=4, pady=6)
        ctk.CTkButton(self, text="✕", width=36, fg_color="#8B0000", hover_color="#A52A2A", command=self._remove).grid(
            row=0, column=5, padx=8, pady=6
        )

    def _browse(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.file_var.set(path)

    def _remove(self) -> None:
        self.on_remove(self)

    def to_dataset(self) -> GBSADataset | None:
        name = self.name_var.get().strip()
        filepath = self.file_var.get().strip()
        if not name or not filepath:
            return None
        return GBSADataset(name=name, filepath=filepath)


class AnalysisApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1280x800")
        self.minsize(MIN_WIDTH, MIN_HEIGHT)

        self.grid_columnconfigure(0, weight=0, minsize=380)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._md_rows: list[DatasetRow] = []
        self._gbsa_rows: list[GBSARow] = []
        self._worker: threading.Thread | None = None

        self._build_sidebar()
        self._build_main()
        self._add_md_row()
        self._add_gbsa_row()

        self.bind("<Configure>", self._on_resize)

    def _build_sidebar(self) -> None:
        sidebar = ctk.CTkScrollableFrame(self, width=380, label_text="Configuration")
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        sidebar.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            sidebar,
            text="MD & GBSA\nAnalysis Studio",
            font=ctk.CTkFont(size=22, weight="bold"),
            justify="left",
        ).grid(row=0, column=0, sticky="w", pady=(0, 12))

        self.mode_tabs = ctk.CTkTabview(sidebar, height=520)
        self.mode_tabs.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        self.mode_tabs.add("MD Trajectory")
        self.mode_tabs.add("GBSA / MMPBSA")

        self._build_md_panel(self.mode_tabs.tab("MD Trajectory"))
        self._build_gbsa_panel(self.mode_tabs.tab("GBSA / MMPBSA"))

        self.progress = ctk.CTkProgressBar(sidebar, mode="indeterminate")
        self.progress.grid(row=2, column=0, sticky="ew", pady=(8, 4))
        self.progress.grid_remove()

        self.run_btn = ctk.CTkButton(
            sidebar,
            text="Run Analysis",
            height=44,
            font=ctk.CTkFont(size=16, weight="bold"),
            command=self._run_analysis,
        )
        self.run_btn.grid(row=3, column=0, sticky="ew", pady=(4, 8))

        self.log_box = ctk.CTkTextbox(sidebar, height=140, wrap="word")
        self.log_box.grid(row=4, column=0, sticky="ew")
        self.log_box.insert("1.0", "Ready.\n")
        self.log_box.configure(state="disabled")

    def _build_md_panel(self, parent: Any) -> None:
        parent.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(parent, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", pady=(4, 4))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(header, text="Trajectories", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(header, text="+ Add", width=70, command=self._add_md_row).grid(row=0, column=1)

        self.md_rows_frame = ctk.CTkFrame(parent, fg_color="transparent")
        self.md_rows_frame.grid(row=1, column=0, sticky="ew")
        self.md_rows_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(parent, text="Analyses", font=ctk.CTkFont(weight="bold")).grid(row=2, column=0, sticky="w", pady=(12, 4))
        self.md_checks: dict[str, ctk.BooleanVar] = {}
        checks_frame = ctk.CTkFrame(parent, fg_color="transparent")
        checks_frame.grid(row=3, column=0, sticky="ew")
        for index, (key, label) in enumerate(MD_PLOT_TYPES.items()):
            var = ctk.BooleanVar(value=key in ("rmsd", "rmsf"))
            self.md_checks[key] = var
            ctk.CTkCheckBox(checks_frame, text=label, variable=var).grid(row=index // 2, column=index % 2, sticky="w", padx=4, pady=2)

        ctk.CTkLabel(parent, text="RMSD selection").grid(row=4, column=0, sticky="w", pady=(12, 2))
        self.rmsd_select_var = ctk.StringVar(value="name CA")
        ctk.CTkEntry(parent, textvariable=self.rmsd_select_var).grid(row=5, column=0, sticky="ew")

        ctk.CTkLabel(parent, text="Plot layout", font=ctk.CTkFont(weight="bold")).grid(row=6, column=0, sticky="w", pady=(12, 4))
        self.md_plot_mode = ctk.StringVar(value="combined")
        ctk.CTkRadioButton(parent, text="Combined (overlay datasets)", variable=self.md_plot_mode, value="combined").grid(
            row=7, column=0, sticky="w"
        )
        ctk.CTkRadioButton(parent, text="Individual (one plot per dataset)", variable=self.md_plot_mode, value="individual").grid(
            row=8, column=0, sticky="w"
        )

    def _build_gbsa_panel(self, parent: Any) -> None:
        parent.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(parent, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", pady=(4, 4))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(header, text="GBSA CSV files", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(header, text="+ Add", width=70, command=self._add_gbsa_row).grid(row=0, column=1)

        self.gbsa_rows_frame = ctk.CTkFrame(parent, fg_color="transparent")
        self.gbsa_rows_frame.grid(row=1, column=0, sticky="ew")
        self.gbsa_rows_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(parent, text="Plot types", font=ctk.CTkFont(weight="bold")).grid(row=2, column=0, sticky="w", pady=(12, 4))
        self.gbsa_checks: dict[str, ctk.BooleanVar] = {}
        checks_frame = ctk.CTkFrame(parent, fg_color="transparent")
        checks_frame.grid(row=3, column=0, sticky="ew")
        for index, (key, label) in enumerate(GBSA_PLOT_TYPES.items()):
            var = ctk.BooleanVar(value=True)
            self.gbsa_checks[key] = var
            ctk.CTkCheckBox(checks_frame, text=label, variable=var).grid(row=index // 2, column=index % 2, sticky="w", padx=4, pady=2)

        ctk.CTkLabel(parent, text="Rolling window (frames)").grid(row=4, column=0, sticky="w", pady=(12, 2))
        self.rolling_var = ctk.StringVar(value="10")
        ctk.CTkEntry(parent, textvariable=self.rolling_var, width=80).grid(row=5, column=0, sticky="w")

        ctk.CTkLabel(parent, text="Plot layout", font=ctk.CTkFont(weight="bold")).grid(row=6, column=0, sticky="w", pady=(12, 4))
        self.gbsa_plot_mode = ctk.StringVar(value="combined")
        ctk.CTkRadioButton(parent, text="Combined (overlay datasets)", variable=self.gbsa_plot_mode, value="combined").grid(
            row=7, column=0, sticky="w"
        )
        ctk.CTkRadioButton(
            parent, text="Individual (one plot per dataset)", variable=self.gbsa_plot_mode, value="individual"
        ).grid(row=8, column=0, sticky="w")

    def _build_main(self) -> None:
        main = ctk.CTkFrame(self)
        main.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(0, weight=1)

        self.plot_viewer = PlotViewer(main)
        self.plot_viewer.grid(row=0, column=0, sticky="nsew")

        self.summary_box = ctk.CTkTextbox(main, height=100, wrap="word")
        self.summary_box.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.summary_box.insert("1.0", "Summary statistics will appear here after analysis.\n")
        self.summary_box.configure(state="disabled")

    def _on_resize(self, _event: tk.Event) -> None:
        width = max(self.winfo_width(), MIN_WIDTH)
        if width < 900:
            self.grid_columnconfigure(0, weight=1)
            self.grid_columnconfigure(1, weight=2)
        else:
            self.grid_columnconfigure(0, weight=0)
            self.grid_columnconfigure(1, weight=1)

    def _add_md_row(self) -> None:
        row = DatasetRow(self.md_rows_frame, on_remove=self._remove_md_row)
        row.grid(row=len(self._md_rows), column=0, sticky="ew", pady=4)
        row.name_var.set(f"Dataset {len(self._md_rows) + 1}")
        self._md_rows.append(row)

    def _remove_md_row(self, row: DatasetRow) -> None:
        if len(self._md_rows) <= 1:
            messagebox.showwarning("Cannot remove", "At least one trajectory row is required.")
            return
        self._md_rows.remove(row)
        row.destroy()
        for index, item in enumerate(self._md_rows):
            item.grid(row=index, column=0, sticky="ew", pady=4)

    def _add_gbsa_row(self) -> None:
        row = GBSARow(self.gbsa_rows_frame, on_remove=self._remove_gbsa_row)
        row.grid(row=len(self._gbsa_rows), column=0, sticky="ew", pady=4)
        row.name_var.set(f"Run {len(self._gbsa_rows) + 1}")
        self._gbsa_rows.append(row)

    def _remove_gbsa_row(self, row: GBSARow) -> None:
        if len(self._gbsa_rows) <= 1:
            messagebox.showwarning("Cannot remove", "At least one GBSA file row is required.")
            return
        self._gbsa_rows.remove(row)
        row.destroy()
        for index, item in enumerate(self._gbsa_rows):
            item.grid(row=index, column=0, sticky="ew", pady=4)

    def _log(self, message: str) -> None:
        def append() -> None:
            self.log_box.configure(state="normal")
            self.log_box.insert("end", message + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")

        self.after(0, append)

    def _set_running(self, running: bool) -> None:
        def update() -> None:
            if running:
                self.run_btn.configure(state="disabled", text="Running…")
                self.progress.grid()
                self.progress.start()
            else:
                self.run_btn.configure(state="normal", text="Run Analysis")
                self.progress.stop()
                self.progress.grid_remove()

        self.after(0, update)

    def _set_summary(self, text: str) -> None:
        def update() -> None:
            self.summary_box.configure(state="normal")
            self.summary_box.delete("1.0", "end")
            self.summary_box.insert("1.0", text)
            self.summary_box.configure(state="disabled")

        self.after(0, update)

    def _run_analysis(self) -> None:
        if self._worker and self._worker.is_alive():
            messagebox.showinfo("Busy", "Analysis is already running.")
            return

        mode = self.mode_tabs.get()
        if mode == "MD Trajectory":
            self._worker = threading.Thread(target=self._run_md, daemon=True)
        else:
            self._worker = threading.Thread(target=self._run_gbsa, daemon=True)
        self._worker.start()

    def _run_md(self) -> None:
        self._set_running(True)
        try:
            datasets = [row.to_dataset() for row in self._md_rows]
            datasets = [d for d in datasets if d is not None]
            if not datasets:
                raise ValueError("Fill in label, topology, and trajectory for at least one row.")

            plot_types = [key for key, var in self.md_checks.items() if var.get()]
            combined = self.md_plot_mode.get() == "combined"

            self._log(f"Starting MD analysis ({len(datasets)} dataset(s), {len(plot_types)} plot type(s))...")
            results = run_md_analysis(
                datasets,
                plot_types,
                combined=combined,
                rmsd_select=self.rmsd_select_var.get().strip() or "name CA",
                progress=self._log,
            )

            def show_results() -> None:
                self.plot_viewer.clear()
                summaries = []
                for result in results:
                    self.plot_viewer.add_plot(result.label, result.figure, result.summary)
                    if result.summary:
                        summaries.append(result.summary)
                self._set_summary("\n".join(summaries) if summaries else "Analysis complete.")
                self._log("MD analysis finished.")

            self.after(0, show_results)
        except Exception as exc:
            self._log(f"Error: {exc}")
            self.after(0, lambda: messagebox.showerror("MD Analysis Error", str(exc)))
        finally:
            self._set_running(False)

    def _run_gbsa(self) -> None:
        self._set_running(True)
        try:
            datasets = [row.to_dataset() for row in self._gbsa_rows]
            datasets = [d for d in datasets if d is not None]
            if not datasets:
                raise ValueError("Fill in label and CSV path for at least one row.")

            plot_types = [key for key, var in self.gbsa_checks.items() if var.get()]
            combined = self.gbsa_plot_mode.get() == "combined"
            window = int(self.rolling_var.get().strip() or "10")

            self._log(f"Starting GBSA analysis ({len(datasets)} file(s), {len(plot_types)} plot type(s))...")
            results = run_gbsa_analysis(
                datasets,
                plot_types,
                combined=combined,
                rolling_window=window,
                progress=self._log,
            )

            def show_results() -> None:
                self.plot_viewer.clear()
                summaries = []
                for result in results:
                    self.plot_viewer.add_plot(result.label, result.figure, result.summary)
                    if result.summary:
                        summaries.append(result.summary)
                self._set_summary("\n".join(dict.fromkeys(summaries)) if summaries else "Analysis complete.")
                self._log("GBSA analysis finished.")

            self.after(0, show_results)
        except Exception as exc:
            self._log(f"Error: {exc}")
            self.after(0, lambda: messagebox.showerror("GBSA Analysis Error", str(exc)))
        finally:
            self._set_running(False)


def run_app() -> None:
    app = AnalysisApp()
    app.mainloop()
