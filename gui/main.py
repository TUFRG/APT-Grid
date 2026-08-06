import ctypes
import json
import os
import queue
import subprocess
import shutil
import sys
import threading
import tkinter as tk
import webbrowser
from datetime import datetime
from tkinter import filedialog, ttk, messagebox


APP_TITLE = "GridWorks"
WINDOW_SIZE = "1280x800"
SIDEBAR_WIDTH = 126
LOGO_FILENAME = "logo.png"
ICONS_FOLDER = "icons"
TOPBAR_HEIGHT = 120
SECOND_LOGO_FILENAME = "second_logo.png"
SIDEBAR_BG = "#eef7ff"
CONTENT_BG = "#f8fbff"
TOPBAR_BG = "#eaf5ff"
MAIN_BLUE = "#005298"


PARAMETER_HELP_TEXT = {
    # Basic Setup
    "Nb": "",
    "periodic": "",
    "scale": "",
    "nrad": "",

    # Boundary Layer reference inputs
    "rhoref": "",
    "Uref": "",
    "LrefHub": "",
    "LrefCas": "",
    "LrefBla": "",
    "muref": "",
    "yPlusHub": "",
    "yPlusCas": "",
    "yPlusBla": "",

    # Boundary Layer manual results
    "delHub": "",
    "delCas": "",
    "delBla": "",
    "dy1Hub": "",
    "dy1Cas": "",
    "dy1Bla": "",

    # Mesh Tuning
    "gRad": "",
    "gTan": "",
    "additionalTangentialRefine": "",
    "additionalAxialRefine": "",
    "dax1primeLE": "",
    "rLE": "",
    "dax1primeTE": "",
    "rTE": "",
    "rUpFar": "",
    "rDnFar": "",

    # Advanced
    "percentVal": "",
    "percentValNonCutLE": "",
    "percentValNonCutTE": "",
    "angConstraintCurves": "",
    "angConstraintOffsets": "",
}


class ParameterHelp:
    def __init__(self, widget, title, text):
        self.widget = widget
        self.title = title
        self.text = text
        widget.bind("<Button-1>", self.show_definition_window)
        widget.bind("<Enter>", self._on_enter)
        widget.bind("<Leave>", self._on_leave)

    def _on_enter(self, event=None):
        self.widget.configure(cursor="hand2")

    def _on_leave(self, event=None):
        self.widget.configure(cursor="")

    def show_definition_window(self, event=None):
        popup = tk.Toplevel(self.widget)
        popup.title(f"{self.title} Definition")
        popup.configure(bg="white")
        popup.resizable(False, False)
        popup.transient(self.widget.winfo_toplevel())
        popup.grab_set()

        popup.update_idletasks()
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 20
        popup.geometry(f"420x190+{x}+{y}")

        header = tk.Label(
            popup,
            text=self.title,
            font=("Segoe UI", 12, "bold"),
            fg=MAIN_BLUE,
            bg="white",
            anchor="w"
        )
        header.pack(fill="x", padx=22, pady=(20, 8))

        definition = tk.Label(
            popup,
            text=self.text,
            font=("Segoe UI", 10),
            fg="#1f2933",
            bg="white",
            justify="left",
            wraplength=370,
            anchor="nw",
            height=4
        )
        definition.pack(fill="both", expand=True, padx=22, pady=(0, 16))

        close_button = tk.Button(
            popup,
            text="Close",
            command=popup.destroy,
            font=("Segoe UI", 9),
            fg=MAIN_BLUE,
            bg="#eef7ff",
            activeforeground=MAIN_BLUE,
            activebackground="#d6ecff",
            relief="flat",
            bd=0,
            padx=18,
            pady=7,
            cursor="hand2"
        )
        close_button.pack(anchor="e", padx=22, pady=(0, 20))

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass


class AppState:
    def __init__(self):
        self.defaults = {
            # Files
            "dataPath": "",
            "hubFileName": "",
            "casFileName": "",
            "bladeCurveFile": "",
            "outputPath": "",

            # Basic Setup
            "Nb": 20,
            "periodic": 0,
            "scale": 0.001,
            "nrad": 40,

            # Boundary Layer
            "rhoref": 1.2,
            "Uref": 100.0,
            "LrefHub": 178.0,
            "LrefCas": 178.0,
            "LrefBla": 35.0,
            "muref": 1.8e-5,
            "yPlusHub": 5,
            "yPlusCas": 5,
            "yPlusBla": 5,
            "delHub": 0.0,
            "delCas": 0.0,
            "delBla": 0.0,
            "dy1Hub": 0.0,
            "dy1Cas": 0.0,
            "dy1Bla": 0.0,
            "autoBL": True,

            # Mesh Tuning
            "gRad": 2,
            "gTan": 2,
            "additionalTangentialRefine": 8,
            "dax1primeLE": 0.003,
            "rLE": 1.2,
            "dax1primeTE": 0.002,
            "rTE": 1.2,
            "additionalAxialRefine": 2,
            "rUpFar": 1.1,
            "rDnFar": 1.1,

            # Advanced
            "percentVal": 0.04,
            "percentValNonCutLE": 0.02,
            "percentValNonCutTE": 0.00,
            "angConstraintCurves": 10,
            "angConstraintOffsets": 1,
        }

        self.values = self.defaults.copy()

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = value

    def reset_group(self, keys):
        for key in keys:
            self.values[key] = self.defaults[key]

    def reset_files(self):
        self.reset_group([
            "dataPath",
            "hubFileName",
            "casFileName",
            "bladeCurveFile",
            "outputPath",
        ])

    def reset_basic_setup(self):
        self.reset_group(["Nb", "periodic", "scale", "nrad"])

    def reset_boundary_layer(self):
        self.reset_group([
            "rhoref", "Uref", "LrefHub", "LrefCas", "LrefBla", "muref",
            "yPlusHub", "yPlusCas", "yPlusBla",
            "delHub", "delCas", "delBla",
            "dy1Hub", "dy1Cas", "dy1Bla",
            "autoBL",
        ])

    def reset_mesh_tuning(self):
        self.reset_group([
            "gRad", "gTan", "additionalTangentialRefine",
            "dax1primeLE", "rLE", "dax1primeTE", "rTE",
            "additionalAxialRefine", "rUpFar", "rDnFar",
        ])

    def reset_advanced(self):
        self.reset_group([
            "percentVal",
            "percentValNonCutLE",
            "percentValNonCutTE",
            "angConstraintCurves",
            "angConstraintOffsets",
        ])


class BasePage(ttk.Frame):
    def __init__(self, parent, app, title: str):
        super().__init__(parent, padding=24, style="Content.TFrame")
        self.app = app
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self.title_label = ttk.Label(self, text=title, style="PageTitle.TLabel")
        self.title_label.grid(row=0, column=0, sticky="w")

        self.body = ttk.Frame(self, padding=(0, 24, 0, 0), style="Content.TFrame")
        self.body.grid(row=1, column=0, sticky="nsew")
        self.body.columnconfigure(0, weight=1)


    def _create_parameter_label(self, parent, label_text, help_key):
        label_frame = ttk.Frame(parent, style="Content.TFrame")

        label = ttk.Label(label_frame, text=label_text, style="Field.TLabel")
        label.grid(row=0, column=0, sticky="w")

        help_text = PARAMETER_HELP_TEXT.get(help_key, "")
        help_icon = tk.Canvas(
            label_frame,
            width=14,
            height=14,
            bg=CONTENT_BG,
            highlightthickness=0,
            bd=0,
            cursor="hand2"
        )
        help_icon.create_oval(1, 1, 13, 13, fill="white", outline="#9bb7d4", width=1)
        help_icon.create_text(7, 7, text="?", fill="#5f7ea8", font=("Segoe UI", 7, "bold"))
        help_icon.grid(row=0, column=1, sticky="w", padx=(5, 0))
        ParameterHelp(help_icon, label_text, help_text)

        return label_frame


class HomePage(BasePage):
    def __init__(self, parent, app):
        super().__init__(parent, app, title="Home")
        self.title_label.grid_remove()
        self._step_icons = []
        self.hero_icon_image = None
        self.tagline_full_text = "Guided mesh generation for blade passages."
        self.tagline_animation_index = 0
        self.tagline_after_id = None
        self._build_page()

    def _build_page(self):
        self.body.columnconfigure(0, weight=1)
        self.body.configure(padding=(0, 30, 0, 14))

        self.body.rowconfigure(0, weight=0)
        self.body.rowconfigure(1, weight=0)
        self.body.rowconfigure(2, weight=1)
        self.body.rowconfigure(3, weight=0)

        # --------------------------------------------------
        # Hero section: icon, name, animated pitch, single CTA
        # --------------------------------------------------
        hero = tk.Frame(self.body, bg=CONTENT_BG)
        hero.grid(row=0, column=0, sticky="ew", pady=(6, 46))
        hero.columnconfigure(0, weight=1)

        hero_icon = self._load_hero_icon()
        if hero_icon is not None:
            self.hero_icon_image = hero_icon
            icon_label = tk.Label(hero, image=self.hero_icon_image, bg=CONTENT_BG, bd=0)
            icon_label.grid(row=0, column=0, pady=(62, 2))
        else:
            badge = tk.Canvas(
                hero, width=96, height=96, bg=CONTENT_BG,
                highlightthickness=0, bd=0
            )
            badge.grid(row=0, column=0, pady=(62, 2))
            self._draw_hero_badge(badge)

        title_frame = tk.Frame(hero, bg=CONTENT_BG, width=430, height=106)
        title_frame.grid(row=1, column=0)
        title_frame.grid_propagate(False)

        name_label = tk.Label(
            title_frame,
            text="GridWorks",
            font=("Segoe UI", 30, "bold"),
            fg=MAIN_BLUE,
            bg=CONTENT_BG
        )
        name_label.place(relx=0.5, y=0, anchor="n")

        powered_label = tk.Label(
            title_frame,
            text="powered by TUFRG",
            font=("Segoe UI", 11, "italic"),
            fg="#5f7ea8",
            bg=CONTENT_BG
        )
        powered_label.place(x=268, y=64, anchor="nw")

        self.tagline_label = tk.Label(
            hero,
            text="",
            font=("Segoe UI", 12),
            fg="#5f7ea8",
            bg=CONTENT_BG
        )
        self.tagline_label.grid(row=2, column=0, pady=(6, 58))

        start_button = tk.Button(
            hero,
            text="Start New Project",
            command=lambda: self.app.show_page("Files"),
            font=("Segoe UI", 11, "bold"),
            fg="white",
            bg=MAIN_BLUE,
            activeforeground="white",
            activebackground="#003f73",
            disabledforeground="white",
            relief="flat",
            bd=0,
            padx=30,
            pady=11,
            cursor="hand2"
        )
        start_button.grid(row=3, column=0)

        # --------------------------------------------------
        # Workflow strip: four steps connected by a light line.
        # The icons are shown without border boxes.
        # --------------------------------------------------
        steps_section = tk.Frame(self.body, bg=CONTENT_BG)
        steps_section.grid(row=1, column=0, sticky="ew", padx=60, pady=(40, 30))
        steps_section.columnconfigure(0, weight=1)

        steps_heading = tk.Label(
            steps_section,
            text="How it works",
            font=("Segoe UI", 11, "bold"),
            fg="#5f7ea8",
            bg=CONTENT_BG
        )
        steps_heading.grid(row=0, column=0, pady=(0, 18))

        steps_row = tk.Frame(steps_section, bg=CONTENT_BG)
        steps_row.grid(row=1, column=0, sticky="ew", pady=(58, 0))

        workflow_steps = [
            ("1", "Files", "folder.png"),
            ("2", "Setup", "grid.png"),
            ("3", "Mesh Controls", "tuning.png"),
            ("4", "Run", "run.png"),
        ]
        num_steps = len(workflow_steps)

        for col in range(num_steps):
            steps_row.columnconfigure(col, weight=1, uniform="step")

        connector = tk.Canvas(
            steps_row, height=60, bg=CONTENT_BG, highlightthickness=0, bd=0
        )
        connector.grid(row=0, column=0, columnspan=num_steps, sticky="new")

        def _redraw_connector(event, canvas=connector):
            canvas.delete("line")
            if num_steps <= 1:
                return

            step_width = event.width / num_steps
            icon_gap = 48

            for line_index in range(num_steps - 1):
                left_center = step_width * (line_index + 0.5)
                right_center = step_width * (line_index + 1.5)
                canvas.create_line(
                    left_center + icon_gap,
                    30,
                    right_center - icon_gap,
                    30,
                    fill="#c8dcf4",
                    width=3,
                    tags="line"
                )

        connector.bind("<Configure>", _redraw_connector)

        for index, (number, label, icon_file) in enumerate(workflow_steps):
            step = tk.Frame(steps_row, bg=CONTENT_BG)
            step.grid(row=0, column=index, sticky="n")

            tile = tk.Canvas(
                step, width=60, height=60, bg=CONTENT_BG,
                highlightthickness=0, bd=0
            )
            tile.grid(row=0, column=0)

            icon = self._load_nav_icon_small(icon_file)
            if icon is not None:
                self._step_icons.append(icon)
                tile.create_image(30, 30, image=icon, anchor="center")
            else:
                tile.create_text(
                    30, 30, text=number, font=("Segoe UI", 14, "bold"), fill=MAIN_BLUE
                )

            step_label = tk.Label(
                step,
                text=label,
                font=("Segoe UI", 10, "bold"),
                fg="#1f2933",
                bg=CONTENT_BG
            )
            step_label.grid(row=1, column=0, pady=(10, 0))

        # Spacer row keeps the footer pinned near the bottom of the page.
        spacer = tk.Frame(self.body, bg=CONTENT_BG)
        spacer.grid(row=2, column=0, sticky="nsew")

        footer = tk.Label(
            self.body,
            text="Backend by Adekola Adeyemi, Justin Smart, Tony Woo & Jeff Defoe   \u2022   Interface by Misk Damdoum",
            font=("Segoe UI", 9),
            fg="#9bb1c9",
            bg=CONTENT_BG
        )
        footer.grid(row=3, column=0, sticky="s", pady=(8, 0))

    def on_show(self):
        if self.tagline_after_id is not None:
            self.after_cancel(self.tagline_after_id)
            self.tagline_after_id = None

        self.tagline_animation_index = 0
        self.tagline_label.configure(text="")
        self._animate_tagline_once()

    def _animate_tagline_once(self):
        if not hasattr(self, "tagline_label"):
            return

        shown_text = self.tagline_full_text[:self.tagline_animation_index]
        cursor = "|" if self.tagline_animation_index < len(self.tagline_full_text) else ""
        self.tagline_label.configure(text=shown_text + cursor)

        if self.tagline_animation_index < len(self.tagline_full_text):
            self.tagline_animation_index += 1
            self.tagline_after_id = self.after(38, self._animate_tagline_once)
        else:
            self.tagline_after_id = None

    def _load_hero_icon(self):
        path = os.path.join(os.path.dirname(__file__), ICONS_FOLDER, "gui_logo.png")
        if os.path.exists(path):
            try:
                return tk.PhotoImage(file=path)
            except tk.TclError:
                return None
        return None

    def _draw_hero_badge(self, canvas):
        self.app._draw_rounded_rect(canvas, 2, 2, 94, 94, radius=24, fill=MAIN_BLUE)
        # fallback mesh glyph if icons/gui_logo.png is not available
        offset = 22
        step = 17
        for i in range(3):
            x = offset + i * step
            canvas.create_line(x, offset - 4, x, offset + 2 * step + 4, fill="white", width=2)
            y = offset + i * step
            canvas.create_line(offset - 4, y, offset + 2 * step + 4, y, fill="white", width=2)

    def _load_nav_icon_small(self, filename):
        path = os.path.join(os.path.dirname(__file__), ICONS_FOLDER, filename)
        if os.path.exists(path):
            try:
                return tk.PhotoImage(file=path)
            except tk.TclError:
                return None
        return None



class FilesPage(BasePage):
    def __init__(self, parent, app):
        super().__init__(parent, app, title="Files")
        self.file_vars = {}
        self._build_page()
        self.load_state()

    def _build_page(self):
        content = ttk.Frame(self.body, style="Content.TFrame")
        content.grid(row=0, column=0, sticky="nw")

        section = ttk.Label(content, text="File / Path Selection", style="Section.TLabel")
        section.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 20))

        field_specs = [
            ("Input data", "dataPath", "directory"),
            ("Hub curve file", "hubFileName", "file"),
            ("Casing curve file", "casFileName", "file"),
            ("Blade curve file", "bladeCurveFile", "file"),
            ("Output data", "outputPath", "directory"),
        ]

        for row_index, (label_text, key, browse_type) in enumerate(field_specs, start=1):
            label = ttk.Label(content, text=label_text, style="Field.TLabel")
            label.grid(row=row_index, column=0, sticky="w", pady=12, padx=(0, 16))

            var = tk.StringVar()
            entry = ttk.Entry(content, textvariable=var, width=52)
            entry.grid(row=row_index, column=1, sticky="ew", pady=12)
            content.columnconfigure(1, weight=1)

            self.file_vars[key] = var
            var.trace_add(
                "write",
                lambda *args, state_key=key, tk_var=var: self.app.state.set(state_key, tk_var.get())
            )

            browse_button = ttk.Button(
                content,
                text="Browse...",
                takefocus=False,
                command=lambda state_key=key, kind=browse_type: self.browse_for_path(state_key, kind),
            )
            browse_button.grid(row=row_index, column=2, sticky="w", padx=(16, 0), pady=12)

        action_bar = ttk.Frame(self.body, style="Content.TFrame")
        action_bar.grid(row=1, column=0, sticky="se", pady=(28, 0))

        ttk.Button(action_bar, text="Clear All", takefocus=False,command=self.clear_all).grid(row=0, column=0, padx=(0, 12))

    def browse_for_path(self, key, browse_type):
        if browse_type == "directory":
            selected_path = filedialog.askdirectory(title="Select folder")

            if selected_path:
                self.file_vars[key].set(selected_path)

        else:
            selected_path = filedialog.askopenfilename(title="Select file")

            if selected_path:
                selected_folder = os.path.dirname(selected_path)
                selected_filename = os.path.basename(selected_path)

                # Store only the file name because the backend uses dataPath + fileName.
                self.file_vars[key].set(selected_filename)

                # If the user selects one of the input curve files, automatically
                # set Input data to that file's folder.
                if key in ["hubFileName", "casFileName", "bladeCurveFile"]:
                    self.file_vars["dataPath"].set(selected_folder)

    def clear_all(self):
        self.app.state.reset_files()
        self.load_state()

    def load_state(self):
        for key, var in self.file_vars.items():
            var.set(self.app.state.get(key, ""))


class BasicSetupPage(BasePage):
    def __init__(self, parent, app):
        super().__init__(parent, app, title="Basic Setup")

        self.loading_state = False

        self.blades_var = tk.IntVar()
        self.periodic_var = tk.IntVar()
        self.scale_var = tk.DoubleVar()
        self.nrad_var = tk.IntVar()

        self._build_page()
        self.load_state()

        for var in [self.blades_var, self.periodic_var, self.scale_var, self.nrad_var]:
            var.trace_add("write", self.update_state)

    def _build_page(self):
        content = ttk.Frame(self.body, style="Content.TFrame")
        content.grid(row=0, column=0, sticky="nw")

        ttk.Label(content, text="Basic Setup Parameters", style="Section.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 24)
        )

        fields = [
            ("Number of blades", "Nb", self.blades_var, 1, 500, 1),
            ("Scale (m)", "scale", self.scale_var, 0.000001, 1000.0, 0.001),
            ("Radial points (outside of hub and casing boundary layers)", "nrad", self.nrad_var, 1, 1000, 1),
        ]

        for row, (label_text, key, var, min_val, max_val, step) in enumerate(fields, start=1):
            label = self._create_parameter_label(content, label_text, key)
            label.grid(row=row, column=0, sticky="w", padx=(0, 24), pady=12)

            spinbox = ttk.Spinbox(
                content,
                from_=min_val,
                to=max_val,
                increment=step,
                textvariable=var,
                width=14,
                takefocus=False,
            )
            spinbox.grid(row=row, column=1, sticky="w", pady=12)

            def clear_selection(event):
                event.widget.after_idle(lambda: event.widget.selection_clear())

            spinbox.bind("<<Increment>>", clear_selection)
            spinbox.bind("<<Decrement>>", clear_selection)
            spinbox.bind("<ButtonRelease-1>", clear_selection)
            spinbox.bind("<KeyRelease>", clear_selection)
            spinbox.bind("<FocusIn>", clear_selection)

        periodic_label = self._create_parameter_label(content, "Periodic mode", "periodic")
        periodic_label.grid(row=4, column=0, sticky="w", padx=(0, 24), pady=12)

        ttk.Checkbutton(
            content,
            text="Enabled",
            variable=self.periodic_var,
            onvalue=1,
            offvalue=0,
            takefocus=False
        ).grid(row=4, column=1, sticky="w", pady=12)

        action_bar = ttk.Frame(self.body, style="Content.TFrame")
        action_bar.grid(row=1, column=0, sticky="se", pady=(28, 0))

        ttk.Button(
            action_bar,
            text="Clear All",
            takefocus=False,
            command=self.clear_all
        ).grid(row=0, column=0, padx=(0, 12))

    def update_state(self, *args):
        if self.loading_state:
            return

        try:
            self.app.state.set("Nb", self.blades_var.get())
            self.app.state.set("periodic", int(self.periodic_var.get()))
            self.app.state.set("scale", self.scale_var.get())
            self.app.state.set("nrad", self.nrad_var.get())
        except tk.TclError:
            pass

    def clear_all(self):
        self.app.state.reset_basic_setup()
        self.load_state()

    def load_state(self):
        self.loading_state = True

        self.blades_var.set(self.app.state.get("Nb"))
        self.periodic_var.set(int(self.app.state.get("periodic", 0)))
        self.scale_var.set(self.app.state.get("scale"))
        self.nrad_var.set(self.app.state.get("nrad"))

        self.loading_state = False

class BoundaryLayerPage(BasePage):
    def __init__(self, parent, app):
        super().__init__(parent, app, title="Boundary Layer")

        self.vars = {}
        self.manual_widgets = []
        self.loading_state = False
        self.auto_bl_var = tk.BooleanVar()

        self._build_page()
        self.load_state()

    def _build_page(self):
        content = ttk.Frame(self.body, style="Content.TFrame")
        content.grid(row=0, column=0, sticky="nw")

        content.columnconfigure(1, weight=1)
        content.columnconfigure(3, weight=1)

        ttk.Label(
            content,
            text="Reference Inputs",
            style="Section.TLabel"
        ).grid(
            row=0,
            column=0,
            columnspan=4,
            sticky="w",
            pady=(0, 18)
        )

        reference_fields = [
            ("Reference density (kg/m³)", "rhoref"),
            ("Reference velocity (m/s)", "Uref"),
            ("Hub reference length (m)", "LrefHub"),
            ("Casing reference length (m)", "LrefCas"),
            ("Blade reference length (m)", "LrefBla"),
            ("Reference viscosity (kg/m*s)", "muref"),
            ("Hub y+ target", "yPlusHub"),
            ("Casing y+ target", "yPlusCas"),
            ("Blade y+ target", "yPlusBla"),
        ]

        for index, (label_text, key) in enumerate(reference_fields):
            row = 1 + index // 2
            label_col = 0 if index % 2 == 0 else 2
            entry_col = 1 if index % 2 == 0 else 3

            self._add_numeric_field(
                parent=content,
                row=row,
                label_col=label_col,
                entry_col=entry_col,
                label_text=label_text,
                key=key
            )

        results_start_row = 7

        ttk.Separator(content, orient="horizontal").grid(
            row=results_start_row,
            column=0,
            columnspan=4,
            sticky="ew",
            pady=(24, 18)
        )

        ttk.Label(
            content,
            text="Boundary-Layer Results",
            style="Section.TLabel"
        ).grid(
            row=results_start_row + 1,
            column=0,
            columnspan=4,
            sticky="w",
            pady=(0, 12)
        )

        ttk.Checkbutton(
            content,
            text="Auto-calculate boundary-layer results from reference inputs",
            variable=self.auto_bl_var,
            takefocus=False,
            command=self.toggle_auto_boundary_layer
        ).grid(
            row=results_start_row + 2,
            column=0,
            columnspan=4,
            sticky="w",
            pady=(0, 14)
        )

        manual_note = ttk.Label(
            content,
            text="NOTE: please provide boundary-layer thickness and first cell size in input units.",
            style="Muted.TLabel",
            wraplength=820,
            justify="left",
        )
        manual_note.grid(
            row=results_start_row + 3,
            column=0,
            columnspan=4,
            sticky="w",
            pady=(0, 8)
        )
        self.manual_widgets.append(manual_note)

        result_fields = [
            ("Hub boundary-layer thickness", "delHub"),
            ("Casing boundary-layer thickness", "delCas"),
            ("Blade boundary-layer thickness", "delBla"),
            ("Hub first cell size", "dy1Hub"),
            ("Casing first cell size", "dy1Cas"),
            ("Blade first cell size", "dy1Bla"),
        ]

        for index, (label_text, key) in enumerate(result_fields):
            row = results_start_row + 4 + index // 2
            label_col = 0 if index % 2 == 0 else 2
            entry_col = 1 if index % 2 == 0 else 3

            self._add_numeric_field(
                parent=content,
                row=row,
                label_col=label_col,
                entry_col=entry_col,
                label_text=label_text,
                key=key,
                manual=True
            )

        action_bar = ttk.Frame(self.body, style="Content.TFrame")
        action_bar.grid(row=1, column=0, sticky="se", pady=(15, 0))

        ttk.Button(
            action_bar,
            text="Clear All",
            takefocus=False,
            command=self.clear_all
        ).grid(row=0, column=0, padx=(0, 12))

    def _add_numeric_field(self, parent, row, label_col, entry_col, label_text, key, manual=False):
        label = self._create_parameter_label(parent, label_text, key)
        label.grid(
            row=row,
            column=label_col,
            sticky="w",
            padx=(0, 16),
            pady=10
        )

        var = tk.DoubleVar()
        self.vars[key] = var

        min_val, max_val, step = self._get_spinbox_settings(key)

        spinbox = ttk.Spinbox(
            parent,
            from_=min_val,
            to=max_val,
            increment=step,
            textvariable=var,
            width=16,
            takefocus=False,
        )
        spinbox.grid(
            row=row,
            column=entry_col,
            sticky="w",
            padx=(0, 32),
            pady=10
        )

        def clear_selection(event):
            widget = event.widget
            widget.after(10, lambda: widget.selection_clear())
            widget.after(20, lambda: widget.icursor("end"))

        spinbox.bind("<<Increment>>", clear_selection)
        spinbox.bind("<<Decrement>>", clear_selection)
        spinbox.bind("<ButtonRelease-1>", clear_selection)
        spinbox.bind("<KeyRelease>", clear_selection)
        spinbox.bind("<FocusIn>", clear_selection)

        if manual:
            self.manual_widgets.extend([label, spinbox])

        var.trace_add(
            "write",
            lambda *args, state_key=key, tk_var=var: self._update_single_value(state_key, tk_var)
        )

    def _get_spinbox_settings(self, key):
        settings = {
            # Reference inputs
            "rhoref": (0.0001, 100.0, 0.1),
            "Uref": (0.001, 10000.0, 1.0),
            "LrefHub": (0.0001, 100000.0, 1.0),
            "LrefCas": (0.0001, 100000.0, 1.0),
            "LrefBla": (0.0001, 100000.0, 1.0),
            "muref": (0.000000001, 1.0, 0.000001),

            # y+ targets
            "yPlusHub": (0.1, 1000.0, 1.0),
            "yPlusCas": (0.1, 1000.0, 1.0),
            "yPlusBla": (0.1, 1000.0, 1.0),

            # Manual boundary-layer result inputs
            "delHub": (0.0, 100000.0, 0.001),
            "delCas": (0.0, 100000.0, 0.001),
            "delBla": (0.0, 100000.0, 0.001),
            "dy1Hub": (0.0, 100000.0, 0.001),
            "dy1Cas": (0.0, 100000.0, 0.001),
            "dy1Bla": (0.0, 100000.0, 0.001),
        }

        return settings.get(key, (0.0, 100000.0, 1.0))

    def _update_single_value(self, key, var):
        if self.loading_state:
            return

        try:
            self.app.state.set(key, var.get())
        except tk.TclError:
            return

        reference_keys = [
            "rhoref",
            "Uref",
            "LrefHub",
            "LrefCas",
            "LrefBla",
            "muref",
            "yPlusHub",
            "yPlusCas",
            "yPlusBla",
        ]

        if self.auto_bl_var.get() and key in reference_keys:
            self.calculate_boundary_layer_results()

    def calc_bl_delta(self, rho, U, L, mu):
        Re = rho * U * L / mu
        return 0.37 * L / (Re ** (1 / 5))

    def calc_first_cell_size(self, rho, U, L, mu, yplus):
        Re = rho * U * L / mu
        Cf = 0.026 / (Re ** (1 / 7))
        Uf = (Cf * (U ** 2) * 0.5) ** 0.5
        return yplus * (mu / rho) / Uf

    def calculate_boundary_layer_results(self):
        try:
            rho = self.vars["rhoref"].get()
            U = self.vars["Uref"].get()
            mu = self.vars["muref"].get()
            scale = self.app.state.get("scale", 1.0)

            Lhub = self.vars["LrefHub"].get() * scale
            Lcas = self.vars["LrefCas"].get() * scale
            Lbla = self.vars["LrefBla"].get() * scale

            yhub = self.vars["yPlusHub"].get()
            ycas = self.vars["yPlusCas"].get()
            ybla = self.vars["yPlusBla"].get()

            calculated_values = {
                "delHub": self.calc_bl_delta(rho, U, Lhub, mu) / scale,
                "delCas": self.calc_bl_delta(rho, U, Lcas, mu) / scale,
                "delBla": self.calc_bl_delta(rho, U, Lbla, mu) / scale,
                "dy1Hub": self.calc_first_cell_size(rho, U, Lhub, mu, yhub) / scale,
                "dy1Cas": self.calc_first_cell_size(rho, U, Lcas, mu, ycas) / scale,
                "dy1Bla": self.calc_first_cell_size(rho, U, Lbla, mu, ybla) / scale,
            }

            self.loading_state = True

            for key, value in calculated_values.items():
                self.vars[key].set(value)
                self.app.state.set(key, value)

            self.loading_state = False

        except Exception:
            self.loading_state = False

    def toggle_auto_boundary_layer(self):
        auto_enabled = self.auto_bl_var.get()
        self.app.state.set("autoBL", auto_enabled)

        if auto_enabled:
            self.calculate_boundary_layer_results()

            for widget in self.manual_widgets:
                widget.grid_remove()
        else:
            for widget in self.manual_widgets:
                widget.grid()

    def clear_all(self):
        self.app.state.reset_boundary_layer()
        self.load_state()

    def load_state(self):
        self.loading_state = True

        for key, var in self.vars.items():
            var.set(self.app.state.get(key, 0.0))

        self.auto_bl_var.set(self.app.state.get("autoBL", True))

        self.loading_state = False

        self.toggle_auto_boundary_layer()


class PlaceholderPage(BasePage):
    def __init__(self, parent, app, title):
        super().__init__(parent, app, title=title)

        label = ttk.Label(
            self.body,
            text="This is a placeholder.",
            style="Body.TLabel",
            wraplength=820,
            justify="left",
        )
        label.grid(row=0, column=0, sticky="w")


class MeshTuningPage(BasePage):
    def __init__(self, parent, app):
        super().__init__(parent, app, title="Mesh Tuning")

        self.loading_state = False
        self.vars = {}
        self.spinboxes = []

        self._build_page()
        self.load_state()

    def _build_page(self):
        self.body.rowconfigure(0, weight=1)
        self.body.columnconfigure(0, weight=1)

        canvas = tk.Canvas(
            self.body,
            bg=CONTENT_BG,
            highlightthickness=0
        )
        canvas.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(
            self.body,
            orient="vertical",
            command=canvas.yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns")

        canvas.configure(yscrollcommand=scrollbar.set)

        content = ttk.Frame(canvas, style="Content.TFrame")
        canvas_window = canvas.create_window((0, 0), window=content, anchor="nw")

        def update_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def update_canvas_width(event):
            canvas.itemconfig(canvas_window, width=event.width)

        content.bind("<Configure>", update_scroll_region)
        canvas.bind("<Configure>", update_canvas_width)

        def mousewheel_scroll(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda event: canvas.bind_all("<MouseWheel>", mousewheel_scroll))
        canvas.bind("<Leave>", lambda event: canvas.unbind_all("<MouseWheel>"))

        content.columnconfigure(1, weight=1)
        content.columnconfigure(3, weight=1)

        ttk.Label(
            content,
            text="General Mesh Controls",
            style="Section.TLabel"
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 18))

        general_fields = [
            ("Radial grading ratio", "gRad", 0.1, 20.0, 0.1),
            ("Tangential grading ratio", "gTan", 0.1, 20.0, 0.1),
            ("Additional tangential refinement", "additionalTangentialRefine", 0, 200, 1),
            ("Additional axial refinement", "additionalAxialRefine", 0, 200, 1),
        ]

        for index, (label_text, key, min_val, max_val, step) in enumerate(general_fields):
            row = 1 + index // 2
            label_col = 0 if index % 2 == 0 else 2
            entry_col = 1 if index % 2 == 0 else 3

            self._add_spinbox_field(
                parent=content,
                row=row,
                label_col=label_col,
                entry_col=entry_col,
                label_text=label_text,
                key=key,
                min_val=min_val,
                max_val=max_val,
                step=step,
            )

        le_start_row = 4

        ttk.Separator(content, orient="horizontal").grid(
            row=le_start_row,
            column=0,
            columnspan=4,
            sticky="ew",
            pady=(22, 16)
        )

        ttk.Label(
            content,
            text="Leading-Edge Clustering",
            style="Section.TLabel"
        ).grid(row=le_start_row + 1, column=0, columnspan=4, sticky="w", pady=(0, 18))

        le_fields = [
            ("Leading-edge clustering distance", "dax1primeLE", 0.000001, 1.0, 0.001),
            ("Leading-edge expansion ratio", "rLE", 0.1, 10.0, 0.1),
        ]

        for index, (label_text, key, min_val, max_val, step) in enumerate(le_fields):
            row = le_start_row + 2 + index

            self._add_spinbox_field(
                parent=content,
                row=row,
                label_col=0,
                entry_col=1,
                label_text=label_text,
                key=key,
                min_val=min_val,
                max_val=max_val,
                step=step,
            )

        te_start_row = 8

        ttk.Separator(content, orient="horizontal").grid(
            row=te_start_row,
            column=0,
            columnspan=4,
            sticky="ew",
            pady=(22, 16)
        )

        ttk.Label(
            content,
            text="Trailing-Edge Clustering",
            style="Section.TLabel"
        ).grid(row=te_start_row + 1, column=0, columnspan=4, sticky="w", pady=(0, 18))

        te_fields = [
            ("Trailing-edge clustering distance", "dax1primeTE", 0.000001, 1.0, 0.001),
            ("Trailing-edge expansion ratio", "rTE", 0.1, 10.0, 0.1),
        ]

        for index, (label_text, key, min_val, max_val, step) in enumerate(te_fields):
            row = te_start_row + 2 + index

            self._add_spinbox_field(
                parent=content,
                row=row,
                label_col=0,
                entry_col=1,
                label_text=label_text,
                key=key,
                min_val=min_val,
                max_val=max_val,
                step=step,
            )

        far_start_row = 12

        ttk.Separator(content, orient="horizontal").grid(
            row=far_start_row,
            column=0,
            columnspan=4,
            sticky="ew",
            pady=(22, 16)
        )

        ttk.Label(
            content,
            text="Far-Field Expansion",
            style="Section.TLabel"
        ).grid(row=far_start_row + 1, column=0, columnspan=4, sticky="w", pady=(0, 18))

        far_fields = [
            ("Upstream far-field expansion ratio", "rUpFar", 0.1, 10.0, 0.1),
            ("Downstream far-field expansion ratio", "rDnFar", 0.1, 10.0, 0.1),
        ]

        for index, (label_text, key, min_val, max_val, step) in enumerate(far_fields):
            row = far_start_row + 2 + index

            self._add_spinbox_field(
                parent=content,
                row=row,
                label_col=0,
                entry_col=1,
                label_text=label_text,
                key=key,
                min_val=min_val,
                max_val=max_val,
                step=step,
            )


        action_bar = ttk.Frame(content, style="Content.TFrame")
        action_bar.grid(
            row=far_start_row + 6,
            column=0,
            columnspan=4,
            sticky="e",
            padx=(0, 40),            
            pady=(18, 24)
        )

        ttk.Button(
            action_bar,
            text="Clear All",
            takefocus=False,
            command=self.clear_all
        ).grid(row=0, column=0)

    def _add_spinbox_field(self, parent, row, label_col, entry_col, label_text, key, min_val, max_val, step):
        label = self._create_parameter_label(parent, label_text, key)
        label.grid(
            row=row,
            column=label_col,
            sticky="w",
            padx=(0, 16),
            pady=10
        )

        var = tk.StringVar()
        self.vars[key] = var

        spinbox = ttk.Spinbox(
            parent,
            from_=min_val,
            to=max_val,
            increment=step,
            textvariable=var,
            width=16,
            takefocus=False,
        )
        spinbox.grid(
            row=row,
            column=entry_col,
            sticky="w",
            padx=(0, 32),
            pady=10
        )

        self.spinboxes.append(spinbox)

        def clear_selection(event):
            widget = event.widget
            widget.after(10, lambda: widget.selection_clear())
            widget.after(20, lambda: widget.icursor("end"))

        spinbox.bind("<<Increment>>", clear_selection)
        spinbox.bind("<<Decrement>>", clear_selection)
        spinbox.bind("<ButtonRelease-1>", clear_selection)
        spinbox.bind("<KeyRelease>", clear_selection)
        spinbox.bind("<FocusIn>", clear_selection)

        var.trace_add(
            "write",
            lambda *args, state_key=key, tk_var=var: self._update_single_value(state_key, tk_var)
        )

    def _update_single_value(self, key, var):
        if self.loading_state:
            return

        value = var.get()

        if value.strip() == "":
            return

        try:
            if key in ["additionalTangentialRefine", "additionalAxialRefine"]:
                self.app.state.set(key, int(float(value)))
            else:
                self.app.state.set(key, float(value))
        except ValueError:
            pass

    def clear_all(self):
        self.app.state.reset_mesh_tuning()
        self.load_state()

    def load_state(self):
        self.loading_state = True

        for key, var in self.vars.items():
            var.set(str(self.app.state.get(key, "")))

        self.loading_state = False

        
class AdvancedPage(BasePage):
    def __init__(self, parent, app):
        super().__init__(parent, app, title="Advanced")

        self.loading_state = False
        self.vars = {}
        self.spinboxes = []

        self._build_page()
        self.load_state()

    def _build_page(self):
        self.body.columnconfigure(0, weight=1)

        content = ttk.Frame(self.body, style="Content.TFrame")
        content.grid(row=0, column=0, sticky="nw")

        content.columnconfigure(1, weight=1)
        content.columnconfigure(3, weight=1)

        # --------------------------------------------------
        # Blade cut / arclength controls
        # --------------------------------------------------
        ttk.Label(
            content,
            text="Blade Cut / Arclength Controls",
            style="Section.TLabel"
        ).grid(
            row=0,
            column=0,
            columnspan=4,
            sticky="w",
            pady=(0, 18)
        )

        arclength_fields = [
            ("Blade arclength cutoff fraction", "percentVal", 0.0, 1.0, 0.001),
            ("Leading-edge non-cut arclength fraction", "percentValNonCutLE", 0.0, 1.0, 0.001),
            ("Trailing-edge non-cut arclength fraction", "percentValNonCutTE", 0.0, 1.0, 0.001),
        ]

        for index, (label_text, key, min_val, max_val, step) in enumerate(arclength_fields):
            row = 1 + index

            self._add_spinbox_field(
                parent=content,
                row=row,
                label_col=0,
                entry_col=1,
                label_text=label_text,
                key=key,
                min_val=min_val,
                max_val=max_val,
                step=step,
            )

        # --------------------------------------------------
        # Angle constraint controls
        # --------------------------------------------------
        angle_start_row = 5

        ttk.Separator(content, orient="horizontal").grid(
            row=angle_start_row,
            column=0,
            columnspan=4,
            sticky="ew",
            pady=(24, 18)
        )

        ttk.Label(
            content,
            text="Angle Constraint Controls",
            style="Section.TLabel"
        ).grid(
            row=angle_start_row + 1,
            column=0,
            columnspan=4,
            sticky="w",
            pady=(0, 18)
        )

        angle_fields = [
            ("Curve angle constraint (deg)", "angConstraintCurves", 0.0, 90.0, 1.0),
            ("Offset angle constraint (deg)", "angConstraintOffsets", 0.0, 90.0, 1.0),
        ]

        for index, (label_text, key, min_val, max_val, step) in enumerate(angle_fields):
            row = angle_start_row + 2 + index

            self._add_spinbox_field(
                parent=content,
                row=row,
                label_col=0,
                entry_col=1,
                label_text=label_text,
                key=key,
                min_val=min_val,
                max_val=max_val,
                step=step,
            )

        # --------------------------------------------------
        # Clear All button on the actual right side of page
        # --------------------------------------------------
        action_bar = ttk.Frame(self.body, style="Content.TFrame")
        action_bar.grid(
            row=1,
            column=0,
            sticky="e",
            padx=(0, 10),
            pady=(24, 0)
        )

        ttk.Button(
            action_bar,
            text="Clear All",
            takefocus=False,
            command=self.clear_all
        ).grid(row=0, column=0)

    def _add_spinbox_field(self, parent, row, label_col, entry_col, label_text, key, min_val, max_val, step):
        label = self._create_parameter_label(parent, label_text, key)
        label.grid(
            row=row,
            column=label_col,
            sticky="w",
            padx=(0, 24),
            pady=12
        )

        var = tk.StringVar()
        self.vars[key] = var

        spinbox = ttk.Spinbox(
            parent,
            from_=min_val,
            to=max_val,
            increment=step,
            textvariable=var,
            width=16,
            takefocus=False,
        )
        spinbox.grid(
            row=row,
            column=entry_col,
            sticky="w",
            padx=(0, 32),
            pady=12
        )

        self.spinboxes.append(spinbox)

        def clear_selection(event):
            widget = event.widget
            widget.after(10, lambda: widget.selection_clear())
            widget.after(20, lambda: widget.icursor("end"))

        spinbox.bind("<<Increment>>", clear_selection)
        spinbox.bind("<<Decrement>>", clear_selection)
        spinbox.bind("<ButtonRelease-1>", clear_selection)
        spinbox.bind("<KeyRelease>", clear_selection)
        spinbox.bind("<FocusIn>", clear_selection)

        var.trace_add(
            "write",
            lambda *args, state_key=key, tk_var=var: self._update_single_value(state_key, tk_var)
        )

    def _update_single_value(self, key, var):
        if self.loading_state:
            return

        value = var.get()

        if value.strip() == "":
            return

        try:
            self.app.state.set(key, float(value))
        except ValueError:
            pass

    def clear_all(self):
        self.app.state.reset_advanced()
        self.load_state()

    def load_state(self):
        self.loading_state = True

        for key, var in self.vars.items():
            var.set(str(self.app.state.get(key, "")))

        self.loading_state = False


class RunPage(BasePage):
    def __init__(self, parent, app):
        super().__init__(parent, app, title="Run")
        self.process = None
        self.output_queue = queue.Queue()
        self.build_page()
        self.append_console("[Ready] Configure inputs, then run mesh generation.\n")

    def build_page(self):
        self.body.columnconfigure(0, weight=1)
        self.body.rowconfigure(1, weight=1)

        run_card = tk.Frame(
            self.body,
            bg="white",
            highlightbackground="#d8e3ef",
            highlightthickness=1,
            bd=0
        )
        run_card.grid(row=0, column=0, sticky="ew", pady=(0, 24))
        run_card.columnconfigure(0, weight=1)

        run_header = tk.Label(
            run_card,
            text="Run Configuration",
            font=("Segoe UI", 15, "bold"),
            fg=MAIN_BLUE,
            bg="white"
        )
        run_header.grid(row=0, column=0, sticky="w", padx=24, pady=(18, 4))

        run_description = tk.Label(
            run_card,
            text="Start mesh surface generation using the current file paths and parameter settings.",
            font=("Segoe UI", 10),
            fg="#4d5f73",
            bg="white"
        )
        run_description.grid(row=1, column=0, sticky="w", padx=24, pady=(0, 18))

        button_frame = tk.Frame(run_card, bg="white")
        button_frame.grid(row=0, column=1, rowspan=2, sticky="e", padx=24, pady=18)

        self.run_button = tk.Button(
            button_frame,
            text="▶  Generate Surfaces",
            command=self.run_backend,
            font=("Segoe UI", 10, "bold"),
            fg="white",
            bg=MAIN_BLUE,
            activeforeground="white",
            activebackground="#003f73",
            relief="flat",
            bd=0,
            padx=24,
            pady=10,
            cursor="hand2",
            width=22
        )
        self.run_button.grid(row=0, column=0, sticky="e", pady=(0, 8))

        self.bash_button = tk.Button(
            button_frame,
            text="▶  Build OpenFOAM Mesh",
            command=self.run_bash_script,
            font=("Segoe UI", 10, "bold"),
            fg="white",
            bg=MAIN_BLUE,
            activeforeground="white",
            activebackground="#003f73",
            disabledforeground="white",
            relief="flat",
            bd=0,
            padx=24,
            pady=10,
            cursor="hand2",
            width=22
        )
        self.bash_button.grid(row=1, column=0, sticky="e")

        console_card = tk.Frame(
            self.body,
            bg="white",
            highlightbackground="#d8e3ef",
            highlightthickness=1,
            bd=0
        )
        console_card.grid(row=1, column=0, sticky="nsew")
        console_card.columnconfigure(0, weight=1)
        console_card.rowconfigure(1, weight=1)

        console_header_frame = tk.Frame(console_card, bg="white")
        console_header_frame.grid(row=0, column=0, sticky="ew", padx=24, pady=(18, 10))
        console_header_frame.columnconfigure(0, weight=1)

        console_title = tk.Label(
            console_header_frame,
            text="Console Output",
            font=("Segoe UI", 14, "bold"),
            fg=MAIN_BLUE,
            bg="white"
        )
        console_title.grid(row=0, column=0, sticky="w")

        console_frame = tk.Frame(console_card, bg="white")
        console_frame.grid(row=1, column=0, sticky="nsew", padx=24, pady=(0, 16))
        console_frame.columnconfigure(0, weight=1)
        console_frame.rowconfigure(0, weight=1)

        self.console = tk.Text(
            console_frame,
            height=18,
            wrap="word",
            font=("Consolas", 10),
            bg="#fbfdff",
            fg="#1f2933",
            insertbackground=MAIN_BLUE,
            relief="flat",
            bd=0,
            padx=14,
            pady=12
        )
        self.console.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(
            console_frame,
            orient="vertical",
            command=self.console.yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.console.configure(yscrollcommand=scrollbar.set)

        action_frame = tk.Frame(console_card, bg="white")
        action_frame.grid(row=2, column=0, sticky="e", padx=24, pady=(0, 18))

        clear_button = tk.Button(
            action_frame,
            text="Clear Console",
            command=self.clear_console,
            font=("Segoe UI", 9),
            fg=MAIN_BLUE,
            bg="#eef7ff",
            activeforeground=MAIN_BLUE,
            activebackground="#d6ecff",
            relief="flat",
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2"
        )
        clear_button.grid(row=0, column=0, padx=(0, 10))

        save_button = tk.Button(
            action_frame,
            text="Save Log",
            command=self.save_log,
            font=("Segoe UI", 9),
            fg=MAIN_BLUE,
            bg="#eef7ff",
            activeforeground=MAIN_BLUE,
            activebackground="#d6ecff",
            relief="flat",
            bd=0,
            padx=16,
            pady=8,
            cursor="hand2"
        )
        save_button.grid(row=0, column=1)

    def append_console(self, text):
        self.console.insert("end", text)
        self.console.see("end")

    def clear_console(self):
        self.console.delete("1.0", "end")

    def save_log(self):
        output_path = self.app.state.get("outputPath", "")

        if output_path:
            initial_dir = output_path
        else:
            initial_dir = os.path.dirname(__file__)

        filename = filedialog.asksaveasfilename(
            title="Save console log",
            initialdir=initial_dir,
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filename:
            with open(filename, "w", encoding="utf-8") as file:
                file.write(self.console.get("1.0", "end"))

    def set_running_state(self, is_running, run_type="python"):
        if is_running:
            self.run_button.configure(state="disabled")
            self.bash_button.configure(state="disabled")

            if run_type == "bash":
                self.bash_button.configure(
                    text="Running...",
                    fg="white",
                    disabledforeground="white",
                    bg="#003f73"
                )
            else:
                self.run_button.configure(
                    text="Running...",
                    fg="white",
                    disabledforeground="white",
                    bg="#003f73"
                )
        else:
            self.run_button.configure(
                text="▶  Generate Surfaces",
                state="normal",
                fg="white",
                bg=MAIN_BLUE
            )
            self.bash_button.configure(
                text="▶  Build OpenFOAM Mesh",
                state="normal",
                fg="white",
                bg=MAIN_BLUE,
                disabledforeground="white"
            )

    def _candidate_backend_paths(self):
        gui_folder = os.path.dirname(os.path.abspath(__file__))
        project_folder = os.path.abspath(os.path.join(gui_folder, os.pardir))
        cwd = os.getcwd()

        return [
            os.path.join(gui_folder, "bladePassageSurfaceGenerator_v2.py"),
            os.path.join(project_folder, "bladePassageSurfaceGenerator_v2.py"),
            os.path.join(project_folder, "Python", "bladePassageSurfaceGenerator_v2.py"),
            os.path.join(cwd, "bladePassageSurfaceGenerator_v2.py"),
            os.path.join(cwd, "Python", "bladePassageSurfaceGenerator_v2.py"),
            os.path.join(os.path.abspath(os.path.join(cwd, os.pardir)), "Python", "bladePassageSurfaceGenerator_v2.py"),
        ]

    def _find_backend_script(self):
        for candidate in self._candidate_backend_paths():
            if os.path.exists(candidate):
                return os.path.abspath(candidate)

        gui_folder = os.path.dirname(os.path.abspath(__file__))
        project_folder = os.path.abspath(os.path.join(gui_folder, os.pardir))
        ignored_folders = {".git", "__pycache__", ".venv", "venv", "env"}

        for root, dirs, files in os.walk(project_folder):
            dirs[:] = [folder for folder in dirs if folder not in ignored_folders]
            if "bladePassageSurfaceGenerator_v2.py" in files:
                return os.path.join(root, "bladePassageSurfaceGenerator_v2.py")

        return None

    def _validate_inputs(self):
        data_path = self.app.state.get("dataPath", "")
        output_path = self.app.state.get("outputPath", "")
        hub_file = self.app.state.get("hubFileName", "")
        casing_file = self.app.state.get("casFileName", "")
        blade_file = self.app.state.get("bladeCurveFile", "")

        missing = []

        if not data_path:
            missing.append("Input data folder")
        if not hub_file:
            missing.append("Hub curve file")
        if not casing_file:
            missing.append("Casing curve file")
        if not blade_file:
            missing.append("Blade curve file")
        if not output_path:
            missing.append("Output data folder")

        if missing:
            messagebox.showerror(
                "Missing Run Inputs",
                "Please fill in the following fields before running:\n\n" + "\n".join(f"• {item}" for item in missing)
            )
            return False

        if not os.path.isdir(data_path):
            messagebox.showerror("Invalid Input Folder", f"The input data folder does not exist:\n\n{data_path}")
            return False

        for label, filename in [
            ("Hub curve file", hub_file),
            ("Casing curve file", casing_file),
            ("Blade curve file", blade_file),
        ]:
            full_path = os.path.join(data_path, filename)
            if not os.path.exists(full_path):
                messagebox.showerror(
                    "Missing Curve File",
                    f"{label} was not found in the input data folder:\n\n{full_path}"
                )
                return False

        return True

    def _build_run_config(self):
        config = self.app.state.values.copy()

        output_path = config.get("outputPath", "")
        if output_path:
            os.makedirs(output_path, exist_ok=True)

        return config

    def _write_run_config(self, config):
        output_path = config.get("outputPath", "")
        config_path = os.path.join(output_path, "apt_grid_run_config.json")

        with open(config_path, "w", encoding="utf-8") as file:
            json.dump(config, file, indent=4)

        return config_path

    def _print_run_summary(self, config, config_path, backend_script):
        self.append_console("\n" + "=" * 70 + "\n")
        self.append_console("[GUI Input Summary]\n")
        self.append_console("=" * 70 + "\n")
        self.append_console(f"Config file: {config_path}\n")
        self.append_console(f"Backend script: {backend_script}\n\n")

        for key, value in config.items():
            self.append_console(f"{key}: {value}\n")

        self.append_console("=" * 70 + "\n\n")

    def _get_project_folder(self):
        gui_folder = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(gui_folder, os.pardir))

    def _candidate_bash_script_paths(self):
        project_folder = self._get_project_folder()

        return [
            os.path.join(project_folder, "runtest.sh"),
            os.path.join(project_folder, "passageMeshes", "runtest.sh"),
            os.path.join(project_folder, "passageMeshes", "multipassagetest.sh"),
        ]

    def _find_bash_script(self):
        for candidate in self._candidate_bash_script_paths():
            if os.path.exists(candidate):
                return os.path.abspath(candidate)

        return None

    def _find_multipassage_script(self):
        project_folder = self._get_project_folder()
        candidate = os.path.join(project_folder, "passageMeshes", "multipassagetest.sh")
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
        return None

    def _get_bash_working_folder(self, bash_script):
        project_folder = self._get_project_folder()
        script_name = os.path.basename(bash_script)

        if script_name == "multipassagetest.sh":
            return os.path.dirname(bash_script)

        return project_folder

    def _windows_to_wsl_path(self, windows_path):
        try:
            completed = subprocess.run(
                ["wsl", "wslpath", "-a", windows_path],
                capture_output=True,
                text=True,
                check=True
            )
            return completed.stdout.strip()
        except Exception:
            drive, path_tail = os.path.splitdrive(os.path.abspath(windows_path))
            drive_letter = drive.replace(":", "").lower()
            path_tail = path_tail.replace("\\", "/")
            return f"/mnt/{drive_letter}{path_tail}"

    def _is_passage_folder_name(self, folder_name):
        return folder_name.startswith("passage") and folder_name[7:].isdigit()

    def _passage_sort_key(self, folder_name):
        try:
            return int(folder_name[7:])
        except ValueError:
            return 0

    def _get_output_data_folder(self):
        output_path = self.app.state.get("outputPath", "")
        if output_path:
            return os.path.abspath(output_path)

        return os.path.join(self._get_project_folder(), "outputData")

    def _find_generated_passages(self):
        output_folder = self._get_output_data_folder()

        if not os.path.isdir(output_folder):
            return []

        passages = []
        for folder_name in os.listdir(output_folder):
            folder_path = os.path.join(output_folder, folder_name)
            if os.path.isdir(folder_path) and self._is_passage_folder_name(folder_name):
                passages.append(folder_name)

        return sorted(passages, key=self._passage_sort_key)

    def _shell_quote(self, value):
        return "'" + str(value).replace("'", "'\\''") + "'"

    def _posix_join(self, *parts):
        cleaned_parts = []

        for index, part in enumerate(parts):
            part = str(part).replace("\\", "/")

            if index == 0:
                cleaned_parts.append(part.rstrip("/"))
            else:
                cleaned_parts.append(part.strip("/"))

        return "/".join(cleaned_parts)

    def _openfoam_source_command(self):
        return (
            "if [ -f /opt/openfoam12/etc/bashrc ]; then . /opt/openfoam12/etc/bashrc; "
            "elif [ -f /usr/lib/openfoam/openfoam12/etc/bashrc ]; then . /usr/lib/openfoam/openfoam12/etc/bashrc; "
            "else true; fi"
        )

    def _single_passage_shell_command(self, project_folder, output_folder, passage_name="passage0"):
        passage_meshes_folder = self._posix_join(project_folder, "passageMeshes")

        return " && ".join([
            self._openfoam_source_command(),
            f"cd {self._shell_quote(passage_meshes_folder)}",
            f"rm -rf {self._shell_quote(passage_name)}",
            f"mkdir -p {self._shell_quote(passage_name)}",
            f"cp -r template/* {self._shell_quote(passage_name)}/",
            f"mkdir -p {self._shell_quote(passage_name)}/constant/geometry",
            f"mkdir -p {self._shell_quote(passage_name)}/system",
            f"cp {self._shell_quote(self._posix_join(output_folder, passage_name))}/*.stl {self._shell_quote(passage_name)}/constant/geometry/",
            f"cp {self._shell_quote(self._posix_join(output_folder, passage_name, 'passageParameters'))} {self._shell_quote(passage_name)}/system/",
            f"cd {self._shell_quote(passage_name)}",
            "sed -i 's/\\r$//' system/geomUpdate.sh 2>/dev/null || true",
            "sed -i 's/#eval{\\([^}]*\\)}/#calc \"\\1\"/g' system/blockMeshDict 2>/dev/null || true",
            "bash ./geomUpdate.sh",
            "sed -i 's/#eval{\\([^}]*\\)}/#calc \"\\1\"/g' system/blockMeshDict 2>/dev/null || true",
            "blockMesh",
            "checkMesh",
        ])

    def _wsl_single_passage_shell_command(self, source_passage_meshes_folder, source_output_folder, passage_name="passage0"):
        # Use a real multi-line shell script instead of one long command joined by &&.
        # This prevents an earlier failed cd/cp from being hidden by "|| true" and
        # accidentally running blockMesh from the original Windows-mounted folder.
        source_passage_meshes_folder = source_passage_meshes_folder.replace("\\", "/")
        source_output_folder = source_output_folder.replace("\\", "/")
        safe_passage_name = str(passage_name).replace("'", "")

        return rf"""
# This script is written to a real .sh file and then executed by WSL.
# That avoids bash -c quoting/variable-expansion problems from Windows.
# Use a GridWorks-specific variable name to avoid any OpenFOAM/internal conflicts.
if ! command -v blockMesh >/dev/null 2>&1; then
    if [ -f "$HOME/.bashrc" ]; then
        . "$HOME/.bashrc" || true
    fi
fi

if ! command -v blockMesh >/dev/null 2>&1; then
    if [ -f /opt/openfoam12/etc/bashrc ]; then
        . /opt/openfoam12/etc/bashrc || true
    elif [ -f /usr/lib/openfoam/openfoam12/etc/bashrc ]; then
        . /usr/lib/openfoam/openfoam12/etc/bashrc || true
    fi
fi

if ! command -v blockMesh >/dev/null 2>&1; then
    echo "[Error] blockMesh was not found after loading OpenFOAM."
    exit 127
fi

set -e

GW_RUN_ROOT="/tmp/GridWorksOpenFOAM_gridworks"
SRC_PASSAGE_MESHES={self._shell_quote(source_passage_meshes_folder)}
SRC_OUTPUT={self._shell_quote(source_output_folder)}
PASSAGE_NAME={self._shell_quote(safe_passage_name)}

rm -rf "$GW_RUN_ROOT"
mkdir -p "$GW_RUN_ROOT/passageMeshes" "$GW_RUN_ROOT/outputData"

cp -a "$SRC_PASSAGE_MESHES/." "$GW_RUN_ROOT/passageMeshes/"
cp -a "$SRC_OUTPUT/." "$GW_RUN_ROOT/outputData/"

cd "$GW_RUN_ROOT/passageMeshes"
echo "[OpenFOAM staging folder] $(pwd)"

find . -name "*.sh" -exec sed -i 's/\r$//' {{}} \;
mkdir -p template/constant/geometry template/system

if [ -f template/system/blockMeshDict ]; then
    sed -i 's/#eval{{\([^}}]*\)}}/#calc "\1"/g' template/system/blockMeshDict
fi

if [ ! -d "$GW_RUN_ROOT/outputData/$PASSAGE_NAME" ]; then
    echo "[Error] Missing generated folder: $GW_RUN_ROOT/outputData/$PASSAGE_NAME"
    exit 2
fi

if ! ls "$GW_RUN_ROOT/outputData/$PASSAGE_NAME"/*.stl >/dev/null 2>&1; then
    echo "[Error] No STL files found in: $GW_RUN_ROOT/outputData/$PASSAGE_NAME"
    exit 2
fi

if [ ! -f "$GW_RUN_ROOT/outputData/$PASSAGE_NAME/passageParameters" ]; then
    echo "[Error] Missing passageParameters in: $GW_RUN_ROOT/outputData/$PASSAGE_NAME"
    exit 2
fi

rm -rf "$PASSAGE_NAME"
mkdir -p "$PASSAGE_NAME"
cp -a template/. "$PASSAGE_NAME/"
mkdir -p "$PASSAGE_NAME/constant/geometry" "$PASSAGE_NAME/system"
cp "$GW_RUN_ROOT/outputData/$PASSAGE_NAME"/*.stl "$PASSAGE_NAME/constant/geometry/"
cp "$GW_RUN_ROOT/outputData/$PASSAGE_NAME/passageParameters" "$PASSAGE_NAME/system/"

cd "$PASSAGE_NAME"

if [ ! -f ./geomUpdate.sh ]; then
    echo "[Error] Missing geomUpdate.sh in case folder: $(pwd)"
    exit 2
fi

bash ./geomUpdate.sh

if [ -f system/blockMeshDict ]; then
    sed -i 's/#eval{{\([^}}]*\)}}/#calc "\1"/g' system/blockMeshDict
fi

unset FOAM_CASE
export FOAM_CASE="$(pwd)"
echo "[OpenFOAM case folder] $(pwd)"

blockMesh
CHECKMESH_STATUS=0
checkMesh || CHECKMESH_STATUS=$?

# Save the finished OpenFOAM mesh back to the original APT-Grid folder,
# matching the original/manual workflow output location:
# APT-Grid/passageMeshes/passage0/constant/polyMesh
# Important: copy files manually with cat instead of cp -a/cp -R so WSL does
# not try to preserve Linux permissions/timestamps on the Windows drive.
DEST_CASE="$SRC_PASSAGE_MESHES/$PASSAGE_NAME"
SRC_CASE="$GW_RUN_ROOT/passageMeshes/$PASSAGE_NAME"
DEST_POLYMESH="$DEST_CASE/constant/polyMesh"
echo "[Saving OpenFOAM mesh] $DEST_POLYMESH"

copy_plain_tree() {{
    SRC_DIR="$1"
    DST_DIR="$2"
    if [ ! -d "$SRC_DIR" ]; then
        return 0
    fi
    rm -rf "$DST_DIR"
    mkdir -p "$DST_DIR"
    (cd "$SRC_DIR" && find . -type d -print) | while IFS= read -r d; do
        mkdir -p "$DST_DIR/$d"
    done
    (cd "$SRC_DIR" && find . -type f -print) | while IFS= read -r f; do
        mkdir -p "$DST_DIR/$(dirname "$f")"
        cat "$SRC_DIR/$f" > "$DST_DIR/$f"
    done
}}

# Keep the useful case pieces, but skip OpenFOAM dynamicCode build artifacts.
copy_plain_tree "$SRC_CASE/constant/polyMesh" "$DEST_CASE/constant/polyMesh"
copy_plain_tree "$SRC_CASE/constant/geometry" "$DEST_CASE/constant/geometry"
copy_plain_tree "$SRC_CASE/system" "$DEST_CASE/system"
echo "[Saved OpenFOAM mesh] $DEST_POLYMESH"

if [ "$CHECKMESH_STATUS" -ne 0 ]; then
    echo "[Warning] checkMesh returned status $CHECKMESH_STATUS. Mesh files were still saved."
fi
exit 0
""".strip()

    def _wsl_multipassage_shell_command(self, source_passage_meshes_folder, source_output_folder):
        # Stage everything inside Linux first, then run the existing two-passage script.
        source_passage_meshes_folder = source_passage_meshes_folder.replace("\\", "/")
        source_output_folder = source_output_folder.replace("\\", "/")

        return rf"""
# This script is written to a real .sh file and then executed by WSL.
# That avoids bash -c quoting/variable-expansion problems from Windows.
# Use a GridWorks-specific variable name to avoid any OpenFOAM/internal conflicts.
if ! command -v blockMesh >/dev/null 2>&1; then
    if [ -f "$HOME/.bashrc" ]; then
        . "$HOME/.bashrc" || true
    fi
fi

if ! command -v blockMesh >/dev/null 2>&1; then
    if [ -f /opt/openfoam12/etc/bashrc ]; then
        . /opt/openfoam12/etc/bashrc || true
    elif [ -f /usr/lib/openfoam/openfoam12/etc/bashrc ]; then
        . /usr/lib/openfoam/openfoam12/etc/bashrc || true
    fi
fi

if ! command -v blockMesh >/dev/null 2>&1; then
    echo "[Error] blockMesh was not found after loading OpenFOAM."
    exit 127
fi

set -e

GW_RUN_ROOT="/tmp/GridWorksOpenFOAM_gridworks"
SRC_PASSAGE_MESHES={self._shell_quote(source_passage_meshes_folder)}
SRC_OUTPUT={self._shell_quote(source_output_folder)}

rm -rf "$GW_RUN_ROOT"
mkdir -p "$GW_RUN_ROOT/passageMeshes" "$GW_RUN_ROOT/outputData"

cp -a "$SRC_PASSAGE_MESHES/." "$GW_RUN_ROOT/passageMeshes/"
cp -a "$SRC_OUTPUT/." "$GW_RUN_ROOT/outputData/"

cd "$GW_RUN_ROOT/passageMeshes"
echo "[OpenFOAM staging folder] $(pwd)"

find . -name "*.sh" -exec sed -i 's/\r$//' {{}} \;
mkdir -p template/constant/geometry template/system

if [ -f template/system/blockMeshDict ]; then
    sed -i 's/#eval{{\([^}}]*\)}}/#calc "\1"/g' template/system/blockMeshDict
fi

if [ ! -f multipassagetest.sh ]; then
    echo "[Error] Missing multipassagetest.sh in: $(pwd)"
    exit 2
fi

unset FOAM_CASE
MESH_STATUS=0
bash multipassagetest.sh || MESH_STATUS=$?

# Save generated OpenFOAM passage cases back to the original APT-Grid folder.
# Important: copy files manually with cat instead of cp -a/cp -R so Linux/WSL does
# not try to preserve Linux permissions/timestamps on filesystems that may not support them.
copy_plain_tree() {{
    SRC_DIR="$1"
    DST_DIR="$2"
    if [ ! -d "$SRC_DIR" ]; then
        return 0
    fi
    rm -rf "$DST_DIR"
    mkdir -p "$DST_DIR"
    (cd "$SRC_DIR" && find . -type d -print) | while IFS= read -r d; do
        mkdir -p "$DST_DIR/$d"
    done
    (cd "$SRC_DIR" && find . -type f -print) | while IFS= read -r f; do
        mkdir -p "$DST_DIR/$(dirname "$f")"
        cat "$SRC_DIR/$f" > "$DST_DIR/$f"
    done
}}

for CASE_DIR in passage*; do
    if [ -d "$CASE_DIR" ]; then
        echo "[Saving OpenFOAM case] $SRC_PASSAGE_MESHES/$CASE_DIR"
        SRC_CASE="$GW_RUN_ROOT/passageMeshes/$CASE_DIR"
        DEST_CASE="$SRC_PASSAGE_MESHES/$CASE_DIR"
        mkdir -p "$DEST_CASE/constant"
        # Keep the useful case pieces, but skip OpenFOAM dynamicCode build artifacts.
        copy_plain_tree "$SRC_CASE/constant/polyMesh" "$DEST_CASE/constant/polyMesh"
        copy_plain_tree "$SRC_CASE/constant/geometry" "$DEST_CASE/constant/geometry"
        copy_plain_tree "$SRC_CASE/system" "$DEST_CASE/system"
    fi
done

echo "[Saved OpenFOAM cases] $SRC_PASSAGE_MESHES"
if [ "$MESH_STATUS" -ne 0 ]; then
    echo "[Warning] multipassagetest.sh returned status $MESH_STATUS. Any generated case files were still saved."
fi
exit 0
""".strip()

    def _write_shell_script(self, shell_command, script_filename):
        """Write a temporary shell script to outputData and return its local path."""
        output_folder = self._get_output_data_folder()
        os.makedirs(output_folder, exist_ok=True)
        script_path = os.path.join(output_folder, script_filename)

        with open(script_path, "w", encoding="utf-8", newline="\n") as script_file:
            script_file.write("#!/usr/bin/env bash\n")
            script_file.write(shell_command.strip())
            script_file.write("\n")

        try:
            os.chmod(script_path, 0o755)
        except OSError:
            pass

        return script_path

    def _write_wsl_shell_script(self, shell_command, script_filename):
        """Write a temporary WSL shell script to outputData and return its WSL path."""
        script_path = self._write_shell_script(shell_command, script_filename)
        return self._windows_to_wsl_path(script_path)

    def _build_multipassage_command(self):
        project_folder = self._get_project_folder()
        working_folder = os.path.join(project_folder, "passageMeshes")
        bash_script = self._find_multipassage_script()

        if bash_script is None:
            return None, working_folder, None

        if os.name == "nt":
            if shutil.which("wsl"):
                wsl_passage_meshes_folder = self._windows_to_wsl_path(working_folder)
                wsl_output_folder = self._windows_to_wsl_path(self._get_output_data_folder())
                shell_command = self._wsl_multipassage_shell_command(wsl_passage_meshes_folder, wsl_output_folder)
                wsl_script_path = self._write_wsl_shell_script(shell_command, "gridworks_openfoam_multipassage.sh")
                return ["wsl", "bash", wsl_script_path], working_folder, bash_script

            bash_exe = shutil.which("bash")
            if bash_exe:
                shell_command = self._wsl_multipassage_shell_command(
                    working_folder.replace("\\", "/"),
                    self._get_output_data_folder().replace("\\", "/")
                )
                script_path = self._write_shell_script(shell_command, "gridworks_openfoam_multipassage.sh")
                return [bash_exe, script_path], working_folder, bash_script

            return None, working_folder, bash_script

        # Native Linux/macOS path: run OpenFOAM directly with bash, no WSL path conversion.
        shell_command = self._wsl_multipassage_shell_command(
            working_folder.replace("\\", "/"),
            self._get_output_data_folder().replace("\\", "/")
        )
        script_path = self._write_shell_script(shell_command, "gridworks_openfoam_multipassage.sh")
        return ["bash", script_path], working_folder, bash_script

    def _build_bash_command(self, bash_script, working_folder):
        if os.name == "nt":
            if shutil.which("wsl"):
                wsl_working_folder = self._windows_to_wsl_path(working_folder)
                wsl_script = self._windows_to_wsl_path(bash_script)
                script_name = os.path.basename(wsl_script)

                if os.path.dirname(bash_script) == working_folder:
                    shell_command = (
                        f'cd "{wsl_working_folder}" || exit 1; '
                        'find . -name "*.sh" -exec sed -i \'s/\\r$//\' {} \\; 2>/dev/null || true; '
                        'mkdir -p template/constant/geometry template/system; '
                        'sed -i \'s/#eval{\\([^}]*\\)}/#calc "\\1"/g\' template/system/blockMeshDict 2>/dev/null || true; '
                        f'bash "{script_name}"'
                    )
                else:
                    shell_command = (
                        f'cd "{wsl_working_folder}" || exit 1; '
                        f'sed -i \'s/\\r$//\' "{wsl_script}" 2>/dev/null || true; '
                        f'bash "{wsl_script}"'
                    )

                return ["wsl", "bash", "-lc", shell_command]

            bash_exe = shutil.which("bash")
            if bash_exe:
                return [bash_exe, bash_script]

            return None

        return ["bash", bash_script]

    def _build_single_passage_command(self, passage_name="passage0"):
        project_folder = self._get_project_folder()
        working_folder = os.path.join(project_folder, "passageMeshes")

        if os.name == "nt":
            if shutil.which("wsl"):
                wsl_passage_meshes_folder = self._windows_to_wsl_path(working_folder)
                wsl_output_folder = self._windows_to_wsl_path(self._get_output_data_folder())
                shell_command = self._wsl_single_passage_shell_command(
                    wsl_passage_meshes_folder,
                    wsl_output_folder,
                    passage_name
                )
                wsl_script_path = self._write_wsl_shell_script(shell_command, "gridworks_openfoam_single.sh")
                return ["wsl", "bash", wsl_script_path], working_folder

            bash_exe = shutil.which("bash")
            if bash_exe:
                shell_command = self._wsl_single_passage_shell_command(
                    working_folder.replace("\\", "/"),
                    self._get_output_data_folder().replace("\\", "/"),
                    passage_name
                )
                script_path = self._write_shell_script(shell_command, "gridworks_openfoam_single.sh")
                return [bash_exe, script_path], working_folder

            return None, working_folder

        # Native Linux/macOS path: run OpenFOAM directly with bash, no WSL path conversion.
        shell_command = self._wsl_single_passage_shell_command(
            working_folder.replace("\\", "/"),
            self._get_output_data_folder().replace("\\", "/"),
            passage_name
        )
        script_path = self._write_shell_script(shell_command, "gridworks_openfoam_single.sh")
        return ["bash", script_path], working_folder

    def _print_bash_summary(self, workflow_name, passages, working_folder, command, bash_script=None):
        self.append_console("\n" + "=" * 70 + "\n")
        self.append_console("[OpenFOAM Mesh Summary]\n")
        self.append_console("=" * 70 + "\n")
        self.append_console(f"Workflow: {workflow_name}\n")
        self.append_console(f"Detected passages: {', '.join(passages)}\n")
        if bash_script:
            self.append_console(f"Bash script: {bash_script}\n")
        self.append_console(f"Working folder: {working_folder}\n")
        self.append_console(f"Command: {' '.join(command)}\n")
        self.append_console("=" * 70 + "\n\n")

    def run_bash_script(self):
        if self.process is not None and self.process.poll() is None:
            messagebox.showwarning(
                "Run Already Active",
                "A run is already active."
            )
            return

        passages = self._find_generated_passages()

        if not passages:
            messagebox.showerror(
                "No Generated Passages Found",
                "Could not find any generated passage folders.\n\n"
                "Run Generate Surfaces first and make sure the output folder contains passage0."
            )
            return

        if "passage0" not in passages:
            messagebox.showerror(
                "Missing passage0",
                "The output folder contains passage folders, but passage0 was not found.\n\n"
                "The OpenFOAM workflow expects passage0 to exist."
            )
            return

        if len(passages) == 1:
            workflow_name = "Single passage OpenFOAM build"
            bash_script = None
            command, working_folder = self._build_single_passage_command("passage0")
        else:
            workflow_name = "Two-passage OpenFOAM build"
            command, working_folder, bash_script = self._build_multipassage_command()

            if bash_script is None:
                messagebox.showerror(
                    "Multipassage Script Not Found",
                    "Multiple passage folders were found, but passageMeshes/multipassagetest.sh was not found."
                )
                return

            if len(passages) > 2:
                messagebox.showwarning(
                    "Only Two-Passage Script Available",
                    "More than two generated passage folders were found.\n\n"
                    "The current multipassagetest.sh script is hard-coded for passage0 and passage1, "
                    "so only those two will be processed by this button."
                )

        if command is None:
            messagebox.showerror(
                "Bash Not Available",
                "Could not find WSL or bash on this computer.\n\n"
                "On Windows, this step should usually be run through WSL with OpenFOAM installed."
            )
            return

        self.set_running_state(True, "bash")
        self._print_bash_summary(workflow_name, passages, working_folder, command, bash_script)
        self.append_console("[OpenFOAM Mesh Build Started]\n")

        thread = threading.Thread(
            target=self._run_bash_thread,
            args=(command, working_folder),
            daemon=True
        )
        thread.start()

        self.after(100, self._process_output_queue)

    def _run_bash_thread(self, command, working_folder):
        try:
            self.output_queue.put(f"[Command] {' '.join(command)}\n\n")

            process_cwd = None if command and command[0] == "wsl" else working_folder

            self.process = subprocess.Popen(
                command,
                cwd=process_cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            if self.process.stdout is not None:
                for line in self.process.stdout:
                    self.output_queue.put(line)

            return_code = self.process.wait()

            if return_code == 0:
                self.output_queue.put("\n[Run Complete] OpenFOAM mesh build finished successfully.\n")
            else:
                self.output_queue.put(f"\n[Run Failed] OpenFOAM mesh build exited with code {return_code}.\n")

        except Exception as error:
            self.output_queue.put(f"\n[Error] {error}\n")

        finally:
            self.process = None
            self.after(0, lambda: self.set_running_state(False))


    def run_backend(self):
        if self.process is not None and self.process.poll() is None:
            messagebox.showwarning(
                "Run Already Active",
                "A mesh generation run is already active."
            )
            return

        if not self._validate_inputs():
            return

        backend_script = self._find_backend_script()
        if backend_script is None:
            messagebox.showerror(
                "Backend Not Found",
                "Could not find bladePassageSurfaceGenerator_v2.py.\n\n"
                "Expected it in the project folder or the Python folder."
            )
            return

        try:
            config = self._build_run_config()
            config_path = self._write_run_config(config)
        except Exception as error:
            messagebox.showerror("Config Error", f"Could not write the run config file:\n\n{error}")
            return

        self.set_running_state(True, "python")
        self._print_run_summary(config, config_path, backend_script)
        self.append_console("[Run Started]\n")

        thread = threading.Thread(
            target=self._run_backend_thread,
            args=(backend_script, config_path),
            daemon=True
        )
        thread.start()

        self.after(100, self._process_output_queue)

    def _run_backend_thread(self, backend_script, config_path):
        try:
            backend_folder = os.path.dirname(backend_script)

            command = [
                sys.executable,
                "-u",
                backend_script,
                "--config",
                config_path
            ]

            self.output_queue.put(f"[Command] {' '.join(command)}\n\n")

            self.process = subprocess.Popen(
                command,
                cwd=backend_folder,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            if self.process.stdout is not None:
                for line in self.process.stdout:
                    self.output_queue.put(line)

            return_code = self.process.wait()

            if return_code == 0:
                self.output_queue.put("\n[Run Complete] Mesh surface generation finished successfully.\n")
            else:
                self.output_queue.put(f"\n[Run Failed] Backend exited with code {return_code}.\n")

        except Exception as error:
            self.output_queue.put(f"\n[Error] {error}\n")

        finally:
            self.process = None
            self.after(0, lambda: self.set_running_state(False))

    def _process_output_queue(self):
        try:
            while True:
                text = self.output_queue.get_nowait()
                self.append_console(text)
        except queue.Empty:
            pass

        if self.process is not None:
            self.after(100, self._process_output_queue)
        else:
            # One extra drain after the process finishes.
            try:
                while True:
                    text = self.output_queue.get_nowait()
                    self.append_console(text)
            except queue.Empty:
                pass



class AptGridApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.state = AppState()
        self.logo_image = None
        self.nav_icons = {}

        self.title(APP_TITLE)
        self.geometry(WINDOW_SIZE)
        self.minsize(1100, 700)
        self.configure(bg=SIDEBAR_BG)

        self._configure_styles()
        self._build_layout()
        self._build_floating_logos()
        self._create_pages()
        self._create_info_button()
        self.show_page("Home")

    def _configure_styles(self):
        style = ttk.Style(self)
        try: style.theme_use("clam")
        except tk.TclError: pass
        style.configure("Sidebar.TFrame", background=SIDEBAR_BG)
        style.configure("Content.TFrame", background=CONTENT_BG)
        style.configure("Topbar.TFrame", background=TOPBAR_BG)
        style.configure("Header.TLabel", background=SIDEBAR_BG)
        style.configure("TopbarLogo.TLabel", background=TOPBAR_BG)
        style.configure("TopbarTitle.TLabel", font=("Segoe UI", 24, "bold"), background=TOPBAR_BG, foreground=MAIN_BLUE)
        style.configure("PageTitle.TLabel", font=("Segoe UI", 24, "bold"), background=CONTENT_BG, foreground=MAIN_BLUE)
        style.configure("Section.TLabel", font=("Segoe UI", 15, "bold"), background=CONTENT_BG, foreground=MAIN_BLUE)
        style.configure("Field.TLabel", font=("Segoe UI", 11), background=CONTENT_BG)
        style.configure("Body.TLabel", font=("Segoe UI", 11), background=CONTENT_BG)
        style.configure("Muted.TLabel", font=("Segoe UI", 10), foreground="#5f7ea8", background=CONTENT_BG)
        style.configure("Nav.TButton", font=("Segoe UI", 12, "bold"), padding=(18, 14), anchor="center")
        style.map("Nav.TButton", background=[("pressed", "#d6ecff"), ("active", "#eaf5ff")])
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 8), background="#d6ecff", foreground=MAIN_BLUE)
        style.configure("TButton", font=("Segoe UI", 10), padding=(10, 8), focuscolor="")
        style.map("TButton", focuscolor=[("pressed", ""), ("active", ""), ("focus", "")])
        style.configure("TEntry", padding=6)
        style.configure("TSpinbox", padding=5)
        style.configure("TCheckbutton", background=CONTENT_BG, font=("Segoe UI", 10))
        style.configure("TSeparator", background="#c8dcf4")

    def _build_layout(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.sidebar = ttk.Frame(
            self,
            style="Sidebar.TFrame",
            width=SIDEBAR_WIDTH,
            padding=(22, 34)
        )
        self.sidebar.grid(row=0, column=0, sticky="ns")
        self.sidebar.grid_propagate(False)
        self.sidebar.columnconfigure(0, weight=1)
        self.sidebar.rowconfigure(0, weight=1)
        self.sidebar.rowconfigure(1, weight=0)

        self.nav_container = ttk.Frame(self.sidebar, style="Sidebar.TFrame")
        self.nav_container.grid(row=0, column=0, sticky="n")
        self.nav_container.columnconfigure(0, weight=1)

        self.info_container = ttk.Frame(self.sidebar, style="Sidebar.TFrame")
        self.info_container.grid(row=1, column=0, sticky="s", pady=(0, 0))
        self.info_container.columnconfigure(0, weight=1)

        self.content = ttk.Frame(self, style="Content.TFrame", padding=(16, 16, 16, 16))
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.columnconfigure(0, weight=1)
        self.content.rowconfigure(0, weight=1)

    def _build_sidebar_header(self):
        logo_path = os.path.join(os.path.dirname(__file__), ICONS_FOLDER, LOGO_FILENAME)

        if os.path.exists(logo_path):
            try:
                self.logo_image = tk.PhotoImage(file=logo_path)

                # resize if too large
                self.logo_image = self.logo_image.subsample(4, 4)

                logo_label = ttk.Label(
                    self.sidebar,
                    image=self.logo_image,
                    style="Header.TLabel"
                )

                logo_label.grid(
                    row=0,
                    column=0,
                    sticky="w",
                    padx=(45, 0),
                    pady=(0, 20)
                )

                return

            except Exception as e:
                print("Logo load failed:", e)

        fallback = ttk.Label(
            self.sidebar,
            text="APT Grid",
            style="Header.TLabel",
            font=("Segoe UI", 20, "bold")
        )

        fallback.grid(
            row=0,
            column=0,
            sticky="w",
            pady=(0, 20)
        )

    def _build_topbar(self):

        first_logo_path = os.path.join(
            os.path.dirname(__file__),
            ICONS_FOLDER,
            LOGO_FILENAME
        )

        second_logo_path = os.path.join(
            os.path.dirname(__file__),
            ICONS_FOLDER,
            SECOND_LOGO_FILENAME
        )

        if os.path.exists(first_logo_path):
            try:
                self.topbar_logo_1 = tk.PhotoImage(file=first_logo_path)
                self.topbar_logo_1 = self.topbar_logo_1.subsample(6, 6)

                ttk.Label(
                    self.topbar,
                    image=self.topbar_logo_1,
                    style="TopbarLogo.TLabel"
                ).place(
                    relx=1.0,
                    x=-250,
                    rely=0.5,
                    anchor="e"
                )

            except Exception as e:
                print("First logo failed:", e)

        if os.path.exists(second_logo_path):
            try:
                self.topbar_logo_2 = tk.PhotoImage(file=second_logo_path)
                self.topbar_logo_2 = self.topbar_logo_2.subsample(3, 3)

                ttk.Label(
                    self.topbar,
                    image=self.topbar_logo_2,
                    style="TopbarLogo.TLabel"
                ).place(
                    relx=1.0,
                    x=-20,
                    rely=0.5,
                    anchor="e"
                )

            except Exception as e:
                print("Second logo failed:", e)

    
    def _load_nav_icon(self, filename):
        path = os.path.join(os.path.dirname(__file__), ICONS_FOLDER, filename)
        if os.path.exists(path):
            try:
                # Load sidebar icons at their saved PNG size.
                # For crisp icons, export each PNG around 40x40 or 40x50 px
                # and do not resize/subsample it in Tkinter.
                return tk.PhotoImage(file=path)
            except tk.TclError:
                return None
        return None

    def _build_floating_logos(self):
        first_logo_path = os.path.join(
            os.path.dirname(__file__),
            ICONS_FOLDER,
            LOGO_FILENAME
        )

        second_logo_path = os.path.join(
            os.path.dirname(__file__),
            ICONS_FOLDER,
            SECOND_LOGO_FILENAME
        )

        if os.path.exists(first_logo_path):
            try:
                self.floating_logo_1 = tk.PhotoImage(file=first_logo_path)
                self.floating_logo_1 = self.floating_logo_1.subsample(6, 6)

                logo_1_label = tk.Label(
                    self,
                    image=self.floating_logo_1,
                    bg=CONTENT_BG,
                    borderwidth=0,
                    cursor="hand2"
                )
                logo_1_label.place(
                    relx=1.0,
                    x=-260,
                    y=26,
                    anchor="ne"
                )
                logo_1_label.bind(
                    "<Button-1>",
                    lambda event: webbrowser.open(
                        "https://www.uwindsor.ca/engineering/research/408/turbomachinery-and-unsteady-flows-research-lab"
                    )
                )

            except Exception as e:
                print("First logo failed:", e)

        if os.path.exists(second_logo_path):
            try:
                self.floating_logo_2 = tk.PhotoImage(file=second_logo_path)
                self.floating_logo_2 = self.floating_logo_2.subsample(3, 3)

                logo_2_label = tk.Label(
                    self,
                    image=self.floating_logo_2,
                    bg=CONTENT_BG,
                    borderwidth=0,
                    cursor="hand2"
                )
                logo_2_label.place(
                    relx=1.0,
                    x=-30,
                    y=10,
                    anchor="ne"
                )
                logo_2_label.bind(
                    "<Button-1>",
                    lambda event: webbrowser.open("https://www.uwindsor.ca/")
                )

            except Exception as e:
                print("Second logo failed:", e)

    def _draw_rounded_rect(self, canvas, x1, y1, x2, y2, radius, fill, outline=""):
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1,
        ]

        canvas.create_polygon(
            points,
            smooth=True,
            splinesteps=24,
            fill=fill,
            outline=outline,
            width=2
        )

    def _create_pages(self):
        self.pages = {}
        self.nav_buttons = {}

        page_specs = [
            ("Home", HomePage, "home.png"),
            ("Files", FilesPage, "folder.png"),
            ("Basic Setup", BasicSetupPage, "grid.png"),
            ("Boundary Layer", BoundaryLayerPage, "layer.png"),
            ("Mesh Tuning", MeshTuningPage, "tuning.png"),
            ("Advanced", AdvancedPage, "advanced.png"),
            ("Run", RunPage, "run.png"),
        ]

        self.nav_container.columnconfigure(0, weight=1)

        for row_index, (name, page_class, icon_file) in enumerate(page_specs):
            icon = self._load_nav_icon(icon_file)
            self.nav_icons[name] = icon

            button = tk.Canvas(
                self.nav_container,
                width=82,
                height=82,
                bg=SIDEBAR_BG,
                highlightthickness=0,
                bd=0,
                cursor="hand2"
            )
            button.grid(row=row_index, column=0, sticky="n", pady=8)

            if icon is not None:
                button.create_image(41, 41, image=icon, anchor="center")

            button.bind(
                "<Button-1>",
                lambda event, page_name=name: self.show_page(page_name)
            )

            self.nav_buttons[name] = button

            page = page_class(self.content, self)
            page.grid(row=0, column=0, sticky="nsew")
            self.pages[name] = page

    def _create_info_button(self):
        info_icon = self._load_nav_icon("info.png")
        self.nav_icons["Info"] = info_icon

        self.info_button = tk.Canvas(
            self.info_container,
            width=82,
            height=82,
            bg=SIDEBAR_BG,
            highlightthickness=0,
            bd=0,
            cursor="hand2"
        )
        self.info_button.grid(row=0, column=0, sticky="s")

        if info_icon is not None:
            self.info_button.create_image(41, 41, image=info_icon, anchor="center")

        self.info_button.bind(
            "<Button-1>",
            lambda event: self._show_info_popup()
        )

    def _show_info_popup(self):
        messagebox.showinfo(
            "About BladeForge",
            (
                "BladeForge\n\n"
                "This graphical interface supports the blade passage "
                "grid-generation workflow by providing a guided environment for "
                "selecting geometry files, configuring mesh parameters, and "
                "launching the backend generation process.\n\n"
                "Developed for the TUFRG research team at the University of Windsor.\n\n"
                "For any inquiries, please contact:\n"
                "jdefoe@uwindsor.ca"
            )
        )

    def show_page(self, name: str):
        page = self.pages[name]
        if hasattr(page, "load_state"):
            page.load_state()
        if hasattr(page, "on_show"):
            page.on_show()
        page.tkraise()
        self._highlight_nav(name)
        self.content.focus_set()

    def _highlight_nav(self, active_name: str):
        for name, button in self.nav_buttons.items():
            button.delete("all")

            if name == active_name:
                self._draw_rounded_rect(
                    button,
                    3,
                    3,
                    79,
                    79,
                    radius=22,
                    fill="#d6ecff",
                    outline=MAIN_BLUE
                )

            icon = self.nav_icons.get(name)

            if icon is not None:
                button.create_image(41, 41, image=icon, anchor="center")

    def dump_state_to_terminal(self):
        print("\n" + "=" * 50)
        print("CURRENT APP STATE")
        print("=" * 50)
        for key, value in self.state.values.items():
            print(f"{key}: {value}")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    app = AptGridApp()
    app.mainloop()