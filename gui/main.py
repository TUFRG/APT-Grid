import ctypes
import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, ttk, messagebox


APP_TITLE = "APT-Grid Interface"
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

        title_label = ttk.Label(self, text=title, style="PageTitle.TLabel")
        title_label.grid(row=0, column=0, sticky="w")

        self.body = ttk.Frame(self, padding=(0, 24, 0, 0), style="Content.TFrame")
        self.body.grid(row=1, column=0, sticky="nsew")
        self.body.columnconfigure(0, weight=1)


class HomePage(BasePage):
    def __init__(self, parent, app):
        super().__init__(parent, app, title="APT-Grid Interface")
        self.body.rowconfigure(1, weight=1)

        intro = ttk.Label(
            self.body,
            text=(
                "Welcome to the APT-Grid Interface. "
                "Use the sidebar to select input files, configure grid parameters, "
                "adjust mesh settings, and run the blade passage grid-generation workflow."
            ),
            style="Body.TLabel",
            wraplength=900,
            justify="left",
        )
        intro.grid(row=0, column=0, sticky="w")


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
            ("Number of blades", self.blades_var, 1, 500, 1),
            ("Scale (m)", self.scale_var, 0.000001, 1000.0, 0.001),
            ("Radial points (outside of hub and casing boundary layers)", self.nrad_var, 1, 1000, 1),
        ]

        for row, (label_text, var, min_val, max_val, step) in enumerate(fields, start=1):
            ttk.Label(content, text=label_text, style="Field.TLabel").grid(
                row=row, column=0, sticky="w", padx=(0, 24), pady=12
            )

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

        ttk.Label(content, text="Periodic mode", style="Field.TLabel").grid(
            row=4, column=0, sticky="w", padx=(0, 24), pady=12
        )

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
        label = ttk.Label(parent, text=label_text, style="Field.TLabel")
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
        ttk.Label(parent, text=label_text, style="Field.TLabel").grid(
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
        ttk.Label(
            parent,
            text=label_text,
            style="Field.TLabel"
        ).grid(
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

        self.output_queue = queue.Queue()
        self.is_running = False
        self.last_config_path = None

        self._build_page()
        self._poll_output_queue()

    def _build_page(self):
        self.body.columnconfigure(0, weight=1)
        self.body.rowconfigure(3, weight=1)

        # --------------------------------------------------
        # Run button area
        # --------------------------------------------------
        run_panel = ttk.Frame(self.body, style="Content.TFrame")
        run_panel.grid(row=0, column=0, sticky="ew", pady=(22, 34))
        run_panel.columnconfigure(0, weight=1)

        self.run_button = ttk.Button(
            run_panel,
            text="▶  Run Mesh Generation",
            style="Primary.TButton",
            takefocus=False,
            command=self.start_mesh_generation,
        )
        self.run_button.grid(row=0, column=0)

        # --------------------------------------------------
        # Separator
        # --------------------------------------------------
        ttk.Separator(self.body, orient="horizontal").grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(0, 26)
        )

        # --------------------------------------------------
        # Console title
        # --------------------------------------------------
        console_label = ttk.Label(
            self.body,
            text="Console Output",
            style="Section.TLabel"
        )
        console_label.grid(row=2, column=0, sticky="w", pady=(0, 12))

        # --------------------------------------------------
        # Console output box
        # --------------------------------------------------
        console_frame = ttk.Frame(self.body, style="Content.TFrame")
        console_frame.grid(row=3, column=0, sticky="nsew")
        console_frame.columnconfigure(0, weight=1)
        console_frame.rowconfigure(0, weight=1)

        self.console = tk.Text(
            console_frame,
            height=14,
            wrap="word",
            font=("Consolas", 10),
            bg="white",
            fg="#1f1f1f",
            relief="solid",
            borderwidth=1,
            padx=14,
            pady=12,
            state="disabled"
        )
        self.console.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(
            console_frame,
            orient="vertical",
            command=self.console.yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.console.configure(yscrollcommand=scrollbar.set)

        # --------------------------------------------------
        # Bottom action buttons
        # --------------------------------------------------
        action_bar = ttk.Frame(self.body, style="Content.TFrame")
        action_bar.grid(row=4, column=0, sticky="e", pady=(22, 0))

        ttk.Button(
            action_bar,
            text="Clear Console",
            takefocus=False,
            command=self.clear_console
        ).grid(row=0, column=0, padx=(0, 12))

        ttk.Button(
            action_bar,
            text="Save Log",
            takefocus=False,
            command=self.save_log
        ).grid(row=0, column=1)

        self._append_console("[Ready] Configure inputs, then run mesh generation.\n")

    def start_mesh_generation(self):
        if self.is_running:
            return

        self.clear_console()
        self.is_running = True
        self.run_button.state(["disabled"])

        values = dict(self.app.state.values)

        worker = threading.Thread(
            target=self._run_mesh_generation_worker,
            args=(values,),
            daemon=True
        )
        worker.start()

    def _run_mesh_generation_worker(self, values):
        try:
            self._queue_line("[Step 1/6] Validating run settings...\n")
            errors = self._validate_run_settings(values)

            if errors:
                self._queue_line("[ERROR] Missing required run settings:\n")
                for error in errors:
                    self._queue_line(f"  - {error}\n")
                self._queue_line("\nMesh generation was not started.\n")
                self.output_queue.put(("done", False))
                return

            self._queue_line("[OK] Run settings validated successfully.\n\n")

            self._queue_line("[Step 2/6] Preparing output folder...\n")
            output_path = values.get("outputPath", "")
            os.makedirs(output_path, exist_ok=True)
            self._queue_line(f"[OK] Output folder ready: {output_path}\n\n")

            self._queue_line("[Step 3/6] Saving GUI run configuration...\n")
            config_path = os.path.join(output_path, "apt_grid_run_config.json")
            with open(config_path, "w", encoding="utf-8") as file:
                json.dump(values, file, indent=4)

            self.last_config_path = config_path
            self._queue_line(f"[OK] GUI configuration saved: {config_path}\n\n")

            self._queue_run_input_summary(values)
            self._queue_line("[Step 4/6] Locating backend mesh generator...\n")
            backend_script = self._find_backend_script()

            if backend_script is None:
                self._queue_line("[ERROR] Could not find backend file.\n\n")
                self._queue_line("The GUI checked these locations:\n")
                for path in self._candidate_backend_paths():
                    self._queue_line(f"  - {path}\n")

                self._queue_line(
                    "\nFix: place bladePassageSurfaceGenerator_v2.py in the main APT-Grid folder, "
                    "in the Python folder, or in the same folder as this GUI file.\n"
                )
                self.output_queue.put(("done", False))
                return

            backend_folder = os.path.dirname(backend_script)
            self._queue_line(f"[OK] Backend found: {backend_script}\n")
            self._queue_line(f"[OK] Backend working folder: {backend_folder}\n\n")

            self._queue_line("[Step 5/6] Starting backend process...\n")
            self._queue_line("--------------------------------------------------\n")

            command = [
                sys.executable,
                "-u",
                backend_script,
                "--config",
                config_path
            ]

            process = subprocess.Popen(
                command,
                cwd=backend_folder,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            if process.stdout is not None:
                for line in process.stdout:
                    self._queue_line(line)

            return_code = process.wait()

            self._queue_line("--------------------------------------------------\n")

            if return_code == 0:
                self._queue_line("[Step 6/6] Mesh generation completed.\n")
                self._queue_line("[SUCCESS] Mesh generated successfully!\n")
                self._queue_line(f"Output path: {output_path}\n")
                self.output_queue.put(("done", True))
            else:
                self._queue_line("[Step 6/6] Mesh generation stopped with an error.\n")
                self._queue_line(f"[ERROR] Backend returned exit code: {return_code}\n")
                self.output_queue.put(("done", False))

        except Exception as error:
            self._queue_line("\n[ERROR] Unexpected GUI run error:\n")
            self._queue_line(f"{error}\n")
            self.output_queue.put(("done", False))

    def _candidate_backend_paths(self):
        gui_folder = os.path.dirname(os.path.abspath(__file__))
        project_folder = os.path.dirname(gui_folder)
        current_working_folder = os.getcwd()

        candidates = [
            os.path.join(gui_folder, "bladePassageSurfaceGenerator_v2.py"),
            os.path.join(project_folder, "bladePassageSurfaceGenerator_v2.py"),
            os.path.join(project_folder, "Python", "bladePassageSurfaceGenerator_v2.py"),
            os.path.join(current_working_folder, "bladePassageSurfaceGenerator_v2.py"),
            os.path.join(current_working_folder, "Python", "bladePassageSurfaceGenerator_v2.py"),
            os.path.join(current_working_folder, "..", "bladePassageSurfaceGenerator_v2.py"),
        ]

        cleaned_candidates = []
        for path in candidates:
            normalized = os.path.abspath(path)
            if normalized not in cleaned_candidates:
                cleaned_candidates.append(normalized)

        return cleaned_candidates

    def _find_backend_script(self):
        for path in self._candidate_backend_paths():
            if os.path.exists(path):
                return path

        gui_folder = os.path.dirname(os.path.abspath(__file__))
        project_folder = os.path.dirname(gui_folder)

        for root, dirs, files in os.walk(project_folder):
            dirs[:] = [d for d in dirs if d not in {"__pycache__", ".git", ".venv", "venv", "env"}]

            if "bladePassageSurfaceGenerator_v2.py" in files:
                return os.path.join(root, "bladePassageSurfaceGenerator_v2.py")

        return None
    
    def _queue_run_input_summary(self, values):
        self._queue_line("[GUI Input Summary]\n")
        self._queue_line("--------------------------------------------------\n")

        sections = {
            "Files": [
                "dataPath",
                "hubFileName",
                "casFileName",
                "bladeCurveFile",
                "outputPath",
            ],
            "Basic Setup": [
                "Nb",
                "periodic",
                "scale",
                "nrad",
            ],
            "Boundary Layer": [
                "autoBL",
                "rhoref",
                "Uref",
                "LrefHub",
                "LrefCas",
                "LrefBla",
                "muref",
                "yPlusHub",
                "yPlusCas",
                "yPlusBla",
                "delHub",
                "delCas",
                "delBla",
                "dy1Hub",
                "dy1Cas",
                "dy1Bla",
            ],
            "Mesh Tuning": [
                "gRad",
                "gTan",
                "additionalTangentialRefine",
                "dax1primeLE",
                "rLE",
                "dax1primeTE",
                "rTE",
                "additionalAxialRefine",
                "rUpFar",
                "rDnFar",
            ],
            "Advanced": [
                "percentVal",
                "percentValNonCutLE",
                "percentValNonCutTE",
                "angConstraintCurves",
                "angConstraintOffsets",
            ],
        }

        for section_name, keys in sections.items():
            self._queue_line(f"{section_name}:\n")

            for key in keys:
                self._queue_line(f"  {key}: {values.get(key)}\n")

            self._queue_line("\n")

        self._queue_line("--------------------------------------------------\n\n")


    def _validate_run_settings(self, values):
        errors = []

        required_fields = [
            ("Input data folder", "dataPath"),
            ("Hub curve file", "hubFileName"),
            ("Casing curve file", "casFileName"),
            ("Blade curve file", "bladeCurveFile"),
            ("Output data folder", "outputPath"),
        ]

        for label, key in required_fields:
            value = str(values.get(key, "")).strip()
            if not value:
                errors.append(f"{label} is missing.")

        output_path = str(values.get("outputPath", "")).strip()
        if output_path:
            parent_folder = os.path.dirname(output_path) or output_path
            if not os.path.exists(parent_folder):
                errors.append(f"Output folder parent path does not exist: {parent_folder}")

        return errors

    def _queue_line(self, text):
        self.output_queue.put(("text", text))

    def _poll_output_queue(self):
        try:
            while True:
                item_type, payload = self.output_queue.get_nowait()

                if item_type == "text":
                    self._append_console(payload)

                elif item_type == "done":
                    self.is_running = False
                    self.run_button.state(["!disabled"])

                    if payload:
                        self._append_console("\n[Ready] Run finished successfully.\n")
                    else:
                        self._append_console("\n[Ready] Run finished with errors. Review the console output above.\n")

        except queue.Empty:
            pass

        self.after(100, self._poll_output_queue)

    def _append_console(self, text):
        self.console.configure(state="normal")
        self.console.insert("end", text)
        self.console.see("end")
        self.console.configure(state="disabled")

    def clear_console(self):
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")

    def save_log(self):
        log_text = self.console.get("1.0", "end-1c")

        if not log_text.strip():
            self._append_console("[Info] Console is empty. Nothing to save.\n")
            return

        default_name = "apt_grid_log_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"

        selected_path = filedialog.asksaveasfilename(
            title="Save console log",
            defaultextension=".txt",
            initialfile=default_name,
            filetypes=[
                ("Text files", "*.txt"),
                ("All files", "*.*"),
            ]
        )

        if selected_path:
            with open(selected_path, "w", encoding="utf-8") as file:
                file.write(log_text)

            self._append_console(f"\n[Info] Log saved to: {selected_path}\n")



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

                tk.Label(
                    self,
                    image=self.floating_logo_1,
                    bg=CONTENT_BG,
                    borderwidth=0
                ).place(
                    relx=1.0,
                    x=-260,
                    y=26,
                    anchor="ne"
                )

            except Exception as e:
                print("First logo failed:", e)

        if os.path.exists(second_logo_path):
            try:
                self.floating_logo_2 = tk.PhotoImage(file=second_logo_path)
                self.floating_logo_2 = self.floating_logo_2.subsample(3, 3)

                tk.Label(
                    self,
                    image=self.floating_logo_2,
                    bg=CONTENT_BG,
                    borderwidth=0
                ).place(
                    relx=1.0,
                    x=-30,
                    y=10,
                    anchor="ne"
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
            "About APT-Grid Interface",
            (
                "APT-Grid Interface\n\n"
                "Graphical interface for configuring and running the "
                "APT-Grid blade passage grid-generation workflow.\n\n"
                "Backend: TUFRG APT-Grid\n"
                "GUI: Misk Damdoum"
            )
        )

    def show_page(self, name: str):
        page = self.pages[name]
        if hasattr(page, "load_state"):
            page.load_state()
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