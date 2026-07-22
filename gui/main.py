import ctypes
import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
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

        title_label = ttk.Label(self, text=title, style="PageTitle.TLabel")
        title_label.grid(row=0, column=0, sticky="w")

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
        super().__init__(parent, app, title="APT-Grid Interface")

        self.workflow_step_cards = []
        self.workflow_animation_index = 0
        self.workflow_animation_after_id = None

        self._build_page()
        self._start_workflow_animation()

    def _build_page(self):
        self.body.columnconfigure(0, weight=1)
        self.body.configure(padding=(0, 50, 0, 0))

        self.body.rowconfigure(0, weight=0)
        self.body.rowconfigure(1, weight=0)

        # --------------------------------------------------
        # Welcome / overview card
        # --------------------------------------------------
        overview_card = self._create_card(self.body)
        overview_card.grid(row=0, column=0, sticky="ew", pady=(0, 32))
        overview_card.columnconfigure(0, weight=1)
        overview_card.columnconfigure(1, weight=0)

        overview_title = tk.Label(
            overview_card,
            text="Welcome",
            font=("Segoe UI", 16, "bold"),
            fg=MAIN_BLUE,
            bg="white"
        )
        overview_title.grid(row=0, column=0, sticky="w", padx=28, pady=(22, 8))

        overview_text = tk.Label(
            overview_card,
            text=(
                "APT-Grid Interface provides a guided setup environment for selecting geometry files, "
                "configuring blade passage parameters, defining boundary-layer and mesh controls, "
                "and launching the backend surface-generation process."
            ),
            font=("Segoe UI", 10),
            fg="#1f2933",
            bg="white",
            wraplength=1160,
            justify="left"
        )
        overview_text.grid(row=1, column=0, sticky="w", padx=28, pady=(0, 24))

        start_button = tk.Button(
            overview_card,
            text="Start New Project",
            command=lambda: self.app.show_page("Files"),
            font=("Segoe UI", 10, "bold"),
            fg="white",
            bg=MAIN_BLUE,
            activeforeground="white",
            activebackground="#003f73",
            disabledforeground="white",
            relief="flat",
            bd=0,
            padx=22,
            pady=9,
            cursor="hand2"
        )
        start_button.grid(row=0, column=1, rowspan=2, sticky="e", padx=(18, 28), pady=24)

        # --------------------------------------------------
        # Lower layout: workflow left, credits right
        # --------------------------------------------------
        lower_grid = tk.Frame(self.body, bg=CONTENT_BG)
        lower_grid.grid(row=1, column=0, sticky="ew")
        lower_grid.columnconfigure(0, weight=7)
        lower_grid.columnconfigure(1, weight=5)

        workflow_card = self._create_card(lower_grid)
        workflow_card.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        workflow_card.columnconfigure(0, weight=1)

        workflow_title = tk.Label(
            workflow_card,
            text="Workflow Overview",
            font=("Segoe UI", 16, "bold"),
            fg=MAIN_BLUE,
            bg="white"
        )
        workflow_title.grid(row=0, column=0, sticky="w", padx=28, pady=(24, 18))

        workflow_steps = [
            ("01", "Files", "Select geometry files and output path."),
            ("02", "Setup", "Define blade count, scale, and resolution."),
            ("03", "Mesh Controls", "Configure boundary layer, mesh tuning, and advanced settings."),
            ("04", "Run", "Write the configuration file and launch the backend generator."),
        ]

        diagram_frame = tk.Frame(workflow_card, bg="white")
        diagram_frame.grid(row=1, column=0, sticky="ew", padx=28, pady=(0, 26))
        diagram_frame.columnconfigure(0, weight=1)

        for index, (number, title, description) in enumerate(workflow_steps):
            step_card = tk.Frame(
                diagram_frame,
                bg="#fbfdff",
                highlightbackground="#c8dcf4",
                highlightthickness=1,
                bd=0,
                height=76
            )
            step_card.grid(row=index * 2, column=0, sticky="ew")
            step_card.grid_propagate(False)
            step_card.columnconfigure(0, weight=0)
            step_card.columnconfigure(1, weight=1)
            step_card.rowconfigure(0, weight=1)

            badge = tk.Label(
                step_card,
                text=number,
                font=("Segoe UI", 9, "bold"),
                fg="white",
                bg=MAIN_BLUE,
                width=4,
                height=1
            )
            badge.grid(row=0, column=0, sticky="nsw", padx=(18, 16))

            text_block = tk.Frame(step_card, bg="#fbfdff")
            text_block.grid(row=0, column=1, sticky="w", padx=(0, 18))

            step_title = tk.Label(
                text_block,
                text=title,
                font=("Segoe UI", 10, "bold"),
                fg=MAIN_BLUE,
                bg="#fbfdff",
                anchor="w"
            )
            step_title.grid(row=0, column=0, sticky="w")

            step_desc = tk.Label(
                text_block,
                text=description,
                font=("Segoe UI", 9),
                fg="#4d5f73",
                bg="#fbfdff",
                justify="left",
                wraplength=560,
                anchor="w"
            )
            step_desc.grid(row=1, column=0, sticky="w", pady=(4, 0))

            self.workflow_step_cards.append({
                "card": step_card,
                "badge": badge,
                "text_block": text_block,
                "title": step_title,
                "description": step_desc,
            })

            if index < len(workflow_steps) - 1:
                arrow = tk.Label(
                    diagram_frame,
                    text="↓",
                    font=("Segoe UI", 12, "bold"),
                    fg=MAIN_BLUE,
                    bg="white"
                )
                arrow.grid(row=index * 2 + 1, column=0, sticky="w", padx=36, pady=3)

        credits_card = self._create_card(lower_grid)
        credits_card.grid(row=0, column=1, sticky="nsew", padx=(18, 0))
        credits_card.columnconfigure(0, weight=1)
        credits_card.rowconfigure(3, weight=1)

        credits_title = tk.Label(
            credits_card,
            text="Development Credits",
            font=("Segoe UI", 16, "bold"),
            fg=MAIN_BLUE,
            bg="white"
        )
        credits_title.grid(row=0, column=0, sticky="w", padx=28, pady=(24, 24))

        self._add_credit_row(
            credits_card,
            row=1,
            heading="Backend logic developed by",
            names="Adekola Adeyemi, Justin Smart, Tony Woo, and Jeff Defoe"
        )

        self._add_credit_row(
            credits_card,
            row=2,
            heading="Software interface designed by",
            names="Misk Damdoum"
        )

    def _start_workflow_animation(self):
        self._animate_workflow_steps()

    def _animate_workflow_steps(self):
        if not self.workflow_step_cards:
            return

        for index, step in enumerate(self.workflow_step_cards):
            self._set_step_active(step, index == self.workflow_animation_index)

        self.workflow_animation_index = (self.workflow_animation_index + 1) % len(self.workflow_step_cards)
        self.workflow_animation_after_id = self.after(950, self._animate_workflow_steps)

    def _set_step_active(self, step, active):
        if active:
            card_bg = "#eef7ff"
            border_color = MAIN_BLUE
            title_fg = MAIN_BLUE
            desc_fg = "#1f2933"
        else:
            card_bg = "#fbfdff"
            border_color = "#c8dcf4"
            title_fg = MAIN_BLUE
            desc_fg = "#4d5f73"

        step["card"].configure(
            bg=card_bg,
            highlightbackground=border_color,
            highlightcolor=border_color,
            highlightthickness=2 if active else 1,
        )
        step["badge"].configure(bg=MAIN_BLUE)
        step["text_block"].configure(bg=card_bg)
        step["title"].configure(bg=card_bg, fg=title_fg)
        step["description"].configure(bg=card_bg, fg=desc_fg)

    def _create_card(self, parent):
        return tk.Frame(
            parent,
            bg="white",
            highlightbackground="#d8e3ef",
            highlightthickness=1,
            bd=0
        )

    def _add_credit_row(self, parent, row, heading, names):
        container = tk.Frame(parent, bg="white")
        container.grid(row=row, column=0, sticky="ew", padx=28, pady=(0, 28))
        container.columnconfigure(0, weight=1)

        heading_label = tk.Label(
            container,
            text=heading,
            font=("Segoe UI", 10, "bold"),
            fg="#1f2933",
            bg="white"
        )
        heading_label.grid(row=0, column=0, sticky="w")

        names_label = tk.Label(
            container,
            text=names,
            font=("Segoe UI", 10),
            fg="#4d5f73",
            bg="white",
            wraplength=440,
            justify="left"
        )
        names_label.grid(row=1, column=0, sticky="w", pady=(10, 0))


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

        self.run_button = tk.Button(
            run_card,
            text="▶  Run Mesh Generation",
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
            cursor="hand2"
        )
        self.run_button.grid(row=0, column=1, rowspan=2, sticky="e", padx=24, pady=18)

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

    def set_running_state(self, is_running):
        if is_running:
            self.run_button.configure(
                text="Running...",
                state="disabled",
                fg="white",
                disabledforeground="white",
                bg="#7fa6c9"
            )
        else:
            self.run_button.configure(
                text="▶  Run Mesh Generation",
                state="normal",
                fg="white",
                bg=MAIN_BLUE
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

        self.set_running_state(True)
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
            "About APT-Grid Interface",
            (
                "APT-Grid Interface\n\n"
                "This graphical interface supports the APT-Grid blade passage "
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