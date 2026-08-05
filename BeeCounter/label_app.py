'''
Bee-counting labeling tool.

Workflow: pick a hive and a timestamp, load the 4 camera images taken at that
time. For each camera: draw a rectangle over a zone with "typical" bee density,
count the bees in that zone by clicking on them, then enter a multiplication
factor yourself (a suggestion computed from the zone/image area ratio is shown,
but you decide the final factor) to get an estimated bee count for the full
image. Once done for the 4 cameras, save the session to a CSV.

Run with:
    python label_app.py

Author: Cyril Monette
'''

import calendar as calendar_module
import datetime as dt
import os
import tkinter as tk
import uuid
from tkinter import filedialog, messagebox, ttk
from zoneinfo import ZoneInfo

import cv2
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

import pandas as pd

from image_source import DEFAULT_IMAGES_ROOT, find_camera_images, list_hives
from results_store import append_results
from Preprocessing.preproc import beautify_frame  # noqa: E402 - needs image_source's sys.path setup first

POINT_MARKER_STYLE = dict(marker="x", color="red", markersize=8, markeredgewidth=2)
ZONE_SELECTOR_KWARGS = dict(
    useblit=True,
    button=[1],
    minspanx=5,
    minspany=5,
    spancoords="pixels",
    interactive=True,
)

class SimpleToolbar(NavigationToolbar2Tk):
    '''
    NavigationToolbar2Tk without the Pan/Zoom/Subplots buttons.

    Those toolbar tools set toolbar.mode to a non-empty string while toggled
    on (not just while actively dragging), which silently blocked every
    left-click bee count until the button was clicked again to untoggle it.
    Scroll-wheel zoom and right-click-drag pan replace them without that trap.
    '''

    toolitems = [t for t in NavigationToolbar2Tk.toolitems if t[0] not in ("Pan", "Zoom", "Subplots")]


_FRENCH_MONTHS = [
    "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
    "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre",
]
_FRENCH_WEEKDAYS = ["Lu", "Ma", "Me", "Je", "Ve", "Sa", "Di"]


class SimpleDatePicker(ttk.Frame):
    '''
    Minimal calendar-dropdown date picker built from plain ttk widgets only.

    Written instead of using tkcalendar.DateEntry, whose popup calendar
    renders blank and freezes the app on this machine's Tcl/Tk (8.6.13,
    macOS) - a known Tk/aqua rendering bug in that patch version. This
    widget never grabs input, so even if something goes wrong with the
    popup, the rest of the app stays usable.
    '''

    _ENTRY_FORMATS = ["%d/%m/%Y", "%d.%m.%Y", "%d-%m-%Y", "%Y-%m-%d"]

    def __init__(self, parent, initial_date=None):
        super().__init__(parent)
        self._date = initial_date or dt.date.today()
        self._popup = None
        self._popup_frame = None
        self._view_year = self._date.year
        self._view_month = self._date.month

        self._entry_var = tk.StringVar()
        self._entry = ttk.Entry(self, textvariable=self._entry_var, width=10)
        self._entry.pack(side="left")
        self._entry.bind("<Return>", self._on_entry_commit)
        self._entry.bind("<FocusOut>", self._on_entry_commit)
        self._entry.bind("<Button-1>", lambda e: self._close_popup())
        self._toggle_button = ttk.Button(self, text="\U0001F4C5", width=3, command=self._toggle_popup)
        self._toggle_button.pack(side="left", padx=(2, 0))
        self._refresh_entry()

    def get_date(self) -> dt.date:
        return self._date

    def set_date(self, date_value: dt.date):
        self._date = date_value
        self._view_year, self._view_month = date_value.year, date_value.month
        self._refresh_entry()

    def _refresh_entry(self):
        self._entry_var.set(self._date.strftime("%d/%m/%Y"))

    def _on_entry_commit(self, event=None):
        text = self._entry_var.get().strip()
        for fmt in self._ENTRY_FORMATS:
            try:
                parsed = dt.datetime.strptime(text, fmt).date()
                break
            except ValueError:
                continue
        else:
            messagebox.showerror("Date invalide", f"Format de date non reconnu: {text!r}\nUtilisez JJ/MM/AAAA.")
            self._refresh_entry()  # revert to the last valid date
            return
        self._date = parsed
        self._view_year, self._view_month = parsed.year, parsed.month
        if self._popup is not None:
            self._refresh_nav_controls()
            self._render_day_grid()

    def _toggle_popup(self):
        if self._popup is not None:
            self._close_popup()
        else:
            self._open_popup()

    def _open_popup(self):
        self._view_year, self._view_month = self._date.year, self._date.month

        self._popup = tk.Toplevel(self)
        self._popup.wm_overrideredirect(True)
        self._popup.wm_attributes("-topmost", True)
        x = self._toggle_button.winfo_rootx()
        y = self._toggle_button.winfo_rooty() + self._toggle_button.winfo_height()
        self._popup.geometry(f"+{x}+{y}")
        self._popup.bind("<Escape>", lambda e: self._close_popup())

        self._popup_frame = ttk.Frame(self._popup, borderwidth=1, relief="solid", padding=4)
        self._popup_frame.pack()

        # The nav controls (month combobox, year spinbox) are built once and never
        # destroyed while the popup is open: destroying a widget from inside its own
        # <<ComboboxSelected>> callback (as a "clear + rebuild everything" approach
        # would do) is what made the popup vanish on this machine's Tk. Only the day
        # grid below gets rebuilt when the visible month/year changes.
        nav = ttk.Frame(self._popup_frame)
        nav.pack(fill="x")
        ttk.Button(nav, text="◀", width=2, command=self._prev_month).pack(side="left")

        self._month_var = tk.StringVar()
        month_combo = ttk.Combobox(nav, textvariable=self._month_var, values=_FRENCH_MONTHS,
                                    state="readonly", width=9)
        month_combo.pack(side="left", padx=(2, 2))
        month_combo.bind("<<ComboboxSelected>>", self._on_month_combo)

        self._year_var = tk.StringVar()
        year_spin = ttk.Spinbox(nav, from_=1970, to=2100, textvariable=self._year_var, width=5,
                                 command=self._on_year_spin_change)
        year_spin.pack(side="left")
        year_spin.bind("<Return>", self._on_year_spin_change)
        year_spin.bind("<FocusOut>", self._on_year_spin_change)

        ttk.Button(nav, text="▶", width=2, command=self._next_month).pack(side="left", padx=(2, 0))

        self._grid_frame = ttk.Frame(self._popup_frame)
        self._grid_frame.pack(pady=(4, 0))

        ttk.Button(self._popup_frame, text="Aujourd'hui", command=self._pick_today).pack(pady=(4, 0))

        self._refresh_nav_controls()
        self._render_day_grid()
        self._popup.focus_set()

    def _close_popup(self):
        if self._popup is not None:
            popup = self._popup
            self._popup = None
            self._popup_frame = None
            self._grid_frame = None
            # Deferred: day/"Aujourd'hui" buttons that call _close_popup() live inside
            # this very popup, and destroying it synchronously from inside its own
            # button callback is what left keyboard focus stuck on this Tk build (the
            # same self-destruction pattern that broke the month combobox before).
            # Destroying on the next idle tick, after the click has fully unwound, and
            # explicitly forcing focus back to the main window fixes both.
            self.after_idle(popup.destroy)
            self.after_idle(lambda: self.winfo_toplevel().focus_force())

    def _refresh_nav_controls(self):
        self._month_var.set(_FRENCH_MONTHS[self._view_month - 1])
        self._year_var.set(str(self._view_year))

    def _render_day_grid(self):
        for child in self._grid_frame.winfo_children():
            child.destroy()

        for col, wd in enumerate(_FRENCH_WEEKDAYS):
            ttk.Label(self._grid_frame, text=wd, width=3, anchor="center").grid(row=0, column=col)

        cal = calendar_module.Calendar(firstweekday=0)
        for row, week in enumerate(cal.monthdayscalendar(self._view_year, self._view_month), start=1):
            for col, day in enumerate(week):
                if day == 0:
                    ttk.Label(self._grid_frame, text="", width=3).grid(row=row, column=col)
                    continue
                is_selected = (day == self._date.day and self._view_month == self._date.month
                               and self._view_year == self._date.year)
                text = f"[{day}]" if is_selected else str(day)
                ttk.Button(self._grid_frame, text=text, width=3,
                           command=lambda d=day: self._pick_day(d)).grid(row=row, column=col)

    def _on_month_combo(self, event=None):
        self._view_month = _FRENCH_MONTHS.index(self._month_var.get()) + 1
        self._render_day_grid()

    def _on_year_spin_change(self, event=None):
        try:
            year = int(self._year_var.get())
        except (TypeError, ValueError):
            year = self._view_year
        self._view_year = max(1970, min(2100, year))
        self._refresh_nav_controls()  # reflect any clamping back into the spinbox text
        self._render_day_grid()

    def _prev_month(self):
        self._view_month -= 1
        if self._view_month == 0:
            self._view_month = 12
            self._view_year -= 1
        self._refresh_nav_controls()
        self._render_day_grid()

    def _next_month(self):
        self._view_month += 1
        if self._view_month == 13:
            self._view_month = 1
            self._view_year += 1
        self._refresh_nav_controls()
        self._render_day_grid()

    def _pick_day(self, day):
        self._date = dt.date(self._view_year, self._view_month, day)
        self._refresh_entry()
        self._close_popup()

    def _pick_today(self):
        self._date = dt.date.today()
        self._view_year, self._view_month = self._date.year, self._date.month
        self._refresh_entry()
        self._close_popup()


class CameraState:
    '''In-progress labeling state for a single camera image.'''

    def __init__(self, camera_name: str, path: str):
        self.camera_name = camera_name
        self.path = path
        self.image = None  # lazily loaded np.ndarray (grayscale)
        self.zone = None  # (x0, y0, x1, y1) in real image pixel coords, or None
        self.points = []  # list of (x, y) in real image pixel coords
        self.point_artists = []  # matplotlib artists matching self.points, 1:1
        self.factor_text = ""  # what the user typed in the factor entry
        self.validated = False

    def get_image(self):
        if self.image is None and self.path is not None:
            img = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise IOError(f"Could not read image: {self.path}")
            self.image = beautify_frame(img)
        return self.image


class BeeLabelApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BeeCounter - Labellisation")
        self.geometry("1400x900")

        self.images_root = DEFAULT_IMAGES_ROOT
        self.session_id = None
        self.requested_timestamp = None
        self.hive_nb = None
        self.camera_states: list[CameraState] = []
        self.current_camera_idx = None
        self.interaction_mode = "navigate"  # 'navigate' | 'zone' | 'count'
        self.rect_selector = None
        self._pan_active = False
        self._pan_start_xy = None
        self._pan_start_xlim = None
        self._pan_start_ylim = None

        self._build_top_panel()
        self._build_main_panel()

    # ---------------------------------------------------------------- top panel

    def _build_top_panel(self):
        top = ttk.Frame(self, padding=8)
        top.pack(side="top", fill="x")

        ttk.Label(top, text="Dossier images:").grid(row=0, column=0, sticky="w")
        self.images_root_var = tk.StringVar(value=self.images_root)
        ttk.Entry(top, textvariable=self.images_root_var, width=60).grid(row=0, column=1, columnspan=3, sticky="we", padx=4)
        ttk.Button(top, text="Parcourir...", command=self._browse_images_root).grid(row=0, column=4, sticky="w")

        ttk.Label(top, text="Ruche:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.hive_var = tk.StringVar()
        self.hive_combo = ttk.Combobox(top, textvariable=self.hive_var, state="readonly", width=8)
        self.hive_combo.grid(row=1, column=1, sticky="w", pady=(4, 0))

        ttk.Label(top, text="Date/heure (UTC):").grid(row=1, column=2, sticky="w", padx=(12, 0), pady=(4, 0))
        now_utc = dt.datetime.now(ZoneInfo("UTC"))

        datetime_frame = ttk.Frame(top)
        datetime_frame.grid(row=1, column=3, sticky="w", pady=(4, 0))
        self.date_entry = SimpleDatePicker(datetime_frame, initial_date=now_utc.date())
        self.date_entry.pack(side="left")

        self.hour_var = tk.StringVar(value=f"{now_utc.hour:02d}")
        hour_spin = ttk.Spinbox(datetime_frame, from_=0, to=23, width=3, textvariable=self.hour_var, wrap=True)
        hour_spin.pack(side="left", padx=(6, 0))
        hour_spin.bind("<Return>", lambda e: self._commit_hour())
        hour_spin.bind("<FocusOut>", lambda e: self._commit_hour())

        ttk.Label(datetime_frame, text=":").pack(side="left")

        self.minute_var = tk.StringVar(value=f"{now_utc.minute:02d}")
        minute_spin = ttk.Spinbox(datetime_frame, from_=0, to=59, width=3, textvariable=self.minute_var, wrap=True)
        minute_spin.pack(side="left")
        minute_spin.bind("<Return>", lambda e: self._commit_minute())
        minute_spin.bind("<FocusOut>", lambda e: self._commit_minute())

        ttk.Button(top, text="Charger", command=self.on_load_click).grid(row=1, column=4, sticky="w", padx=(12, 0), pady=(4, 0))

        self._refresh_hive_list()

    def _browse_images_root(self):
        chosen = filedialog.askdirectory(
            title="Choisir le dossier images",
            initialdir=self.images_root_var.get() or os.path.expanduser("~"),
            mustexist=True,
        )
        if chosen:
            self.images_root_var.set(chosen)
            self._refresh_hive_list()

    def _refresh_hive_list(self):
        hives = list_hives(self.images_root_var.get())
        self.hive_combo["values"] = [str(h) for h in hives]
        if hives:
            self.hive_var.set(str(hives[0]))

    def _commit_hour(self):
        try:
            hour = int(self.hour_var.get())
        except (TypeError, ValueError):
            hour = 0
        self.hour_var.set(f"{max(0, min(23, hour)):02d}")

    def _commit_minute(self):
        try:
            minute = int(self.minute_var.get())
        except (TypeError, ValueError):
            minute = 0
        self.minute_var.set(f"{max(0, min(59, minute)):02d}")

    # --------------------------------------------------------------- main panel

    def _build_main_panel(self):
        main = ttk.Frame(self)
        main.pack(side="top", fill="both", expand=True)

        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)

        self.tabs_frame = ttk.Frame(left)
        self.tabs_frame.pack(side="top", fill="x", pady=(4, 0))
        self.tab_buttons = []

        mode_frame = ttk.Frame(left)
        mode_frame.pack(side="top", fill="x", pady=4)
        self.mode_var = tk.StringVar(value="navigate")
        for value, label in [("navigate", "Naviguer"), ("zone", "Dessiner la zone"), ("count", "Compter les abeilles")]:
            ttk.Radiobutton(mode_frame, text=label, value=value, variable=self.mode_var,
                             command=self._on_mode_change).pack(side="left", padx=4)
        ttk.Button(mode_frame, text="Annuler dernier point", command=self._undo_last_point).pack(side="left", padx=(20, 4))
        ttk.Button(mode_frame, text="Effacer tous les points", command=self._clear_points).pack(side="left", padx=4)

        self.figure = Figure(figsize=(9, 7))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=left)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.toolbar = SimpleToolbar(self.canvas, left)
        self.toolbar.update()
        self.canvas.mpl_connect("button_press_event", self._on_canvas_button_press)
        self.canvas.mpl_connect("button_release_event", self._on_canvas_button_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

        self.status_var = tk.StringVar(value="Chargez une ruche et un timestamp pour commencer.")
        ttk.Label(left, textvariable=self.status_var, anchor="w").pack(side="top", fill="x", padx=4, pady=(0, 4))

        right = ttk.Frame(main, padding=8, width=320)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        self._build_side_panel(right)

    def _build_side_panel(self, parent):
        ttk.Label(parent, text="Caméra courante", font=("", 11, "bold")).pack(anchor="w")
        self.camera_label_var = tk.StringVar(value="-")
        ttk.Label(parent, textvariable=self.camera_label_var).pack(anchor="w", pady=(0, 10))

        ttk.Label(parent, text="Zone sélectionnée (optionnelle):").pack(anchor="w")
        self.zone_info_var = tk.StringVar(value="(aucune)")
        ttk.Label(parent, textvariable=self.zone_info_var, wraplength=280, justify="left").pack(anchor="w")
        ttk.Button(parent, text="Supprimer la zone", command=self._clear_zone).pack(anchor="w", pady=(2, 10))

        ttk.Label(parent, text="Abeilles comptées:").pack(anchor="w")
        self.count_var = tk.StringVar(value="0")
        ttk.Label(parent, textvariable=self.count_var, font=("", 14, "bold")).pack(anchor="w", pady=(0, 10))

        ttk.Label(parent, text="Suggestion (aire image / aire zone):").pack(anchor="w")
        self.suggestion_var = tk.StringVar(value="-")
        ttk.Label(parent, textvariable=self.suggestion_var).pack(anchor="w")
        ttk.Button(parent, text="Copier la suggestion → facteur", command=self._copy_suggestion_to_factor).pack(anchor="w", pady=(2, 10))

        ttk.Label(parent, text="Facteur de multiplication (à vous de le fixer):").pack(anchor="w")
        self.factor_var = tk.StringVar(value="")
        self.factor_var.trace_add("write", lambda *_: self._on_factor_change())
        ttk.Entry(parent, textvariable=self.factor_var, width=12).pack(anchor="w", pady=(0, 10))

        ttk.Label(parent, text="Estimation sur cette image:").pack(anchor="w")
        self.estimate_var = tk.StringVar(value="-")
        ttk.Label(parent, textvariable=self.estimate_var, font=("", 14, "bold")).pack(anchor="w", pady=(0, 10))

        ttk.Button(parent, text="Valider cette caméra", command=self._validate_camera).pack(anchor="w", pady=(0, 20))

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(parent, text="Résumé de la session", font=("", 11, "bold")).pack(anchor="w")
        self.summary_var = tk.StringVar(value="-")
        ttk.Label(parent, textvariable=self.summary_var, wraplength=280, justify="left").pack(anchor="w", pady=(4, 10))
        ttk.Button(parent, text="Enregistrer la session (CSV)", command=self._save_session).pack(anchor="w")

    # -------------------------------------------------------------------- load

    def on_load_click(self):
        self.images_root = self.images_root_var.get().strip()
        if not self.hive_var.get():
            messagebox.showerror("Erreur", "Aucune ruche disponible dans ce dossier d'images.")
            return
        hive_nb = int(self.hive_var.get())

        date_val = self.date_entry.get_date()
        try:
            hour = int(self.hour_var.get())
            minute = int(self.minute_var.get())
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError
        except (ValueError, TypeError):
            messagebox.showerror("Erreur", f"Heure invalide: {self.hour_var.get()}:{self.minute_var.get()}")
            return

        try:
            ts = pd.Timestamp(dt.datetime(date_val.year, date_val.month, date_val.day, hour, minute))
            ts = ts.tz_localize("UTC")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Erreur", f"Date/heure invalide: {exc}")
            return

        try:
            camera_paths = find_camera_images(hive_nb, ts, self.images_root)
        except Exception as exc:  # noqa: BLE001 - surface any fetch error to the user
            messagebox.showerror("Erreur lors de la recherche des images", str(exc))
            return

        n_missing = sum(1 for p in camera_paths.values() if p is None)
        if n_missing == len(camera_paths):
            messagebox.showerror("Erreur", "Aucune image trouvée pour cette ruche/timestamp.")
            return
        if n_missing:
            messagebox.showwarning("Images manquantes", f"{n_missing} caméra(s) sans image disponible pour ce timestamp.")

        self.hive_nb = hive_nb
        self.requested_timestamp = ts
        self.session_id = uuid.uuid4().hex[:12]
        self.camera_states = [CameraState(name, path) for name, path in sorted(camera_paths.items())]

        self._build_tabs()
        self._show_camera(0)
        self._update_summary()
        self.status_var.set(f"Session {self.session_id} chargée: ruche {hive_nb}, {ts}.")

    def _build_tabs(self):
        for widget in self.tabs_frame.winfo_children():
            widget.destroy()
        self.tab_buttons = []
        for idx, state in enumerate(self.camera_states):
            text = state.camera_name + (" (indisponible)" if state.path is None else "")
            btn = ttk.Button(self.tabs_frame, text=text, command=lambda i=idx: self._show_camera(i))
            btn.pack(side="left", padx=2)
            btn.configure(state="disabled" if state.path is None else "normal")
            self.tab_buttons.append(btn)

    # --------------------------------------------------------------- rendering

    def _show_camera(self, idx: int):
        if self.current_camera_idx is not None:
            self._stash_factor_entry()
        self.current_camera_idx = idx
        self._render_camera()
        self._load_factor_entry()
        self._update_camera_side_panel()
        self._update_tab_styles()

    def _current_state(self) -> CameraState:
        return self.camera_states[self.current_camera_idx]

    def _render_camera(self):
        state = self._current_state()
        self.ax.cla()

        try:
            img = state.get_image()
        except IOError as exc:
            self.status_var.set(str(exc))
            self.canvas.draw_idle()
            return

        self.ax.imshow(img, cmap="gray")
        self.ax.set_title(state.camera_name)
        h, w = img.shape[:2]

        # (Re)create the rectangle selector for this axes; ax.cla() destroyed the previous one.
        self.rect_selector = RectangleSelector(self.ax, self._on_zone_select, **ZONE_SELECTOR_KWARGS)
        self.rect_selector.set_active(self.interaction_mode == "zone")
        if state.zone is not None:
            x0, y0, x1, y1 = state.zone
            try:
                self.rect_selector.extents = (x0, x1, y0, y1)
            except Exception:  # noqa: BLE001 - purely a visual convenience, safe to skip
                pass

        # Re-plot the counted points as fresh artists (old ones died with ax.cla()).
        state.point_artists = []
        for x, y in state.points:
            (artist,) = self.ax.plot([x], [y], **POINT_MARKER_STYLE)
            state.point_artists.append(artist)

        self.canvas.draw_idle()

        # Seed the toolbar's "home" view with this freshly-drawn full extent, so the
        # "Reset original view" button has something correct to reset back to. Our
        # custom scroll-zoom / right-click-pan never touch the toolbar's nav stack
        # (toolbar.update() only clears it, it doesn't record the current view), so
        # without this, clicking Home was a no-op.
        self.toolbar.update()
        self.toolbar.push_current()
        self.camera_label_var.set(f"{state.camera_name}  ({w}x{h} px)  -  {os.path.basename(state.path)}")

    def _update_tab_styles(self):
        for idx, (state, btn) in enumerate(zip(self.camera_states, self.tab_buttons)):
            prefix = "✓ " if state.validated else ""
            marker = " <" if idx == self.current_camera_idx else ""
            text = prefix + state.camera_name + (" (indisponible)" if state.path is None else "") + marker
            btn.configure(text=text)

    # ------------------------------------------------------------------ modes

    def _on_mode_change(self):
        self.interaction_mode = self.mode_var.get()
        if self.rect_selector is not None:
            self.rect_selector.set_active(self.interaction_mode == "zone")

    def _on_zone_select(self, eclick, erelease):
        state = self._current_state()
        x0, x1 = sorted([eclick.xdata, erelease.xdata])
        y0, y1 = sorted([eclick.ydata, erelease.ydata])
        state.zone = (x0, y0, x1, y1)
        self._update_camera_side_panel()

    def _clear_zone(self):
        if self.current_camera_idx is None:
            return
        self._current_state().zone = None
        if self.rect_selector is not None:
            self.rect_selector.clear()
        self.canvas.draw_idle()
        self._update_camera_side_panel()

    def _on_canvas_button_press(self, event):
        if event.button == 3:
            if event.inaxes == self.ax:
                self._pan_active = True
                self._pan_start_xy = (event.x, event.y)
                self._pan_start_xlim = self.ax.get_xlim()
                self._pan_start_ylim = self.ax.get_ylim()
            return
        if event.button == 1:
            self._on_canvas_click(event)

    def _on_canvas_button_release(self, event):
        if event.button == 3:
            self._pan_active = False

    def _on_canvas_motion(self, event):
        if not self._pan_active or event.x is None or event.y is None:
            return
        inv = self.ax.transData.inverted()
        x0_data, y0_data = inv.transform(self._pan_start_xy)
        x1_data, y1_data = inv.transform((event.x, event.y))
        dx = x1_data - x0_data
        dy = y1_data - y0_data
        xlim = self._pan_start_xlim
        ylim = self._pan_start_ylim
        self.ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
        self.ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
        self.canvas.draw_idle()

    def _on_canvas_click(self, event):
        if self.interaction_mode != "count" or self.current_camera_idx is None:
            return
        if self.toolbar.mode != "":
            return  # pan/zoom tool active, don't count this click
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        state = self._current_state()
        (artist,) = self.ax.plot([event.xdata], [event.ydata], **POINT_MARKER_STYLE)
        state.points.append((event.xdata, event.ydata))
        state.point_artists.append(artist)
        self.canvas.draw_idle()
        self._update_camera_side_panel()

    def _on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        if event.button == "up":
            scale_factor = 1 / 1.2
        elif event.button == "down":
            scale_factor = 1.2
        else:
            return

        xdata, ydata = event.xdata, event.ydata
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()

        relx = (x1 - xdata) / (x1 - x0)
        rely = (y1 - ydata) / (y1 - y0)
        new_width = (x1 - x0) * scale_factor
        new_height = (y1 - y0) * scale_factor

        self.ax.set_xlim(xdata - new_width * (1 - relx), xdata + new_width * relx)
        self.ax.set_ylim(ydata - new_height * (1 - rely), ydata + new_height * rely)
        self.canvas.draw_idle()

    def _undo_last_point(self):
        if self.current_camera_idx is None:
            return
        state = self._current_state()
        if not state.points:
            return
        state.points.pop()
        artist = state.point_artists.pop()
        artist.remove()
        self.canvas.draw_idle()
        self._update_camera_side_panel()

    def _clear_points(self):
        if self.current_camera_idx is None:
            return
        state = self._current_state()
        for artist in state.point_artists:
            artist.remove()
        state.points = []
        state.point_artists = []
        self.canvas.draw_idle()
        self._update_camera_side_panel()

    # --------------------------------------------------------------- side panel

    def _stash_factor_entry(self):
        self._current_state().factor_text = self.factor_var.get()

    def _load_factor_entry(self):
        self.factor_var.set(self._current_state().factor_text)

    def _on_factor_change(self):
        if self.current_camera_idx is not None:
            self._current_state().factor_text = self.factor_var.get()
        self._update_estimate()

    def _zone_and_image_area(self, state: CameraState):
        if state.zone is None or state.image is None:
            return None, None
        x0, y0, x1, y1 = state.zone
        zone_area = abs(x1 - x0) * abs(y1 - y0)
        h, w = state.image.shape[:2]
        image_area = h * w
        return zone_area, image_area

    def _update_camera_side_panel(self):
        state = self._current_state()

        if state.zone is not None:
            x0, y0, x1, y1 = state.zone
            self.zone_info_var.set(f"({x0:.0f}, {y0:.0f}) → ({x1:.0f}, {y1:.0f})  |  {abs(x1-x0):.0f}x{abs(y1-y0):.0f} px")
        else:
            self.zone_info_var.set("(aucune)")

        self.count_var.set(str(len(state.points)))

        zone_area, image_area = self._zone_and_image_area(state)
        if zone_area:
            self.suggestion_var.set(f"{image_area / zone_area:.2f}  (image={image_area:.0f}px² / zone={zone_area:.0f}px²)")
        else:
            self.suggestion_var.set("-")

        self._update_estimate()

    def _update_estimate(self):
        if self.current_camera_idx is None:
            self.estimate_var.set("-")
            return
        state = self._current_state()
        try:
            factor = float(self.factor_var.get())
        except (TypeError, ValueError):
            self.estimate_var.set("-")
            return
        estimate = len(state.points) * factor
        self.estimate_var.set(f"{estimate:.1f}")

    def _copy_suggestion_to_factor(self):
        state = self._current_state()
        zone_area, image_area = self._zone_and_image_area(state)
        if not zone_area:
            messagebox.showinfo("Info", "Dessinez d'abord une zone.")
            return
        self.factor_var.set(f"{image_area / zone_area:.4f}")

    # ------------------------------------------------------------------ validate

    def _validate_camera(self):
        state = self._current_state()
        try:
            factor = float(self.factor_var.get())
            if factor <= 0:
                raise ValueError
        except (TypeError, ValueError):
            messagebox.showerror("Erreur", "Entrez un facteur de multiplication valide (nombre > 0).")
            return

        state.validated = True
        self._update_tab_styles()
        self._update_summary()
        self.status_var.set(f"Caméra {state.camera_name} validée: {len(state.points)} abeilles x {factor} = {len(state.points) * factor:.1f}.")

    def _update_summary(self):
        lines = []
        total = 0.0
        n_validated = 0
        for state in self.camera_states:
            if state.validated:
                factor = float(state.factor_text)
                estimate = len(state.points) * factor
                total += estimate
                n_validated += 1
                lines.append(f"{state.camera_name}: {len(state.points)} x {factor:.3g} = {estimate:.1f}")
            elif state.path is not None:
                lines.append(f"{state.camera_name}: (non validée)")
        lines.append(f"\nTotal ruche ({n_validated}/{len(self.camera_states)} caméras validées): {total:.1f} abeilles")
        self.summary_var.set("\n".join(lines))

    # ---------------------------------------------------------------------- save

    def _save_session(self):
        rows = []
        saved_at = dt.datetime.now(ZoneInfo("UTC")).isoformat()
        for state in self.camera_states:
            if not state.validated:
                continue
            x0, y0, x1, y1 = state.zone if state.zone is not None else (None, None, None, None)
            zone_area, image_area = self._zone_and_image_area(state)
            factor = float(state.factor_text)
            rows.append({
                "session_id": self.session_id,
                "saved_at": saved_at,
                "hive": self.hive_nb,
                "requested_timestamp": self.requested_timestamp.isoformat(),
                "camera": state.camera_name,
                "image_path": state.path,
                "image_width_px": state.image.shape[1],
                "image_height_px": state.image.shape[0],
                "zone_x0": x0, "zone_y0": y0, "zone_x1": x1, "zone_y1": y1,
                "zone_area_px": zone_area,
                "image_area_px": image_area,
                "area_ratio_suggestion": (image_area / zone_area) if zone_area else None,
                "bee_count": len(state.points),
                "multiplication_factor": factor,
                "estimated_total_bees": len(state.points) * factor,
            })

        if not rows:
            messagebox.showerror("Erreur", "Aucune caméra validée, rien à enregistrer.")
            return

        n_total = len(self.camera_states)
        if len(rows) < n_total:
            proceed = messagebox.askyesno(
                "Caméras incomplètes",
                f"Seulement {len(rows)}/{n_total} caméras sont validées. Enregistrer quand même ?",
            )
            if not proceed:
                return

        append_results(rows)
        messagebox.showinfo("Enregistré", f"{len(rows)} ligne(s) ajoutée(s) au CSV de résultats.")
        self.status_var.set(f"Session {self.session_id} enregistrée ({len(rows)} caméra(s)).")


if __name__ == "__main__":
    app = BeeLabelApp()
    app.mainloop()
