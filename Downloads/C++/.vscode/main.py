"""
Quantum Harmonic Oscillator Visualizer
========================================
A beginner-friendly Physics DA (Data Analysis / Demonstration Activity)
project that visually demonstrates the core idea of the quantum harmonic
oscillator: unlike a classical particle, a quantum particle trapped in a
harmonic potential can only have certain DISCRETE (quantized) energy
values, and its position is described by a probability distribution
rather than a single fixed trajectory.

Extra features (beyond the basic requirement) included for a stronger
presentation:
    - A textbook-style "particle in a well" energy diagram: quantized
      levels drawn INSIDE the actual parabolic potential V(x) = 0.5x^2,
      each level spanning its own classical turning points.
    - A "Compare States" mode that overlays a second quantum number n2
      on both the wavefunction and probability graphs, so a viewer can
      directly see how node count and spread change with n.
    - A "Classical vs Quantum" overlay on the probability density graph,
      showing the classical particle-on-a-spring probability distribution
      next to the quantum one -- a very effective visual for a viva.
    - A "Play Animation" button that automatically steps n = 0 -> 5,
      handy for a live demonstration.
    - A live status readout of node count and classical turning points.
    - A Matplotlib navigation toolbar (zoom / pan / save figure).
    - A "Save Graphs as Image" button for including plots in a report.
    - Keyboard shortcuts: click a slider, then use Left/Right arrow keys to
      change it -- the main "n" slider and the "Compare n2" slider are
      controlled independently based on which one has keyboard focus.
    - A SCROLLABLE sidebar, so controls are never clipped or cut off,
      no matter how small the screen or window is resized.

Run with:
    python main.py
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
# Wrapped in try/except so a missing library gives a clear, friendly
# message instead of a confusing traceback.
# ---------------------------------------------------------------------------
import sys

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
except ImportError:
    print("ERROR: 'tkinter' is not installed.\n"
          "tkinter usually comes with Python, but on some Linux systems "
          "you must install it separately, e.g.:\n"
          "    sudo apt-get install python3-tk\n")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: 'numpy' is not installed.\n"
          "Please install the required libraries first:\n"
          "    pip install -r requirements.txt")
    sys.exit(1)

try:
    from scipy.special import eval_hermite, factorial
except ImportError:
    print("ERROR: 'scipy' is not installed.\n"
          "Please install the required libraries first:\n"
          "    pip install -r requirements.txt")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("TkAgg")  # Use the Tkinter-compatible backend
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
except ImportError:
    print("ERROR: 'matplotlib' is not installed.\n"
          "Please install the required libraries first:\n"
          "    pip install -r requirements.txt")
    sys.exit(1)


# ---------------------------------------------------------------------------
# COLOR THEME (dark, modern, presentation-friendly)
# ---------------------------------------------------------------------------
BG_MAIN = "#1e1e2e"
BG_CARD = "#2a2a3d"
GRID_COLOR = "#44445a"
FG_TEXT = "#e8e8f0"
FG_SUBTEXT = "#a9a9c2"
ACCENT_COLOR = "#7aa2f7"     # blue  - primary quantum state
ACCENT_COLOR2 = "#f7768e"    # pink  - highlighted / selected level
ACCENT_COLOR3 = "#9ece6a"    # green - comparison quantum state
ACCENT_COLOR4 = "#e0af68"    # gold  - classical comparison curve

FONT_TITLE = ("Segoe UI", 20, "bold")
FONT_HEADER = ("Segoe UI", 13, "bold")
FONT_NORMAL = ("Segoe UI", 11)
FONT_SMALL = ("Segoe UI", 9)
FONT_MONO = ("Consolas", 12, "bold")


# ---------------------------------------------------------------------------
# PHYSICS MODEL
# ---------------------------------------------------------------------------
class QuantumHarmonicOscillator:
    """
    Encapsulates the physics of the 1D Quantum Harmonic Oscillator (QHO).

    Normalized (natural) units are used, a standard convention in
    introductory quantum mechanics courses:
        hbar (reduced Planck's constant) = 1
        omega (angular frequency)        = 1
        m (mass of the particle)         = 1

    Energy:        E_n = (n + 1/2) hbar * omega  =  (n + 1/2)
    Wavefunction:  psi_n(x) = [1 / sqrt(2^n n!)] * (1/pi^(1/4))
                                * H_n(x) * exp(-x^2 / 2)
    where H_n(x) is the n-th physicists' Hermite polynomial.
    """

    def __init__(self, hbar=1.0, omega=1.0, mass=1.0):
        self.hbar = hbar
        self.omega = omega
        self.mass = mass

    def energy(self, n: int) -> float:
        """Return the quantized energy E_n for quantum number n."""
        return (n + 0.5) * self.hbar * self.omega

    def turning_point(self, n: int) -> float:
        """
        Classical turning point amplitude A for a classical oscillator
        that has the SAME energy as quantum state n. Comes from
        E_n = 0.5 * m * omega^2 * A^2  =>  A = sqrt(2 * E_n / (m*omega^2)).
        In normalized units (m = omega = 1): A = sqrt(2 * E_n).
        """
        return np.sqrt(2.0 * self.energy(n) / (self.mass * self.omega ** 2))

    def potential(self, x: np.ndarray) -> np.ndarray:
        """Classical harmonic potential V(x) = 0.5 * m * omega^2 * x^2."""
        return 0.5 * self.mass * self.omega ** 2 * x ** 2

    def wavefunction(self, n: int, x: np.ndarray) -> np.ndarray:
        """Return the normalized wavefunction psi_n(x) at positions x."""
        hermite_part = eval_hermite(n, x)
        normalization = 1.0 / np.sqrt((2.0 ** n) * factorial(n)) \
            * (1.0 / np.pi ** 0.25)
        gaussian = np.exp(-x ** 2 / 2.0)
        return normalization * hermite_part * gaussian

    def probability_density(self, n: int, x: np.ndarray) -> np.ndarray:
        """Return |psi_n(x)|^2, the quantum probability density."""
        psi = self.wavefunction(n, x)
        return psi ** 2

    def classical_probability_density(self, n: int, x: np.ndarray,
                                       epsilon: float = 1e-3) -> np.ndarray:
        """
        Return the probability density of a CLASSICAL oscillator that has
        the same total energy as quantum state n. A classical mass on a
        spring spends more time near its turning points (where it moves
        slowly) than near the center (where it moves fastest), giving:

            P_classical(x) = 1 / (pi * sqrt(A^2 - x^2))   for |x| < A

        This is the standard normalized classical distribution. A small
        epsilon is used to prevent a division-by-zero blow-up exactly at
        the turning points, purely for a well-behaved plot.
        """
        A = self.turning_point(n)
        denom = np.sqrt(np.maximum(A ** 2 - x ** 2, epsilon ** 2))
        return np.where(np.abs(x) < A, 1.0 / (np.pi * denom), 0.0)

    @staticmethod
    def node_count(n: int) -> int:
        """The wavefunction psi_n has exactly n nodes (zero crossings)."""
        return n


# ---------------------------------------------------------------------------
# SCROLLABLE FRAME HELPER
# ---------------------------------------------------------------------------
class ScrollableFrame(ttk.Frame):
    """
    A frame that scrolls vertically when its content is taller than the
    visible area. This is the key fix for the sidebar controls getting
    clipped/cut off on smaller windows or screens: instead of the last
    card being squeezed into an unreadable sliver, the user can now
    simply scroll down to see everything.
    """

    def __init__(self, parent, width=340, bg=BG_MAIN):
        super().__init__(parent, width=width)
        self.pack_propagate(False)

        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, width=width)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # IMPORTANT: pack the scrollbar FIRST, then the canvas.
        # If the canvas (which has expand=True) is packed first, it claims
        # the entire available space immediately, leaving nothing for the
        # scrollbar to occupy -- it gets squeezed to zero width and
        # silently disappears, which is exactly what caused the sidebar
        # to look "cropped" with no way to reach the rest of the content.
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        # This inner frame holds the actual content (the cards).
        self.content = ttk.Frame(self.canvas, style="TFrame")
        self.window_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")

        self.content.bind("<Configure>", self._on_content_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Mouse-wheel scrolling (Windows/Mac use <MouseWheel>, Linux uses Button-4/5)
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

    def _on_content_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        # Make the inner content frame always match the canvas width
        self.canvas.itemconfig(self.window_id, width=event.width)

    def _bind_mousewheel(self, event):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)   # Windows / macOS
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _unbind_mousewheel(self, event):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_linux(self, event):
        direction = -1 if event.num == 4 else 1
        self.canvas.yview_scroll(direction, "units")


# ---------------------------------------------------------------------------
# CUSTOM TOOLBAR (Back/Forward removed)
# ---------------------------------------------------------------------------
class CleanToolbar(NavigationToolbar2Tk):
    """
    A trimmed-down version of Matplotlib's navigation toolbar.

    The default toolbar includes "Back" and "Forward" history buttons that
    start out disabled (since there is no zoom/pan history yet). Tkinter
    renders disabled icon buttons with a washed-out hatched pattern, which
    can look like a rendering glitch against a dark theme. Since these two
    buttons add little value for this demo anyway, they are simply left
    out -- Home, Pan, Zoom, Subplots, and Save are kept.
    """
    toolitems = [t for t in NavigationToolbar2Tk.toolitems if t[0] not in ("Back", "Forward")]


# ---------------------------------------------------------------------------
# MAIN APPLICATION (GUI)
# ---------------------------------------------------------------------------
class QHOApp(tk.Tk):
    """The main Tkinter application window."""

    MAX_N = 5
    X_RANGE = 6.0
    NUM_POINTS = 500
    ANIMATION_DELAY_MS = 900

    def __init__(self):
        super().__init__()

        # --- Window setup ---------------------------------------------------
        self.title("Quantum Harmonic Oscillator Visualizer")
        self.geometry("1320x800")
        self.minsize(1000, 600)
        self.configure(bg=BG_MAIN)

        # --- Physics engine and state ---------------------------------------
        self.qho = QuantumHarmonicOscillator(hbar=1.0, omega=1.0, mass=1.0)
        self.current_n = tk.IntVar(value=0)
        self.compare_n = tk.IntVar(value=2)
        self.show_probability = True
        self.compare_enabled = tk.BooleanVar(value=False)
        self.classical_overlay = tk.BooleanVar(value=False)
        self.animating = False
        self._animation_job = None
        self.x_values = np.linspace(-self.X_RANGE, self.X_RANGE, self.NUM_POINTS)

        # --- Build the interface --------------------------------------------
        self._configure_styles()
        self._build_layout()
        self._bind_shortcuts()

        self._update_all()

        self.protocol("WM_DELETE_WINDOW", self._on_exit)

    # -------------------------------------------------------------------
    # STYLE CONFIGURATION
    # -------------------------------------------------------------------
    def _configure_styles(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background=BG_MAIN)
        style.configure("Card.TFrame", background=BG_CARD)

        style.configure("TLabel", background=BG_MAIN, foreground=FG_TEXT, font=FONT_NORMAL)
        style.configure("Card.TLabel", background=BG_CARD, foreground=FG_TEXT, font=FONT_NORMAL)
        style.configure("Header.TLabel", background=BG_CARD, foreground=FG_TEXT, font=FONT_HEADER)
        style.configure("Title.TLabel", background=BG_MAIN, foreground=FG_TEXT, font=FONT_TITLE)
        style.configure("Sub.TLabel", background=BG_MAIN, foreground=FG_SUBTEXT, font=FONT_SMALL)
        style.configure("Mono.TLabel", background=BG_CARD, foreground=ACCENT_COLOR, font=FONT_MONO)
        style.configure("Mono2.TLabel", background=BG_CARD, foreground=ACCENT_COLOR3, font=FONT_MONO)
        style.configure("Explain.TLabel", background=BG_CARD, foreground=FG_SUBTEXT,
                         font=FONT_NORMAL, wraplength=270, justify="left")
        style.configure("Status.TLabel", background=BG_CARD, foreground=FG_SUBTEXT,
                         font=("Consolas", 10))

        style.configure("Horizontal.TScale", background=BG_CARD)
        style.configure("TCheckbutton", background=BG_CARD, foreground=FG_TEXT, font=FONT_NORMAL)
        style.map("TCheckbutton", background=[("active", BG_CARD)])

        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=8)
        style.map("Accent.TButton", background=[("!disabled", ACCENT_COLOR)],
                  foreground=[("!disabled", "#101018")])

        style.configure("Play.TButton", font=("Segoe UI", 10, "bold"), padding=8)
        style.map("Play.TButton", background=[("!disabled", ACCENT_COLOR3)],
                  foreground=[("!disabled", "#101018")])

        style.configure("Danger.TButton", font=("Segoe UI", 10, "bold"), padding=8)
        style.map("Danger.TButton", background=[("!disabled", ACCENT_COLOR2)],
                  foreground=[("!disabled", "#101018")])

        style.configure("Vertical.TScrollbar", background=BG_CARD, troughcolor=BG_MAIN,
                         arrowcolor=FG_TEXT)

    # -------------------------------------------------------------------
    # LAYOUT CONSTRUCTION
    # -------------------------------------------------------------------
    def _build_layout(self):
        # ---- Title bar ------------------------------------------------
        title_frame = ttk.Frame(self, style="TFrame")
        title_frame.pack(side="top", fill="x", padx=20, pady=(15, 5))

        ttk.Label(title_frame, text="Quantum Harmonic Oscillator Visualizer",
                  style="Title.TLabel").pack(anchor="w")
        ttk.Label(title_frame,
                  text="Interactive demonstration of energy quantization  |  "
                       "Normalized units used: \u0127 = 1, \u03c9 = 1, m = 1",
                  style="Sub.TLabel").pack(anchor="w", pady=(2, 0))

        # ---- Body: sidebar (left, scrollable) + plots (right) ----------
        body = ttk.Frame(self, style="TFrame")
        body.pack(side="top", fill="both", expand=True, padx=20, pady=10)

        # The scrollable sidebar wrapper -- FIXES the clipped-button bug.
        self.sidebar_wrapper = ScrollableFrame(body, width=340, bg=BG_MAIN)
        self.sidebar_wrapper.pack(side="left", fill="y", padx=(0, 15))
        self.sidebar = self.sidebar_wrapper.content

        self.plot_area = ttk.Frame(body, style="TFrame")
        self.plot_area.pack(side="left", fill="both", expand=True)

        self._build_sidebar()
        self._build_plots()

    # ---- small helper for consistent card creation -----------------------
    def _new_card(self, title):
        card = tk.Frame(self.sidebar, bg=BG_CARD, highlightbackground=GRID_COLOR,
                         highlightthickness=1)
        card.pack(side="top", fill="x", pady=(0, 12), padx=(2, 6))
        inner = ttk.Frame(card, style="Card.TFrame")
        inner.pack(fill="x", padx=14, pady=12)
        ttk.Label(inner, text=title, style="Header.TLabel").pack(anchor="w")
        return inner

    # ---- Sidebar ------------------------------------------------------
    def _build_sidebar(self):
        # ============ CARD 1: Quantum number control ============
        c1 = self._new_card("Energy Level Selector")

        self.n_label = ttk.Label(c1, text="Quantum Number: n = 0", style="Mono.TLabel")
        self.n_label.pack(anchor="w", pady=(8, 4))

        self.n_slider = ttk.Scale(c1, from_=0, to=self.MAX_N, orient="horizontal",
                                   command=self._on_slider_change)
        self.n_slider.set(0)
        self.n_slider.pack(fill="x", pady=(0, 4))

        ttk.Label(c1, text=f"(range: n = 0 to {self.MAX_N}  |  click the slider, then "
                            f"use \u2190/\u2192 keys)",
                  style="Explain.TLabel").pack(anchor="w")

        # ============ CARD 2: Energy information ============
        c2 = self._new_card("Energy Information")

        ttk.Label(c2, text="E\u2099 = (n + 1/2) \u0127\u03c9", style="Card.TLabel").pack(
            anchor="w", pady=(8, 2))
        self.energy_value_label = ttk.Label(c2, text="E\u2080 = 0.500", style="Mono.TLabel")
        self.energy_value_label.pack(anchor="w", pady=(2, 6))

        self.status_label = ttk.Label(c2, text="Nodes: 0  |  Turning point: \u00b11.00",
                                       style="Status.TLabel")
        self.status_label.pack(anchor="w", pady=(0, 6))

        ttk.Label(c2, text="The energy is QUANTIZED: only these specific values are "
                            "allowed, never anything in between (unlike a classical "
                            "oscillator, which can have ANY energy).",
                  style="Explain.TLabel").pack(anchor="w")

        # ============ CARD 3: Compare States (interactive) ============
        c3 = self._new_card("Compare Two States")

        ttk.Checkbutton(c3, text="Overlay a second state n\u2082", variable=self.compare_enabled,
                         command=self._update_all).pack(anchor="w", pady=(8, 4))

        self.compare_label = ttk.Label(c3, text="n\u2082 = 2", style="Mono2.TLabel")
        self.compare_label.pack(anchor="w", pady=(2, 2))

        self.compare_slider = ttk.Scale(c3, from_=0, to=self.MAX_N, orient="horizontal",
                                         command=self._on_compare_slider_change)
        self.compare_slider.set(2)
        self.compare_slider.pack(fill="x", pady=(0, 4))

        ttk.Label(c3, text="See how the number of nodes and the spread of the "
                            "wavefunction change between two states side by side. "
                            "Click this slider, then use \u2190/\u2192 keys to control n\u2082 "
                            "(the main n slider above is unaffected while this one "
                            "has focus).",
                  style="Explain.TLabel").pack(anchor="w")

        # ============ CARD 4: Classical vs Quantum ============
        c4 = self._new_card("Classical vs Quantum")

        ttk.Checkbutton(c4, text="Show classical probability curve",
                         variable=self.classical_overlay,
                         command=self._update_all).pack(anchor="w", pady=(8, 4))

        ttk.Label(c4, text="A classical mass on a spring spends more time near its "
                            "turning points (moving slowly) than at the center. "
                            "Compare that gold curve to the quantum result below "
                            "-- for low n they look nothing alike, but for high n "
                            "the quantum curve starts to resemble the classical one "
                            "(the correspondence principle).",
                  style="Explain.TLabel").pack(anchor="w")

        # ============ CARD 5: Simple explanation ============
        c5 = self._new_card("Simple Explanation")
        ttk.Label(c5, text="The particle cannot have any random energy. It can only "
                            "exist in specific energy states (n = 0, 1, 2, ...). The "
                            "wavefunction \u03c8\u2099(x) describes the quantum state, while "
                            "|\u03c8\u2099(x)|\u00b2 represents the probability density -- i.e. "
                            "how likely the particle is to be found at position x.",
                  style="Explain.TLabel").pack(anchor="w", pady=(8, 0))

        # ============ CARD 6: Controls ============
        c6 = self._new_card("Controls")

        self.animate_btn = ttk.Button(c6, text="\u25b6 Play Animation (n: 0\u21925)",
                                       style="Play.TButton", command=self._toggle_animation)
        self.animate_btn.pack(fill="x", pady=(8, 4))

        ttk.Button(c6, text="Reset", style="Accent.TButton",
                   command=self._on_reset).pack(fill="x", pady=4)

        self.toggle_btn = ttk.Button(c6, text="Hide Probability Density",
                                      style="Accent.TButton",
                                      command=self._on_toggle_probability)
        self.toggle_btn.pack(fill="x", pady=4)

        ttk.Button(c6, text="Save Graphs as Image...", style="Accent.TButton",
                   command=self._on_save_image).pack(fill="x", pady=4)

        ttk.Button(c6, text="Exit", style="Danger.TButton",
                   command=self._on_exit).pack(fill="x", pady=(4, 2))

    # ---- Plot area --------------------------------------------------------
    def _build_plots(self):
        plt.style.use("dark_background")

        # Left column (tall): textbook-style energy-level-in-a-well diagram
        # Right column: wavefunction (top) and probability density (bottom)
        self.fig = plt.Figure(figsize=(8.5, 7.0), dpi=100, facecolor=BG_MAIN)
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1, 1.6],
                                    height_ratios=[1, 1], hspace=0.45, wspace=0.3)

        self.ax_levels = self.fig.add_subplot(gs[:, 0])
        self.ax_wave = self.fig.add_subplot(gs[0, 1])
        self.ax_prob = self.fig.add_subplot(gs[1, 1])

        for ax in (self.ax_levels, self.ax_wave, self.ax_prob):
            ax.set_facecolor(BG_CARD)
            ax.tick_params(colors=FG_SUBTEXT)
            for spine in ax.spines.values():
                spine.set_color(GRID_COLOR)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_area)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        # Matplotlib navigation toolbar: lets students zoom/pan into a
        # wavefunction during the viva, and save a plot directly.
        toolbar_frame = tk.Frame(self.plot_area, bg=BG_MAIN)
        toolbar_frame.pack(side="bottom", fill="x")
        self.toolbar = CleanToolbar(self.canvas, toolbar_frame)
        try:
            self.toolbar.config(background=BG_MAIN)
            for child in self.toolbar.winfo_children():
                child.config(background=BG_MAIN)
        except tk.TclError:
            pass  # Some platforms restrict toolbar restyling; safe to skip.
        self.toolbar.update()

    # -------------------------------------------------------------------
    # KEYBOARD SHORTCUTS
    # -------------------------------------------------------------------
    def _bind_shortcuts(self):
        """
        Bind Left/Right arrow keys directly to EACH slider (not the whole
        window), so arrow keys control whichever slider currently has
        keyboard focus: click the main "n" slider and the arrows move n;
        click the "Compare" n2 slider and the arrows move n2 instead.

        Returning "break" stops the event from also reaching ttk.Scale's
        own built-in arrow-key handling, which would otherwise move the
        slider by a small fractional step in addition to our integer step.
        """
        self.n_slider.bind("<Left>", lambda e: self._on_arrow(self._step_n, -1))
        self.n_slider.bind("<Right>", lambda e: self._on_arrow(self._step_n, 1))

        self.compare_slider.bind("<Left>", lambda e: self._on_arrow(self._step_n2, -1))
        self.compare_slider.bind("<Right>", lambda e: self._on_arrow(self._step_n2, 1))

        # Clicking a slider gives it keyboard focus (ttk.Scale supports
        # this natively, but we set it explicitly to be safe).
        self.n_slider.bind("<Button-1>", lambda e: self.n_slider.focus_set(), add="+")
        self.compare_slider.bind("<Button-1>", lambda e: self.compare_slider.focus_set(), add="+")

    @staticmethod
    def _on_arrow(step_function, delta):
        step_function(delta)
        return "break"

    def _step_n(self, delta):
        n = max(0, min(self.MAX_N, self.current_n.get() + delta))
        self.current_n.set(n)
        self.n_slider.set(n)
        self._update_all()

    def _step_n2(self, delta):
        n2 = max(0, min(self.MAX_N, self.compare_n.get() + delta))
        self.compare_n.set(n2)
        self.compare_slider.set(n2)
        self.compare_label.config(text=f"n\u2082 = {n2}")
        if self.compare_enabled.get():
            self._update_all()

    # -------------------------------------------------------------------
    # EVENT HANDLERS
    # -------------------------------------------------------------------
    def _on_slider_change(self, value_str):
        try:
            n = int(round(float(value_str)))
        except (ValueError, TypeError):
            n = 0
        n = max(0, min(self.MAX_N, n))

        if abs(self.n_slider.get() - n) > 1e-6:
            self.n_slider.set(n)

        if n != self.current_n.get():
            self.current_n.set(n)
            self._update_all()

    def _on_compare_slider_change(self, value_str):
        try:
            n2 = int(round(float(value_str)))
        except (ValueError, TypeError):
            n2 = 0
        n2 = max(0, min(self.MAX_N, n2))

        if abs(self.compare_slider.get() - n2) > 1e-6:
            self.compare_slider.set(n2)

        if n2 != self.compare_n.get():
            self.compare_n.set(n2)
            self.compare_label.config(text=f"n\u2082 = {n2}")
            if self.compare_enabled.get():
                self._update_all()

    def _on_reset(self):
        self._stop_animation()
        self.current_n.set(0)
        self.n_slider.set(0)
        self.compare_n.set(2)
        self.compare_slider.set(2)
        self.compare_label.config(text="n\u2082 = 2")
        self.compare_enabled.set(False)
        self.classical_overlay.set(False)
        if not self.show_probability:
            self.show_probability = True
            self.toggle_btn.config(text="Hide Probability Density")
        self._update_all()

    def _on_toggle_probability(self):
        self.show_probability = not self.show_probability
        self.toggle_btn.config(
            text="Hide Probability Density" if self.show_probability
            else "Show Probability Density")
        self._update_all()

    def _on_save_image(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
            title="Save current graphs as an image")
        if not path:
            return
        try:
            self.fig.savefig(path, facecolor=self.fig.get_facecolor(), dpi=150)
            messagebox.showinfo("Saved", f"Graphs saved to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Save Failed", f"Could not save the image:\n{exc}")

    def _toggle_animation(self):
        self.animating = not self.animating
        if self.animating:
            self.animate_btn.config(text="\u23f8 Pause Animation")
            self._animate_step()
        else:
            self._stop_animation()

    def _stop_animation(self):
        self.animating = False
        self.animate_btn.config(text="\u25b6 Play Animation (n: 0\u21925)")
        if self._animation_job is not None:
            self.after_cancel(self._animation_job)
            self._animation_job = None

    def _animate_step(self):
        if not self.animating:
            return
        n = self.current_n.get()
        n_next = n + 1 if n < self.MAX_N else 0
        self.current_n.set(n_next)
        self.n_slider.set(n_next)
        self._update_all()
        self._animation_job = self.after(self.ANIMATION_DELAY_MS, self._animate_step)

    def _on_exit(self):
        self._stop_animation()
        answer = messagebox.askyesno("Exit", "Are you sure you want to exit?")
        if answer:
            plt.close("all")
            self.destroy()

    # -------------------------------------------------------------------
    # DRAWING / UPDATE LOGIC
    # -------------------------------------------------------------------
    def _update_all(self):
        try:
            n = self.current_n.get()
            n2 = self.compare_n.get()
            show_compare = self.compare_enabled.get()
            x = self.x_values

            energy = self.qho.energy(n)
            psi = self.qho.wavefunction(n, x)
            prob = self.qho.probability_density(n, x)
            turning = self.qho.turning_point(n)

            # --- Update text labels -----------------------------------
            self.n_label.config(text=f"Quantum Number: n = {n}")
            self.energy_value_label.config(
                text=f"E_{n} = {energy:.3f}  (in units of \u0127\u03c9)")
            self.status_label.config(
                text=f"Nodes: {self.qho.node_count(n)}  |  "
                     f"Turning point: \u00b1{turning:.2f}")

            # --- Redraw all three plots ---------------------------------
            self._draw_energy_levels(n, n2 if show_compare else None)
            self._draw_wavefunction(x, psi, n, n2 if show_compare else None)
            self._draw_probability(x, prob, n, n2 if show_compare else None)

            self.canvas.draw_idle()

        except Exception as exc:
            messagebox.showerror(
                "Unexpected Error",
                f"Something went wrong while updating the graphs:\n{exc}")

    def _draw_energy_levels(self, selected_n, compare_n):
        """
        Draw a textbook-style diagram: the parabolic potential V(x) = 0.5x^2
        with each allowed energy level drawn as a horizontal segment sitting
        INSIDE the potential well, spanning exactly its own classical
        turning points. This is physically accurate (V and E share the
        same units here) and instantly recognizable to anyone who has seen
        a QM textbook figure.
        """
        ax = self.ax_levels
        ax.clear()
        ax.set_facecolor(BG_CARD)

        levels_n = list(range(0, self.MAX_N + 1))
        levels_e = [self.qho.energy(n) for n in levels_n]
        max_turning = self.qho.turning_point(self.MAX_N)

        # Potential curve V(x) = 0.5 x^2, drawn slightly beyond the widest level
        xp = np.linspace(-max_turning * 1.25, max_turning * 1.25, 300)
        ax.plot(xp, self.qho.potential(xp), color=FG_SUBTEXT, linewidth=1.6,
                alpha=0.6, label="V(x) = \u00bd x\u00b2")

        for n, e in zip(levels_n, levels_e):
            A = self.qho.turning_point(n)
            is_selected = (n == selected_n)
            is_compared = (compare_n is not None and n == compare_n)

            if is_selected:
                color, lw, alpha = ACCENT_COLOR2, 3.2, 1.0
            elif is_compared:
                color, lw, alpha = ACCENT_COLOR3, 3.0, 1.0
            else:
                color, lw, alpha = ACCENT_COLOR, 1.6, 0.5

            ax.hlines(e, xmin=-A, xmax=A, color=color, linewidth=lw, alpha=alpha)
            # Label is stacked on two short lines ("n=5" / "E=5.50") rather
            # than one long line, and clip_on=True guarantees it can never
            # render outside the axes box even at the narrowest window size.
            ax.text(A + 0.15, e, f"n={n}\nE={e:.2f}", va="center", ha="left",
                    fontsize=7.8, linespacing=1.3, clip_on=True,
                    color=color if (is_selected or is_compared) else FG_SUBTEXT,
                    fontweight="bold" if (is_selected or is_compared) else "normal")

        # Extra right-hand margin comfortably fits the stacked "n=5 / E=5.50"
        # label at the highest energy level, keeping everything inside the box.
        ax.set_xlim(-max_turning - 0.6, max_turning + 1.5)
        ax.set_ylim(0, max(levels_e) + 1.2)
        ax.set_xlabel("Position (x)", color=FG_TEXT, fontsize=9)
        ax.set_ylabel("Energy  (units of \u0127\u03c9)", color=FG_TEXT)
        ax.set_title("Energy Levels Inside the Potential Well", color=FG_TEXT,
                     fontsize=11.5, fontweight="bold")
        ax.tick_params(colors=FG_SUBTEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.4, alpha=0.3)

    def _draw_wavefunction(self, x, psi, n, n2):
        ax = self.ax_wave
        ax.clear()
        ax.set_facecolor(BG_CARD)

        ax.axhline(0, color=GRID_COLOR, linewidth=1)
        ax.plot(x, psi, color=ACCENT_COLOR, linewidth=2.2, label=f"n = {n}")
        ax.fill_between(x, psi, 0, color=ACCENT_COLOR, alpha=0.15)

        title = f"Wavefunction \u03c8\u2099(x)  for n = {n}"
        if n2 is not None:
            psi2 = self.qho.wavefunction(n2, x)
            ax.plot(x, psi2, color=ACCENT_COLOR3, linewidth=2.0, linestyle="--",
                    label=f"n = {n2} (compare)")
            title = f"Wavefunction \u03c8\u2099(x): n = {n}  vs  n = {n2}"
            ax.legend(loc="upper right", fontsize=8, facecolor=BG_CARD,
                      edgecolor=GRID_COLOR, labelcolor=FG_TEXT)

        ax.set_title(title, color=FG_TEXT, fontsize=11, fontweight="bold")
        ax.set_xlabel("Position (x)", color=FG_TEXT)
        ax.set_ylabel("Wavefunction \u03c8(x)", color=FG_TEXT)
        ax.tick_params(colors=FG_SUBTEXT)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.4)

    def _draw_probability(self, x, prob, n, n2):
        ax = self.ax_prob
        ax.clear()
        ax.set_facecolor(BG_CARD)

        if self.show_probability:
            ax.plot(x, prob, color=ACCENT_COLOR2, linewidth=2.2, label=f"Quantum n={n}")
            ax.fill_between(x, prob, 0, color=ACCENT_COLOR2, alpha=0.25)

            if self.classical_overlay.get():
                classical = self.qho.classical_probability_density(n, x)
                # Cap the display height so the singular edges of the
                # classical curve don't dwarf the quantum curve visually.
                cap = max(np.max(prob) * 3.0, 0.05)
                classical_display = np.minimum(classical, cap)
                ax.plot(x, classical_display, color=ACCENT_COLOR4, linewidth=1.8,
                        linestyle=":", label="Classical (same E)")

            if n2 is not None:
                prob2 = self.qho.probability_density(n2, x)
                ax.plot(x, prob2, color=ACCENT_COLOR3, linewidth=2.0, linestyle="--",
                        label=f"Quantum n={n2}")

            ax.legend(loc="upper right", fontsize=8, facecolor=BG_CARD,
                      edgecolor=GRID_COLOR, labelcolor=FG_TEXT)
            ax.set_title(f"Probability Density |\u03c8\u2099(x)|\u00b2", color=FG_TEXT,
                         fontsize=11, fontweight="bold")
            ax.set_xlabel("Position (x)", color=FG_TEXT)
            ax.set_ylabel("Probability Density", color=FG_TEXT)
            ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.4)
        else:
            ax.text(0.5, 0.5, "Probability Density graph hidden\n"
                               "(click 'Show Probability Density')",
                    transform=ax.transAxes, ha="center", va="center",
                    color=FG_SUBTEXT, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        ax.tick_params(colors=FG_SUBTEXT)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
def main():
    app = QHOApp()
    app.mainloop()


if __name__ == "__main__":
    main()