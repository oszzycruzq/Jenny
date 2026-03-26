"""
Jenny UI — Video-call-style dark interface.

3 Layers:
  1. Background  — near-black full-screen canvas
  2. Video Tile  — large centered rounded card with camera + overlays
  3. Control Bar — floating bottom row of circular icon buttons
"""
import tkinter as tk
import queue
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageTk
import cv2

# ── Palette (RGB tuples) ───────────────────────────────────────────── #
_BG    = (8,   8,  10)
_T_TOP = (26,  15,   7)    # tile top gradient
_T_BOT = (14,   8,   3)    # tile bottom gradient
_AV_F  = (195,  72,  16)   # avatar fill
_AV_RG = (235, 115,  45)   # avatar ring
_GREEN = ( 34, 197,  94)
_RED   = (239,  68,  68)
_CYAN  = (  0, 198, 242)
_WHITE = (255, 255, 255)
_DIM   = (140, 140, 145)
_DARK  = ( 12,  12,  15)   # bar bg
_BTN   = ( 36,  36,  42)
_BTN_H = ( 58,  58,  66)
_BTN_P = ( 20,  20,  25)
_END_N = (175,  25,  25)
_END_H = (215,  35,  35)

_TILE_R = 22
_AV_RAD = 46
_BTN_D  = 52
_BAR_H  = 80
_PAD_X  = 28
_PAD_Y  = 20
_PAD_B  = 14   # gap between tile bottom edge and bar top


# ── Helpers ───────────────────────────────────────────────────────── #

def _hex(c): return "#{:02x}{:02x}{:02x}".format(*c[:3])

def _gradient_v(w, h, top, bot):
    a = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(3):
        a[:, :, i] = np.linspace(top[i], bot[i], h, dtype=np.uint8)[:, None]
    return Image.fromarray(a, "RGB")

def _round_mask(w, h, r):
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w-1, h-1], radius=r, fill=255)
    return m

def _vignette(img, strength=0.5):
    w, h = img.size
    v = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(v)
    cx, cy = w // 2, h // 2
    for i in range(50):
        t = i / 50
        val = int(255 * (1 - strength * t * t))
        d.ellipse([cx - int(cx*(1-t)), cy - int(cy*(1-t)),
                   cx + int(cx*(1-t)), cy + int(cy*(1-t))], fill=val)
    v = v.filter(ImageFilter.GaussianBlur(min(w, h) // 12))
    dark = Image.new("RGB", (w, h), (0, 0, 0))
    return Image.composite(dark, img, v)

def _load_font(filename, size):
    for path in [f"C:/Windows/Fonts/{filename}",
                 "C:/Windows/Fonts/segoeui.ttf",
                 "C:/Windows/Fonts/arial.ttf"]:
        try: return ImageFont.truetype(path, size)
        except: pass
    return ImageFont.load_default()

def _draw_icon(draw, icon, cx, cy, s):
    """Draw a white icon. s = half-size of icon box."""
    c  = (255, 255, 255, 255)
    lw = max(2, s // 4)

    if icon == "mic":
        r = s // 3
        draw.rounded_rectangle([cx-r, cy-s, cx+r, cy+r//4],
                                radius=r, outline=c, width=lw)
        draw.line([cx, cy+r//4, cx, cy+s//2], fill=c, width=lw)
        draw.arc([cx-r, cy-r//3, cx+r, cy+s//2], 0, 180, fill=c, width=lw)

    elif icon == "mic_off":
        r = s // 3
        dim = (110, 110, 115, 255)
        draw.rounded_rectangle([cx-r, cy-s, cx+r, cy+r//4],
                                radius=r, outline=dim, width=lw)
        draw.line([cx-s, cy+s//3, cx+s, cy-s//3],
                  fill=(200, 50, 50, 255), width=lw+1)

    elif icon == "cam":
        bw, bh = int(s*.85), s//2
        draw.rounded_rectangle([cx-bw, cy-bh, cx+bw, cy+bh],
                                radius=4, outline=c, width=lw)
        pr = bh // 2
        draw.ellipse([cx-pr, cy-pr, cx+pr, cy+pr], outline=c, width=lw)

    elif icon == "cam_off":
        bw, bh = int(s*.85), s//2
        dim = (110, 110, 115, 255)
        draw.rounded_rectangle([cx-bw, cy-bh, cx+bw, cy+bh],
                                radius=4, outline=dim, width=lw)
        draw.line([cx-s, cy+bh, cx+s, cy-bh],
                  fill=(200, 50, 50, 255), width=lw+1)

    elif icon == "gesture":
        fw = max(3, s // 4)
        for dx, fh in [(-int(s*.55), int(s*.70)),
                        (-int(s*.18), int(s*.90)),
                        ( int(s*.18), int(s*.90)),
                        ( int(s*.55), int(s*.65))]:
            draw.rounded_rectangle([cx+dx-fw//2, cy-fh,
                                     cx+dx+fw//2, cy+s//4],
                                    radius=fw//2, outline=c, width=lw)

    elif icon == "vision":
        ew, eh = s, s // 2
        draw.ellipse([cx-ew, cy-eh, cx+ew, cy+eh], outline=c, width=lw)
        pr = max(3, s // 3)
        draw.ellipse([cx-pr, cy-pr, cx+pr, cy+pr], outline=c, width=lw)

    elif icon == "end":
        d = int(s * .65)
        draw.line([cx-d, cy-d, cx+d, cy+d], fill=c, width=lw+1)
        draw.line([cx+d, cy-d, cx-d, cy+d], fill=c, width=lw+1)

    elif icon == "more":
        r = max(2, s // 5)
        for dx in [int(-s*.6), 0, int(s*.6)]:
            draw.ellipse([cx+dx-r, cy-r, cx+dx+r, cy+r], fill=c)


# ── Circular button ───────────────────────────────────────────────── #

class _Btn:
    def __init__(self, parent, icon, bg_n=_BTN, bg_h=_BTN_H,
                 size=_BTN_D, on_click=None):
        self.icon = icon
        self.bg_n, self.bg_h = bg_n, bg_h
        self.size = size
        self.on_click = on_click
        self._hover = self._pressed = self._active = False
        self._act_color = _CYAN

        self._cv = tk.Canvas(parent, width=size, height=size,
                              bg=_hex(_DARK), highlightthickness=0,
                              cursor="hand2")
        self._photos = {}
        self._prerender()
        self._show()

        self._cv.bind("<Enter>",           lambda e: self._st(hover=True))
        self._cv.bind("<Leave>",           lambda e: self._st(hover=False))
        self._cv.bind("<ButtonPress-1>",   lambda e: self._st(pressed=True))
        self._cv.bind("<ButtonRelease-1>", self._released)

    def pack(self, **kw): self._cv.pack(**kw)

    def set_active(self, val, color=None):
        self._active = val
        if color: self._act_color = color
        self._prerender()
        self._show()

    def _prerender(self):
        d = self.size
        for k, bg in [("n", self.bg_n), ("h", self.bg_h),
                       ("p", _BTN_P),   ("a", self._act_color)]:
            img  = Image.new("RGBA", (d, d), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.ellipse([2, 2, d-2, d-2], fill=bg + (255,))
            _draw_icon(draw, self.icon, d//2, d//2, d//5)
            out = Image.new("RGB", (d, d), _DARK)
            out.paste(img, mask=img.split()[3])
            self._photos[k] = ImageTk.PhotoImage(out)

    def _st(self, hover=None, pressed=None):
        if hover   is not None: self._hover   = hover
        if pressed is not None: self._pressed = pressed
        self._show()

    def _show(self):
        k  = "p" if self._pressed else "a" if self._active else "h" if self._hover else "n"
        ph = self._photos[k]
        self._cv.delete("all")
        self._cv.create_image(0, 0, anchor="nw", image=ph)
        self._cv._ph = ph

    def _released(self, e):
        self._pressed = False
        self._show()
        if self.on_click: self.on_click()


# ── Main UI ───────────────────────────────────────────────────────── #

class JennyUI:
    def __init__(self, on_mode_change=None, on_quit=None):
        self.on_mode_change = on_mode_change
        self.on_quit        = on_quit

        self._frame_q   = queue.Queue(maxsize=2)
        self._mode      = "gesture"
        self._listening = False
        self._gesture   = "No Hand"
        self._dets      = []
        self._convo     = []
        self._running   = True
        self._pulse     = 0
        self._muted     = False
        self._cam_on    = True
        self._cam_img   = None

        # PIL fonts
        self._f_name   = _load_font("segoeuib.ttf", 20)
        self._f_sub    = _load_font("segoeui.ttf",  12)
        self._f_status = _load_font("segoeui.ttf",  11)
        self._f_av     = _load_font("segoeuib.ttf", 44)
        self._f_conv   = _load_font("segoeui.ttf",  12)

        # Cached masks (invalidated on resize)
        self._mask_key  = None
        self._mask_cache = None

        self._build()

    # ── Build ─────────────────────────────────────────────────────── #

    def _build(self):
        self.root = tk.Tk()
        self.root.title("Jenny AI")
        self.root.configure(bg=_hex(_BG))
        self.root.geometry("1280x720")
        self.root.minsize(800, 500)
        try:
            self.root.state("zoomed")
        except Exception:
            pass

        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        for key, fn in [("g", "gesture"), ("G", "gesture"),
                         ("v", "vision"),  ("V", "vision")]:
            self.root.bind(f"<Key-{key}>",
                           lambda e, m=fn: self._set_mode(m))
        self.root.bind("<Key-q>", lambda e: self._quit())
        self.root.bind("<Key-Q>", lambda e: self._quit())

        # Layer 1+2: main canvas
        self._cv = tk.Canvas(self.root, bg=_hex(_BG), highlightthickness=0)
        self._cv.pack(fill="both", expand=True)

        # Layer 3: control bar
        self._build_bar()
        self._tick()

    def _build_bar(self):
        bar = tk.Frame(self.root, bg=_hex(_DARK), height=_BAR_H)
        bar.place(relx=0.5, rely=1.0, anchor="s", relwidth=1.0)
        bar.pack_propagate(False)

        center = tk.Frame(bar, bg=_hex(_DARK))
        center.place(relx=0.5, rely=0.5, anchor="center")

        self._b_mic  = _Btn(center, "mic",     on_click=self._toggle_mic)
        self._b_cam  = _Btn(center, "cam",     on_click=self._toggle_cam)
        self._b_ges  = _Btn(center, "gesture", on_click=lambda: self._set_mode("gesture"))
        self._b_vis  = _Btn(center, "vision",  on_click=lambda: self._set_mode("vision"))
        self._b_more = _Btn(center, "more")
        self._b_end  = _Btn(center, "end",
                             bg_n=_END_N, bg_h=_END_H,
                             on_click=self._quit)

        gap1 = tk.Frame(center, bg=_hex(_DARK), width=22)
        gap2 = tk.Frame(center, bg=_hex(_DARK), width=22)
        gap3 = tk.Frame(center, bg=_hex(_DARK), width=34)

        for w in [self._b_mic, self._b_cam, gap1,
                  self._b_ges, self._b_vis, gap2,
                  self._b_more, gap3,
                  self._b_end]:
            if isinstance(w, _Btn):
                w.pack(side="left", padx=5)
            else:
                w.pack(side="left")

        self._sync_btns()

    # ── Render loop ───────────────────────────────────────────────── #

    def _tick(self):
        if not self._running:
            return
        self._pulse = (self._pulse + 1) % 24

        try:
            self._cam_img = self._frame_q.get_nowait()
        except queue.Empty:
            pass

        W = self._cv.winfo_width()
        H = self._cv.winfo_height()
        if W > 4 and H > 4:
            scene = self._compose(W, H)
            ph = ImageTk.PhotoImage(scene)
            self._cv.create_image(0, 0, anchor="nw", image=ph)
            self._cv._ph = ph

        self.root.after(33, self._tick)

    def _compose(self, W, H) -> Image.Image:
        scene = Image.new("RGB", (W, H), _BG)

        # Tile bounds
        tx0, ty0 = _PAD_X, _PAD_Y
        tx1, ty1 = W - _PAD_X, H - _BAR_H - _PAD_B
        tw, th   = tx1 - tx0, ty1 - ty0
        if tw < 10 or th < 10:
            return scene

        # Rounded mask (cached)
        mkey = (tw, th)
        if mkey != self._mask_key:
            self._mask_cache = _round_mask(tw, th, _TILE_R)
            self._mask_key   = mkey
        mask = self._mask_cache

        # Tile gradient base
        tile_base = _gradient_v(tw, th, _T_TOP, _T_BOT)
        scene.paste(tile_base, (tx0, ty0), mask)

        # Camera feed
        cam_active = self._cam_img is not None and self._cam_on
        if cam_active:
            cam = self._cam_img.resize((tw, th), Image.LANCZOS)
            cam = _vignette(cam, 0.48)
            scene.paste(cam, (tx0, ty0), mask)

        # Avatar overlay (always shown; semi-transparent when camera is on)
        cx = (tx0 + tx1) // 2
        cy = (ty0 + ty1) // 2
        r  = _AV_RAD
        av_alpha = 160 if cam_active else 255

        ov   = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        odraw = ImageDraw.Draw(ov)
        odraw.ellipse([cx-r, cy-r, cx+r, cy+r],
                       fill=_AV_F + (av_alpha,),
                       outline=_AV_RG + (min(255, av_alpha+40),), width=3)
        try:
            odraw.text((cx, cy), "J", font=self._f_av,
                        fill=(255, 255, 255, av_alpha), anchor="mm")
        except Exception:
            odraw.text((cx - 14, cy - 22), "J",
                        fill=(255, 255, 255, av_alpha))

        scene = Image.alpha_composite(scene.convert("RGBA"), ov).convert("RGB")
        draw  = ImageDraw.Draw(scene)

        # Status dot — top-right of tile
        dx, dy = tx1 - 20, ty0 + 24
        on  = self._pulse < 12
        sc  = _RED if self._listening else _GREEN
        dr  = 9 if (self._listening and on) else 7
        draw.ellipse([dx-dr, dy-dr, dx+dr, dy+dr], fill=sc)
        st  = "Listening" if self._listening else "Ready"
        try:
            draw.text((dx - dr - 8, dy), st,
                       font=self._f_status, fill=sc, anchor="rm")
        except Exception:
            pass

        # Detection / gesture badge — top-left of tile
        if self._mode == "gesture":
            g = self._gesture
            badge = g if g and g not in ("No Hand", "Unknown", "") else None
            badge_color = _CYAN
        else:
            badge = self._dets[0] if self._dets else None
            badge_color = (80, 220, 130)

        if badge:
            try:
                draw.text((tx0 + 20, ty0 + 24), badge,
                           font=self._f_status, fill=badge_color, anchor="lm")
            except Exception:
                pass

        # Conversation overlay — above name label
        if self._convo:
            recent = self._convo[-2:]
            base_y = ty1 - 90
            for i, (role, text) in enumerate(recent):
                lc = (255, 200, 80)  if role == "you" else _CYAN
                lbl = "You:"   if role == "you" else "Jenny:"
                y   = base_y + i * 26
                try:
                    draw.text((tx0 + 20, y), lbl,
                               font=self._f_conv, fill=lc, anchor="lm")
                    draw.text((tx0 + 72, y), text[:72],
                               font=self._f_conv, fill=(230, 230, 235), anchor="lm")
                except Exception:
                    pass

        # Name label — bottom-left of tile
        nx, ny = tx0 + 20, ty1 - 34
        try:
            draw.text((nx, ny), "Jenny",
                       font=self._f_name, fill=_WHITE, anchor="lm")
            draw.text((nx, ny + 20), "Personal Assistant",
                       font=self._f_sub, fill=_DIM, anchor="lm")
        except Exception:
            draw.text((nx, ny), "Jenny AI", fill=_WHITE)

        return scene

    # ── Public API ────────────────────────────────────────────────── #

    def push_frame(self, cv2_frame):
        try:
            rgb = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
            self._frame_q.put_nowait(Image.fromarray(rgb))
        except Exception:
            pass

    def set_listening(self, v): self._listening = v
    def set_mode(self, m):      self._mode = m
    def set_gesture(self, g):   self._gesture = g
    def set_detections(self, d): self._dets = d

    def add_turn(self, role, text):
        self._convo.append((role, text))
        if len(self._convo) > 20:
            self._convo = self._convo[-20:]

    def is_running(self): return self._running
    def run(self):        self.root.mainloop()

    # ── Internal ──────────────────────────────────────────────────── #

    def _set_mode(self, mode):
        self._mode = mode
        self._sync_btns()
        if self.on_mode_change:
            self.on_mode_change(mode)

    def _sync_btns(self):
        self._b_ges.set_active(self._mode == "gesture")
        self._b_vis.set_active(self._mode == "vision")

    def _toggle_mic(self):
        self._muted    = not self._muted
        self._b_mic.icon = "mic_off" if self._muted else "mic"
        self._b_mic._prerender()
        self._b_mic._show()

    def _toggle_cam(self):
        self._cam_on   = not self._cam_on
        self._b_cam.icon = "cam_off" if not self._cam_on else "cam"
        self._b_cam._prerender()
        self._b_cam._show()

    def _quit(self):
        self._running = False
        if self.on_quit:
            self.on_quit()
        try:
            self.root.destroy()
        except Exception:
            pass
