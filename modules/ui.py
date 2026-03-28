"""
Jenny UI — Modern split-panel dark interface.

Layout:
  Left  (~62%) — Camera tile with avatar, mode pill, detection overlays
  Right (~38%) — Conversation panel with chat bubbles + status footer
  Bottom       — Control bar with circular icon buttons
"""
import textwrap
import tkinter as tk
from tkinter import filedialog
import queue
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageTk
import cv2

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    _DND_OK = True
except ImportError:
    _DND_OK = False

# ── Palette ───────────────────────────────────────────────────────── #
_BG       = (10,  10,  14)
_T_TOP    = (22,  14,   6)
_T_BOT    = (10,   6,   2)
_PANEL_BG = (16,  16,  24)
_PANEL_BD = (40,  40,  58)
_AV_F     = (190,  68,  15)
_AV_RG    = (230, 110,  40)
_GREEN    = ( 38, 210,  98)
_RED      = (240,  65,  65)
_CYAN     = (  0, 200, 245)
_WHITE    = (255, 255, 255)
_DIM      = (115, 115, 132)
_DARK     = ( 12,  12,  16)
_BTN      = ( 38,  38,  46)
_BTN_H    = ( 60,  60,  70)
_BTN_P    = ( 20,  20,  26)
_END_N    = (170,  22,  22)
_END_H    = (210,  32,  32)
_BUBBLE_U = ( 28,  78, 160)
_BUBBLE_J = ( 30,  30,  46)
_TEXT_MN  = (235, 235, 242)
_LABEL_U  = (140, 185, 255)
_LABEL_J  = (  0, 200, 245)

# ── Layout ────────────────────────────────────────────────────────── #
_TILE_R    = 20
_AV_RAD    = 28
_BTN_D     = 54
_BAR_H     = 84
_PAD_X     = 16
_PAD_Y     = 14
_PAD_B     = 12
_PANEL_GAP = 12
_PANEL_RAT = 0.37   # conversation panel fraction of canvas width


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

def _vignette(img, strength=0.40):
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

def _text_w(font, text):
    try:
        bb = font.getbbox(text)
        return bb[2] - bb[0]
    except Exception:
        return len(text) * 8

def _draw_icon(draw, icon, cx, cy, s):
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

    elif icon == "photo":
        bw, bh = int(s*.8), int(s*.6)
        draw.rounded_rectangle([cx-bw, cy-bh, cx+bw, cy+bh], radius=3, outline=c, width=lw)
        pr = int(s*.28)
        draw.ellipse([cx-pr, cy-pr, cx+pr, cy+pr], outline=c, width=lw)
        bump = int(s*.2)
        draw.rectangle([cx-bump, cy-bh-lw, cx+bump, cy-bh+lw], fill=c)


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
    def __init__(self, on_mode_change=None, on_quit=None, on_image_drop=None):
        self.on_mode_change = on_mode_change
        self.on_quit        = on_quit
        self.on_image_drop  = on_image_drop

        self._frame_q    = queue.Queue(maxsize=2)
        self._mode       = "gesture"
        self._listening  = False
        self._gesture    = "No Hand"
        self._dets       = []
        self._convo      = []
        self._running    = True
        self._pulse      = 0
        self._muted      = False
        self._cam_on     = True
        self._cam_img    = None
        self._drop_image = None
        self._analyzing  = False

        # PIL fonts
        self._f_name   = _load_font("segoeuib.ttf", 18)
        self._f_sub    = _load_font("segoeui.ttf",  12)
        self._f_status = _load_font("segoeui.ttf",  12)
        self._f_av     = _load_font("segoeuib.ttf", 32)
        self._f_conv   = _load_font("segoeui.ttf",  14)
        self._f_label  = _load_font("segoeuib.ttf", 11)
        self._f_pill   = _load_font("segoeuib.ttf", 12)
        self._f_header = _load_font("segoeuib.ttf", 13)

        # Mask cache
        self._tile_mask_key = None
        self._tile_mask     = None

        self._build()

    # ── Build ──────────────────────────────────────────────────────── #

    def _build(self):
        self.root = TkinterDnD.Tk() if _DND_OK else tk.Tk()
        self.root.title("Jenny AI")
        self.root.configure(bg=_hex(_BG))
        self.root.geometry("1280x720")
        self.root.minsize(900, 540)
        try:
            self.root.state("zoomed")
        except Exception:
            pass

        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        for key, fn in [("g", "gesture"), ("G", "gesture"),
                         ("v", "vision"),  ("V", "vision")]:
            self.root.bind(f"<Key-{key}>", lambda e, m=fn: self._set_mode(m))
        self.root.bind("<Key-q>", lambda e: self._quit())
        self.root.bind("<Key-Q>", lambda e: self._quit())

        self._cv = tk.Canvas(self.root, bg=_hex(_BG), highlightthickness=0)
        self._cv.pack(fill="both", expand=True)

        if _DND_OK:
            self._cv.drop_target_register(DND_FILES)
            self._cv.dnd_bind("<<Drop>>", self._on_file_drop)

        self._build_bar()
        self._tick()

    def _build_bar(self):
        bar = tk.Frame(self.root, bg=_hex(_DARK), height=_BAR_H)
        bar.place(relx=0.5, rely=1.0, anchor="s", relwidth=1.0)
        bar.pack_propagate(False)

        center = tk.Frame(bar, bg=_hex(_DARK))
        center.place(relx=0.5, rely=0.5, anchor="center")

        self._b_mic   = _Btn(center, "mic",     on_click=self._toggle_mic)
        self._b_cam   = _Btn(center, "cam",     on_click=self._toggle_cam)
        self._b_ges   = _Btn(center, "gesture", on_click=lambda: self._set_mode("gesture"))
        self._b_vis   = _Btn(center, "vision",  on_click=lambda: self._set_mode("vision"))
        self._b_photo = _Btn(center, "photo",   on_click=self._open_file_dialog)
        self._b_more  = _Btn(center, "more")
        self._b_end   = _Btn(center, "end", bg_n=_END_N, bg_h=_END_H, on_click=self._quit)

        gap1 = tk.Frame(center, bg=_hex(_DARK), width=22)
        gap2 = tk.Frame(center, bg=_hex(_DARK), width=22)
        gap3 = tk.Frame(center, bg=_hex(_DARK), width=34)

        for w in [self._b_mic, self._b_cam, gap1,
                  self._b_ges, self._b_vis, self._b_photo, gap2,
                  self._b_more, gap3, self._b_end]:
            if isinstance(w, _Btn):
                w.pack(side="left", padx=5)
            else:
                w.pack(side="left")

        self._sync_btns()

    # ── Render loop ────────────────────────────────────────────────── #

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

        # ── Layout ───────────────────────────────────────────────── #
        content_h = H - _BAR_H - _PAD_B
        panel_w   = int(W * _PANEL_RAT) - _PAD_X
        tile_w    = W - _PAD_X * 2 - panel_w - _PANEL_GAP

        tx0, ty0 = _PAD_X, _PAD_Y
        tx1, ty1 = tx0 + tile_w, _PAD_Y + content_h

        px0, py0 = tx1 + _PANEL_GAP, _PAD_Y
        px1, py1 = W - _PAD_X, _PAD_Y + content_h

        tw, th = tx1 - tx0, ty1 - ty0
        if tw < 10 or th < 10 or (px1 - px0) < 40:
            return scene

        # ── Camera tile ───────────────────────────────────────────── #
        mkey = (tw, th)
        if mkey != self._tile_mask_key:
            self._tile_mask     = _round_mask(tw, th, _TILE_R)
            self._tile_mask_key = mkey
        tile_mask = self._tile_mask

        tile_base = _gradient_v(tw, th, _T_TOP, _T_BOT)
        scene.paste(tile_base, (tx0, ty0), tile_mask)

        if self._drop_image is not None:
            drop = self._drop_image.resize((tw, th), Image.LANCZOS)
            scene.paste(drop, (tx0, ty0), tile_mask)
            cam_active = True
        else:
            cam_active = self._cam_img is not None and self._cam_on
            if cam_active:
                cam = self._cam_img.resize((tw, th), Image.LANCZOS)
                cam = _vignette(cam, 0.40)
                scene.paste(cam, (tx0, ty0), tile_mask)

        # ── Tile overlays (single RGBA layer) ────────────────────── #
        ov    = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        odraw = ImageDraw.Draw(ov)

        # Avatar — top-left corner of tile
        av_x  = tx0 + 22 + _AV_RAD
        av_y  = ty0 + 22 + _AV_RAD
        av_a  = 130 if cam_active else 255
        odraw.ellipse([av_x-_AV_RAD, av_y-_AV_RAD, av_x+_AV_RAD, av_y+_AV_RAD],
                       fill=_AV_F + (av_a,),
                       outline=_AV_RG + (min(255, av_a+60),), width=2)
        try:
            odraw.text((av_x, av_y), "J", font=self._f_av,
                        fill=(255, 255, 255, av_a), anchor="mm")
        except Exception:
            pass

        # Mode pill — top-center of tile
        mode_lbl  = "GESTURE" if self._mode == "gesture" else "VISION"
        pill_tw   = _text_w(self._f_pill, mode_lbl)
        pill_ph   = 14
        pill_padh, pill_padv = 14, 6
        pill_w    = pill_tw + pill_padh * 2
        pill_h    = pill_ph + pill_padv * 2
        pill_cx   = (tx0 + tx1) // 2
        pill_cy   = ty0 + 22
        odraw.rounded_rectangle([pill_cx - pill_w//2, pill_cy - pill_h//2,
                                   pill_cx + pill_w//2, pill_cy + pill_h//2],
                                  radius=pill_h // 2,
                                  fill=(0, 0, 0, 150))
        odraw.rounded_rectangle([pill_cx - pill_w//2, pill_cy - pill_h//2,
                                   pill_cx + pill_w//2, pill_cy + pill_h//2],
                                  radius=pill_h // 2,
                                  outline=_CYAN + (200,), width=1)
        try:
            odraw.text((pill_cx, pill_cy), mode_lbl,
                        font=self._f_pill, fill=_CYAN + (230,), anchor="mm")
        except Exception:
            pass

        # Detection / gesture badge — below pill on the left
        if self._mode == "gesture":
            badge   = self._gesture if self._gesture not in ("No Hand", "Unknown", "", None) else None
            badge_c = _CYAN + (220,)
        else:
            badge   = self._dets[0] if self._dets else None
            badge_c = (80, 220, 130, 220)

        if badge:
            try:
                odraw.text((tx0 + 22, ty0 + 60), badge,
                            font=self._f_status, fill=badge_c, anchor="lm")
            except Exception:
                pass

        # Status dot — top-right corner of tile (pulsing)
        dot_on = self._pulse < 12
        dot_c  = _RED if self._listening else _GREEN
        dot_r  = 9 if (self._listening and dot_on) else 7
        dx, dy = tx1 - 20, ty0 + 24
        odraw.ellipse([dx-dot_r, dy-dot_r, dx+dot_r, dy+dot_r], fill=dot_c + (255,))

        # Name label — bottom-left of tile
        nl_x, nl_y = tx0 + 20, ty1 - 38
        odraw.text((nl_x, nl_y),      "Jenny",
                    font=self._f_name, fill=_WHITE + (215,), anchor="lm")
        odraw.text((nl_x, nl_y + 22), "AI Assistant",
                    font=self._f_sub,  fill=_DIM   + (180,), anchor="lm")

        # Analyzing overlay
        if self._analyzing:
            pulse_a = int(180 + 60 * abs((self._pulse % 24) / 12.0 - 1.0))
            lbl = "Analyzing image..."
            lbl_w = _text_w(self._f_name, lbl)
            odraw.text(((tx0+tx1)//2 - lbl_w//2, (ty0+ty1)//2 + 60),
                        lbl, font=self._f_name, fill=_CYAN + (pulse_a,))

        # Drop zone (no camera, no image)
        elif self._drop_image is None and not cam_active:
            dz_pad = 40
            dz = [tx0+dz_pad, ty0+dz_pad, tx1-dz_pad, ty1-dz_pad-30]
            dim_a = _DIM + (140,)
            for i in range(0, int(dz[2]-dz[0]), 14):
                x = dz[0]+i
                odraw.line([x, dz[1], min(x+7, dz[2]), dz[1]], fill=dim_a, width=1)
                odraw.line([x, dz[3], min(x+7, dz[2]), dz[3]], fill=dim_a, width=1)
            for i in range(0, int(dz[3]-dz[1]), 14):
                y = dz[1]+i
                odraw.line([dz[0], y, dz[0], min(y+7, dz[3])], fill=dim_a, width=1)
                odraw.line([dz[2], y, dz[2], min(y+7, dz[3])], fill=dim_a, width=1)
            try:
                odraw.text(((tx0+tx1)//2, (ty0+ty1)//2 - 10),
                             "Drop an image here",
                             font=self._f_name, fill=_DIM+(150,), anchor="mm")
                odraw.text(((tx0+tx1)//2, (ty0+ty1)//2 + 18),
                             "or click the photo button below",
                             font=self._f_sub,  fill=_DIM+(120,), anchor="mm")
            except Exception:
                pass

        scene = Image.alpha_composite(scene.convert("RGBA"), ov).convert("RGB")

        # ── Conversation panel ────────────────────────────────────── #
        scene = self._draw_panel(scene, px0, py0, px1, py1)

        return scene

    def _draw_panel(self, scene: Image.Image, px0, py0, px1, py1) -> Image.Image:
        pw = px1 - px0

        ov    = Image.new("RGBA", scene.size, (0, 0, 0, 0))
        odraw = ImageDraw.Draw(ov)

        # Card background + border
        odraw.rounded_rectangle([px0, py0, px1, py1], radius=_TILE_R,
                                  fill=_PANEL_BG + (255,),
                                  outline=_PANEL_BD + (255,), width=1)

        # Header
        hdr_y = py0 + 22
        odraw.text((px0 + 16, hdr_y), "Conversation",
                    font=self._f_header, fill=_DIM + (255,), anchor="lm")
        sep_y = hdr_y + 14
        odraw.line([px0 + 10, sep_y, px1 - 10, sep_y],
                    fill=_PANEL_BD + (255,), width=1)

        # Footer: listening status
        foot_y  = py1 - 18
        pulse_on = self._pulse < 12
        if self._listening:
            dot_r = 5 if pulse_on else 4
            odraw.ellipse([px0+14-dot_r, foot_y-dot_r,
                            px0+14+dot_r, foot_y+dot_r], fill=_GREEN + (255,))
            odraw.text((px0+26, foot_y), "Listening...",
                        font=self._f_status, fill=_GREEN + (230,), anchor="lm")
        else:
            odraw.ellipse([px0+11, foot_y-4, px0+19, foot_y+4],
                           fill=_DIM + (180,))
            odraw.text((px0+26, foot_y), "Ready",
                        font=self._f_status, fill=_DIM + (200,), anchor="lm")
        sep2_y = foot_y - 14
        odraw.line([px0 + 10, sep2_y, px1 - 10, sep2_y],
                    fill=_PANEL_BD + (255,), width=1)

        # Message area bounds
        msg_top = sep_y + 10
        msg_bot = sep2_y - 10

        # Measure wrap width
        inner_w = pw - 28
        try:
            char_w = max(1, self._f_conv.getbbox("m")[2] - self._f_conv.getbbox("m")[0])
        except Exception:
            char_w = 8
        wrap_chars = max(15, inner_w // char_w)

        # Measure all bubbles
        LINE_H    = 19
        BUB_PAD_H = 12
        BUB_PAD_V = 8
        LABEL_H   = 14
        BUB_GAP   = 8
        BX0       = px0 + 10
        BX1       = px1 - 10

        measured = []  # (role, lines, bub_h)
        for role, text in reversed(self._convo[-10:]):
            lines = textwrap.wrap(text, wrap_chars) or [text[:wrap_chars]]
            bub_h = len(lines) * LINE_H + BUB_PAD_V * 2
            measured.append((role, lines, bub_h))

        # Draw bubble backgrounds (into overlay)
        bubble_pos = []   # (role, lines, bub_y0) — for text pass
        y = msg_bot
        for role, lines, bub_h in measured:
            total = LABEL_H + 3 + bub_h + BUB_GAP
            y -= total
            if y < msg_top:
                break
            bub_color = _BUBBLE_U if role == "you" else _BUBBLE_J
            label_c   = _LABEL_U  if role == "you" else _LABEL_J
            label     = "You"     if role == "you" else "Jenny"
            bub_y0    = y + LABEL_H + 3
            bub_y1    = bub_y0 + bub_h

            odraw.text((BX0 + 2, y + LABEL_H // 2), label,
                        font=self._f_label, fill=label_c + (210,), anchor="lm")
            odraw.rounded_rectangle([BX0, bub_y0, BX1, bub_y1],
                                      radius=10, fill=bub_color + (230,))
            bubble_pos.append((role, lines, bub_y0))

        # Composite overlay (card bg + bubbles, no text yet)
        scene = Image.alpha_composite(scene.convert("RGBA"), ov).convert("RGB")
        draw  = ImageDraw.Draw(scene)

        # Empty state
        if not self._convo and msg_bot - msg_top >= 20:
            draw.text(((px0+px1)//2, (msg_top+msg_bot)//2),
                        "Say something...", font=self._f_sub, fill=_DIM, anchor="mm")
            return scene

        # Draw bubble text (fully opaque, on top of bubble bg)
        for role, lines, bub_y0 in bubble_pos:
            for li, line in enumerate(lines):
                ty = bub_y0 + BUB_PAD_V + li * LINE_H + LINE_H // 2
                draw.text((BX0 + BUB_PAD_H, ty), line,
                            font=self._f_conv, fill=_TEXT_MN, anchor="lm")

        return scene

    # ── Public API ─────────────────────────────────────────────────── #

    def push_frame(self, cv2_frame):
        try:
            rgb = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
            self._frame_q.put_nowait(Image.fromarray(rgb))
        except Exception:
            pass

    def set_listening(self, v):    self._listening = v
    def set_mode(self, m):         self._mode = m
    def set_gesture(self, g):      self._gesture = g
    def set_detections(self, d):   self._dets = d
    def set_analyzing(self, v):    self._analyzing = v
    def set_drop_image(self, img): self._drop_image = img

    def add_turn(self, role, text):
        self._convo.append((role, text))
        if len(self._convo) > 20:
            self._convo = self._convo[-20:]

    def is_running(self): return self._running
    def run(self):        self.root.mainloop()

    # ── Internal ───────────────────────────────────────────────────── #

    def _set_mode(self, mode):
        self._mode = mode
        self._sync_btns()
        if self.on_mode_change:
            self.on_mode_change(mode)

    def _sync_btns(self):
        self._b_ges.set_active(self._mode == "gesture")
        self._b_vis.set_active(self._mode == "vision")

    def _toggle_mic(self):
        self._muted      = not self._muted
        self._b_mic.icon = "mic_off" if self._muted else "mic"
        self._b_mic._prerender()
        self._b_mic._show()

    def _toggle_cam(self):
        self._cam_on     = not self._cam_on
        self._b_cam.icon = "cam_off" if not self._cam_on else "cam"
        self._b_cam._prerender()
        self._b_cam._show()

    def _on_file_drop(self, event):
        path = event.data.strip().strip("{}")
        if self.on_image_drop:
            self.on_image_drop(path)

    def _open_file_dialog(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
                       ("All files", "*.*")]
        )
        if path and self.on_image_drop:
            self.on_image_drop(path)

    def _quit(self):
        self._running = False
        if self.on_quit:
            self.on_quit()
        try:
            self.root.destroy()
        except Exception:
            pass
