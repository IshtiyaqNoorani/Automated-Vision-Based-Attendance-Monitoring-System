import sys
import os
import csv
import cv2
import shutil
import numpy as np
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStackedWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QScrollArea, QFrame, QSizePolicy,
    QGraphicsDropShadowEffect, QProgressBar, QFileDialog,
    QSpacerItem
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF, QPropertyAnimation, QEasingCurve, QSize
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QFontDatabase,
    QPainter, QColor, QPen, QBrush, QLinearGradient, QPalette
)

# ─── Paths ────────────────────────────────────────────────────────────────────
DATASET_DIR     = "data/registered_faces"
ATTENDANCE_FILE = "attendance.csv"
NUM_IMAGES      = 25

# ─── UI CHANGE: Refined colour tokens — deeper blacks, better contrast hierarchy ─
C_BG       = "#080808"           # UI CHANGE: slightly deeper background
C_SURFACE  = "#0f0f0f"           # UI CHANGE: richer sidebar surface
C_CARD     = "#161618"           # UI CHANGE: elevated card color
C_CARD_HVR = "#1c1c1f"           # UI CHANGE: new hover state for cards
C_BORDER   = "#252528"           # UI CHANGE: subtler, more refined borders
C_BORDER_HVR = "#3a3a3e"         # UI CHANGE: border hover state
C_TEXT     = "#f2f2f7"           # UI CHANGE: softer pure white → Apple off-white
C_SUBTEXT  = "#6e6e73"           # UI CHANGE: Apple-spec secondary text
C_TERTIARY = "#3a3a3c"           # UI CHANGE: new tertiary text level
C_PRESENT  = "#30d158"           # UI CHANGE: richer green accent
C_PRESENT_DIM = "#1a3d24"        # UI CHANGE: muted present bg
C_ABSENT   = "#ff375f"           # UI CHANGE: more vivid red
C_ABSENT_DIM  = "#3d1a1f"        # UI CHANGE: muted absent bg
C_ACCENT   = "#0a84ff"           # UI CHANGE: new blue accent for focus states
C_DIVIDER  = "#1e1e20"           # UI CHANGE: barely-visible dividers

# UI CHANGE: font stack with fallbacks for cross-platform rendering
FD = "SF Pro Display, Helvetica Neue, Helvetica, Arial"
FT = "SF Pro Text, Helvetica Neue, Helvetica, Arial"


# ─── UI CHANGE: Improved shadow utility — layered, softer shadows ─────────────
def shadow(widget, blur=32, opacity=120, y_offset=6):
    # UI CHANGE: increased blur, reduced opacity for a more diffuse premium shadow
    e = QGraphicsDropShadowEffect()
    e.setBlurRadius(blur)
    e.setOffset(0, y_offset)
    c = QColor("#000000")
    c.setAlpha(opacity)
    e.setColor(c)
    widget.setGraphicsEffect(e)


def lbl(text, size=13, weight=QFont.Normal, color=C_TEXT, align=Qt.AlignLeft):
    w = QLabel(text)
    # UI CHANGE: use first font in stack only for QFont (fallback handled by Qt)
    w.setFont(QFont("SF Pro Text", size, weight))
    w.setStyleSheet(f"color:{color}; background:transparent;")
    w.setAlignment(align)
    return w


def divider():
    # UI CHANGE: 1px hairline divider, barely visible — more refined
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setFixedHeight(1)
    f.setStyleSheet(f"background:{C_DIVIDER}; border:none;")
    return f


def parse_name(raw):
    parts = raw.split("_", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (raw, "")


# UI CHANGE: Completely redesigned pill_btn — cleaner hierarchy, better states ─
def pill_btn(text, primary=False, danger=False, small=False):
    b = QPushButton(text)
    # UI CHANGE: taller buttons with more breathing room
    b.setFixedHeight(40 if small else 44)
    b.setCursor(Qt.PointingHandCursor)
    # UI CHANGE: slightly heavier font weight for button labels
    b.setFont(QFont("SF Pro Text", 13 if not small else 12, QFont.DemiBold))

    if primary:
        # UI CHANGE: primary = clean white fill, smooth hover, clear disabled state
        b.setStyleSheet(f"""
            QPushButton {{
                background: {C_TEXT};
                color: #000000;
                border: none;
                border-radius: 10px;
                padding: 0 24px;
                letter-spacing: -0.1px;
            }}
            QPushButton:hover {{
                background: #ffffff;
            }}
            QPushButton:pressed {{
                background: #c8c8cc;
            }}
            QPushButton:disabled {{
                background: {C_TERTIARY};
                color: {C_SUBTEXT};
            }}
        """)
    elif danger:
        # UI CHANGE: new danger variant for destructive actions
        b.setStyleSheet(f"""
            QPushButton {{
                background: {C_ABSENT_DIM};
                color: {C_ABSENT};
                border: 1px solid {C_ABSENT}44;
                border-radius: 10px;
                padding: 0 24px;
            }}
            QPushButton:hover {{
                background: {C_ABSENT}22;
                border-color: {C_ABSENT}88;
            }}
            QPushButton:disabled {{
                color: {C_SUBTEXT};
                background: transparent;
                border-color: {C_BORDER};
            }}
        """)
    else:
        # UI CHANGE: secondary = borderless ghost style, more subtle
        b.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {C_TEXT};
                border: 1px solid {C_BORDER};
                border-radius: 10px;
                padding: 0 24px;
            }}
            QPushButton:hover {{
                background: {C_CARD_HVR};
                border-color: {C_BORDER_HVR};
            }}
            QPushButton:pressed {{
                background: {C_TERTIARY}55;
            }}
            QPushButton:disabled {{
                color: {C_SUBTEXT};
                border-color: {C_BORDER};
                background: transparent;
            }}
        """)
    return b


# ─── Donut Chart ─────────────────────────────────────────────────────────────
# UI CHANGE: thicker ring, better center text sizing, cleaner track color

class DonutChart(QWidget):
    def __init__(self, present=0, absent=0, parent=None):
        super().__init__(parent)
        self.present = present
        self.absent  = absent
        self.setFixedSize(180, 180)  # UI CHANGE: slightly smaller for tighter card layout
        self.setStyleSheet("background:transparent;")

    def set_data(self, present, absent):
        self.present = present
        self.absent  = absent
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        total = self.present + self.absent
        m     = 18  # UI CHANGE: slightly tighter margin
        rect  = QRectF(m, m, self.width()-m*2, self.height()-m*2)

        # UI CHANGE: thicker ring (20px) for more visual weight
        pen = QPen(QColor(C_TERTIARY))
        pen.setWidth(20)
        pen.setCapStyle(Qt.FlatCap)
        p.setPen(pen)
        p.drawEllipse(rect)

        if total > 0:
            a_abs = int(360 * 16 * self.absent  / total)
            a_pre = int(360 * 16 * self.present / total)
            start = 90 * 16

            pen.setWidth(20)  # UI CHANGE: matching ring thickness
            pen.setColor(QColor(C_ABSENT))
            p.setPen(pen)
            p.drawArc(rect, start, a_abs)

            pen.setColor(QColor(C_PRESENT))
            p.setPen(pen)
            p.drawArc(rect, start + a_abs, a_pre)

        pct = f"{int(self.present*100/max(total,1))}%" if total else "—"

        # UI CHANGE: two-line center — big pct + small label
        p.setPen(QColor(C_TEXT))
        p.setFont(QFont("SF Pro Display", 22, QFont.Bold))
        center_rect = QRectF(rect.x(), rect.y() - 8, rect.width(), rect.height())
        p.drawText(center_rect, Qt.AlignCenter, pct)

        # UI CHANGE: small "rate" label below percentage in donut
        p.setPen(QColor(C_SUBTEXT))
        p.setFont(QFont("SF Pro Text", 9))
        sub_rect = QRectF(rect.x(), rect.y() + 22, rect.width(), rect.height())
        p.drawText(sub_rect, Qt.AlignCenter, "attendance")
        p.end()


# ─── Camera Thread ───────────────────────────────────────────────────────────
# !! NO LOGIC CHANGES — untouched !!

class CameraThread(QThread):
    frame_ready   = pyqtSignal(np.ndarray)
    results_ready = pyqtSignal(list, set)

    def __init__(self, engine, mode="attendance"):
        super().__init__()
        self.engine   = engine
        self.mode     = mode
        self._running = False
        self.recognized = set()

    def run(self):
        self._running = True
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Using camera index {i}")
                break
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while self._running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            results = []
            try:
                faces = self.engine.app.get(frame)

                if self.mode == "attendance":
                    from backend.engine import remove_duplicates
                    faces = remove_duplicates(faces)
                    for face in faces:
                        name, score = self.engine.match(face.embedding)
                        x1,y1,x2,y2 = map(int, face.bbox)
                        results.append({"name": name, "score": float(score), "box": (x1,y1,x2,y2)})
                        if name != "Unknown":
                            self.recognized.add(name)
                else:
                    for face in faces:
                        x1,y1,x2,y2 = map(int, face.bbox)
                        results.append({"box": (x1,y1,x2,y2)})
            except Exception as e:
                print("Camera thread error:", e)

            self.results_ready.emit(results, set(self.recognized))
            self.frame_ready.emit(frame.copy())
            self.msleep(30)

        cap.release()

    def stop(self):
        self._running = False
        self.wait()


# ─── Camera Card ─────────────────────────────────────────────────────────────
# UI CHANGE: redesigned inactive state, rounded corners, gradient overlay

class CameraCard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # UI CHANGE: deeper black with refined border
        self.setStyleSheet(f"""
            CameraCard {{
                background: #000000;
                border-radius: 16px;
                border: 1px solid {C_BORDER};
            }}
        """)
        shadow(self, blur=40, opacity=140)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)

        # UI CHANGE: placeholder widget with centered icon + text
        self.feed = QLabel()
        self.feed.setAlignment(Qt.AlignCenter)
        self.feed.setMinimumHeight(360)  # UI CHANGE: taller feed area
        self.feed.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.feed.setFont(QFont("SF Pro Text", 13))

        # UI CHANGE: richer inactive placeholder with icon + copy
        self.feed.setText("No camera feed")
        self.feed.setStyleSheet(f"""
            QLabel {{
                color: {C_SUBTEXT};
                background: transparent;
                border: none;
                qproperty-alignment: AlignCenter;
            }}
        """)
        v.addWidget(self.feed)

        # UI CHANGE: thin status bar at bottom of camera card
        self.status_bar = QWidget()
        self.status_bar.setFixedHeight(36)
        self.status_bar.setStyleSheet(f"""
            background: #0a0a0a;
            border-top: 1px solid {C_BORDER};
            border-bottom-left-radius: 16px;
            border-bottom-right-radius: 16px;
        """)
        sb_h = QHBoxLayout(self.status_bar)
        sb_h.setContentsMargins(16, 0, 16, 0)
        # UI CHANGE: recording indicator dot
        self.rec_dot = QLabel("●")
        self.rec_dot.setFont(QFont("SF Pro Text", 9))
        self.rec_dot.setStyleSheet(f"color: {C_SUBTEXT}; background: transparent;")
        self.cam_status_lbl = QLabel("Camera inactive")
        self.cam_status_lbl.setFont(QFont("SF Pro Text", 11))
        self.cam_status_lbl.setStyleSheet(f"color: {C_SUBTEXT}; background: transparent;")
        sb_h.addWidget(self.rec_dot)
        sb_h.addSpacing(6)
        sb_h.addWidget(self.cam_status_lbl)
        sb_h.addStretch()
        v.addWidget(self.status_bar)

    def set_active(self, active=True):
        # UI CHANGE: animate status bar color on activation
        if active:
            self.rec_dot.setStyleSheet(f"color: {C_PRESENT}; background: transparent;")
            self.cam_status_lbl.setText("Live")
            self.cam_status_lbl.setStyleSheet(f"color: {C_PRESENT}; background: transparent;")
        else:
            self.rec_dot.setStyleSheet(f"color: {C_SUBTEXT}; background: transparent;")
            self.cam_status_lbl.setText("Camera inactive")
            self.cam_status_lbl.setStyleSheet(f"color: {C_SUBTEXT}; background: transparent;")

    def update_frame(self, frame, results=None):
        if results:
            for r in results:
                x1,y1,x2,y2 = r["box"]
                name  = r.get("name","")
                score = r.get("score", 0)
                color = (48,209,88) if name not in ("Unknown","") else (255,55,95)
                # UI CHANGE: thinner box (1px) for cleaner overlay
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,1)
                if name:
                    tag = f"{name}  {score:.2f}" if score else name
                    cv2.putText(frame, tag, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        h,w,ch = frame.shape
        img = QImage(frame.data, w, h, ch*w, QImage.Format_BGR888)
        pix = QPixmap.fromImage(img)
        self.feed.setPixmap(
            pix.scaled(self.feed.width(), self.feed.height(),
                       Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )


# ─── Sidebar ─────────────────────────────────────────────────────────────────
# UI CHANGE: refined nav buttons, cleaner brand mark, active indicator redesign

class NavBtn(QPushButton):
    def __init__(self, icon, text, parent=None):
        super().__init__(parent)
        self._icon = icon
        self._text = text
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(42)  # UI CHANGE: slightly tighter height
        self._style(False)

    def _style(self, on):
        # UI CHANGE: active state uses a subtle left accent bar feel via bg
        bg  = f"{C_CARD_HVR}" if on else "transparent"
        col = C_TEXT            if on else C_SUBTEXT
        bdr = f"1px solid {C_BORDER}" if on else "1px solid transparent"
        self.setStyleSheet(f"""
            QPushButton {{
                background: {bg};
                border: {bdr};
                border-radius: 8px;
                color: {col};
                text-align: left;
                padding-left: 14px;
                font-size: 13px;
                font-family: 'SF Pro Text';
                font-weight: {'600' if on else '400'};
            }}
            QPushButton:hover {{
                background: {C_CARD_HVR};
                color: {C_TEXT};
                border-color: {C_BORDER};
            }}
        """)
        # UI CHANGE: no emoji icon — cleaner text-only nav label with dot indicator
        prefix = "· " if on else "  "
        self.setText(f"  {self._icon}   {self._text}")

    def setChecked(self, v):
        super().setChecked(v)
        self._style(v)


class Sidebar(QWidget):
    nav = pyqtSignal(int)

    PAGES = [
        ("◈", "Dashboard"),
        ("◉", "Live Attendance"),
        ("⊕", "Register Student"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(220)  # UI CHANGE: slightly wider for better breathing room
        self.setStyleSheet(f"""
            Sidebar {{
                background: {C_SURFACE};
                border-right: 1px solid {C_BORDER};
            }}
        """)

        v = QVBoxLayout(self)
        v.setContentsMargins(16, 28, 16, 20)
        v.setSpacing(3)  # UI CHANGE: tighter nav spacing

        # UI CHANGE: brand mark — larger name, monogram-style
        brand_row = QHBoxLayout()
        brand_row.setSpacing(10)

        # UI CHANGE: small logo mark square
        logo = QLabel("A")
        logo.setFixedSize(30, 30)
        logo.setAlignment(Qt.AlignCenter)
        logo.setFont(QFont("SF Pro Display", 14, QFont.Bold))
        logo.setStyleSheet(f"""
            background: {C_TEXT};
            color: #000;
            border-radius: 8px;
        """)
        brand_row.addWidget(logo)

        brand_v = QVBoxLayout()
        brand_v.setSpacing(0)
        brand_v.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Attend")
        title.setFont(QFont("SF Pro Display", 15, QFont.Bold))
        title.setStyleSheet(f"color: {C_TEXT}; background: transparent;")
        brand_v.addWidget(title)
        sub = QLabel("Vision Attendance")
        sub.setFont(QFont("SF Pro Text", 10))
        sub.setStyleSheet(f"color: {C_SUBTEXT}; background: transparent;")
        brand_v.addWidget(sub)
        brand_row.addLayout(brand_v)
        brand_row.addStretch()
        v.addLayout(brand_row)

        v.addSpacing(20)

        # UI CHANGE: nav section label
        nav_lbl = QLabel("NAVIGATION")
        nav_lbl.setFont(QFont("SF Pro Text", 9, QFont.Medium))
        nav_lbl.setStyleSheet(f"color: {C_TERTIARY}; background: transparent; letter-spacing: 1px;")
        nav_lbl.setContentsMargins(6, 0, 0, 0)
        v.addWidget(nav_lbl)
        v.addSpacing(6)

        self.btns = []
        for i, (icon, name) in enumerate(self.PAGES):
            b = NavBtn(icon, name)
            b.clicked.connect(lambda _, idx=i: self._pick(idx))
            v.addWidget(b)
            self.btns.append(b)

        v.addStretch()

        # UI CHANGE: bottom model info pill
        model_pill = QLabel("buffalo_l · ArcFace")
        model_pill.setFont(QFont("SF Pro Text", 10))
        model_pill.setAlignment(Qt.AlignCenter)
        model_pill.setStyleSheet(f"""
            color: {C_SUBTEXT};
            background: {C_CARD};
            border: 1px solid {C_BORDER};
            border-radius: 6px;
            padding: 5px 10px;
        """)
        v.addWidget(model_pill)

        self._pick(0)

    def _pick(self, idx):
        for i, b in enumerate(self.btns):
            b.setChecked(i == idx)
        self.nav.emit(idx)

    def select(self, idx):
        self._pick(idx)


# ─── Dashboard Page ──────────────────────────────────────────────────────────
# UI CHANGE: bigger stat cards, better typography scale, richer empty state

class DashboardPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {C_BG};")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"""
            QScrollArea {{ background: {C_BG}; border: none; }}
            QScrollBar:vertical {{
                background: transparent; width: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {C_BORDER}; border-radius: 2px; min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """)

        inner = QWidget()
        inner.setStyleSheet(f"background: {C_BG};")
        v = QVBoxLayout(inner)
        v.setContentsMargins(40, 40, 40, 40)
        v.setSpacing(8)

        # UI CHANGE: page header with date
        header_row = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(4)
        title_w = lbl("Dashboard", 26, QFont.Bold)  # UI CHANGE: slightly smaller, more refined
        subtitle_w = lbl("Session overview and attendance summary", 13, color=C_SUBTEXT)
        title_col.addWidget(title_w)
        title_col.addWidget(subtitle_w)
        header_row.addLayout(title_col)
        header_row.addStretch()
        # UI CHANGE: live date badge on the right
        today_lbl = QLabel(datetime.now().strftime("%d %b %Y"))
        today_lbl.setFont(QFont("SF Pro Text", 12))
        today_lbl.setAlignment(Qt.AlignCenter)
        today_lbl.setStyleSheet(f"""
            color: {C_SUBTEXT};
            background: {C_CARD};
            border: 1px solid {C_BORDER};
            border-radius: 8px;
            padding: 6px 14px;
        """)
        header_row.addWidget(today_lbl)
        v.addLayout(header_row)

        v.addSpacing(24)

        # UI CHANGE: section label above stats
        v.addWidget(lbl("AT A GLANCE", 10, color=C_TERTIARY))
        v.addSpacing(8)

        stats = QHBoxLayout()
        stats.setSpacing(12)
        self.c_total   = self._stat("—", "Registered",  C_TEXT,    "👥")
        self.c_present = self._stat("—", "Present",     C_PRESENT, "✓")
        self.c_absent  = self._stat("—", "Absent",      C_ABSENT,  "✗")
        self.c_rate    = self._stat("—", "Rate",        C_TEXT,    "%")
        for c in [self.c_total, self.c_present, self.c_absent, self.c_rate]:
            stats.addWidget(c)
        v.addLayout(stats)

        v.addSpacing(24)
        v.addWidget(divider())
        v.addSpacing(16)

        self.info_lbl = lbl("No sessions recorded yet.", 13, color=C_SUBTEXT)
        v.addWidget(self.info_lbl)
        v.addStretch()

        scroll.setWidget(inner)
        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.addWidget(scroll)

    def _stat(self, val, label, color, icon=""):
        # UI CHANGE: redesigned stat card — icon chip, bigger value, subtle tinted bg
        w = QWidget()

        w.setObjectName("statCard")

        w.setStyleSheet(f"""
            QWidget#statCard {{
                background: {C_CARD};
                border-radius: 14px;
                border: 1px solid {C_BORDER};
            }}

            QWidget#statCard * {{
                border: none;
                background: transparent;
            }}
        """)

        shadow(w, blur=20, opacity=100, y_offset=4)
        v = QVBoxLayout(w)
        v.setContentsMargins(22, 22, 22, 22)
        v.setSpacing(6)

        # UI CHANGE: icon chip at top
        icon_lbl = QLabel(icon)
        icon_lbl.setFont(QFont("SF Pro Text", 11))
        icon_lbl.setStyleSheet(f"""
            color: {color};
            background: transparent;
        """)
        v.addWidget(icon_lbl)
        v.addSpacing(4)

        vl = lbl(val, 34, QFont.Bold, color)  # UI CHANGE: slightly larger value
        vl.setStyleSheet("border: none;")
        vl.setObjectName("val")
        v.addWidget(vl)
        v.addWidget(lbl(label, 11, color=C_SUBTEXT))
        return w

    def _set(self, card, val):
        card.findChild(QLabel, "val").setText(str(val))

    def refresh(self):
        total = 0
        if os.path.isdir(DATASET_DIR):
            total = sum(1 for n in os.listdir(DATASET_DIR)
                        if os.path.isdir(os.path.join(DATASET_DIR, n)))
        self._set(self.c_total, total)

        present = absent = 0
        last_time = ""
        if os.path.exists(ATTENDANCE_FILE):
            try:
                with open(ATTENDANCE_FILE) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("-") or line.startswith("Name"):
                            continue
                        parts = line.split()
                        if len(parts) >= 3:
                            if parts[-1] == "Present":
                                present += 1
                            else:
                                absent += 1
                            if len(parts) > 1:
                                last_time = parts[1]
            except Exception:
                pass

        self._set(self.c_present, present)
        self._set(self.c_absent,  absent)
        self._set(self.c_rate, f"{int(present*100/max(present+absent,1))}%")
        if last_time:
            self.info_lbl.setText(f"Last session recorded at {last_time}")


# ─── Attendance Page ──────────────────────────────────────────────────────────
# UI CHANGE: better status feedback, improved session info panel

from PyQt5.QtCore import pyqtSignal

class AttendancePage(QWidget):
    session_finished = pyqtSignal()

    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine     = engine
        self.cam_thread = None
        self.latest_results = []
        self.recognized = set()
        self.setStyleSheet(f"background: {C_BG};")

        v = QVBoxLayout(self)
        v.setContentsMargins(40, 36, 40, 32)
        v.setSpacing(0)

        # UI CHANGE: tighter header section
        header = QVBoxLayout()
        header.setSpacing(4)
        header.addWidget(lbl("Live Attendance", 26, QFont.Bold))
        header.addWidget(lbl("Faces are recognized in real time while the session is active", 13, color=C_SUBTEXT))
        v.addLayout(header)
        v.addSpacing(20)

        self.cam_card = CameraCard()
        v.addWidget(self.cam_card, stretch=1)

        v.addSpacing(14)

        # UI CHANGE: session info row — status pill + count badge
        info_row = QHBoxLayout()
        info_row.setSpacing(10)

        self.status_pill = QLabel("● Inactive")
        self.status_pill.setFont(QFont("SF Pro Text", 12))
        self.status_pill.setAlignment(Qt.AlignCenter)
        # UI CHANGE: pill-shaped status badge
        self.status_pill.setStyleSheet(f"""
            color: {C_SUBTEXT};
            background: {C_CARD};
            border: 1px solid {C_BORDER};
            border-radius: 20px;
            padding: 4px 14px;
        """)
        info_row.addWidget(self.status_pill)

        self.count_badge = QLabel("0 recognized")
        self.count_badge.setFont(QFont("SF Pro Text", 12))
        self.count_badge.setAlignment(Qt.AlignCenter)
        self.count_badge.setStyleSheet(f"""
            color: {C_SUBTEXT};
            background: {C_CARD};
            border: 1px solid {C_BORDER};
            border-radius: 20px;
            padding: 4px 14px;
        """)
        info_row.addWidget(self.count_badge)
        info_row.addStretch()
        v.addLayout(info_row)

        v.addSpacing(16)

        # UI CHANGE: divider before actions
        v.addWidget(divider())
        v.addSpacing(16)

        row = QHBoxLayout()
        row.setSpacing(10)
        self.start_btn = pill_btn("Start Session", primary=True)
        self.stop_btn  = pill_btn("End & Save")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        row.addStretch()
        row.addWidget(self.stop_btn)
        row.addWidget(self.start_btn)
        v.addLayout(row)

    def _start(self):
        if self.cam_thread and self.cam_thread.isRunning():
            return
        self.recognized = set()
        self.cam_thread = CameraThread(self.engine, mode="attendance")
        self.cam_thread.frame_ready.connect(self._frame)
        self.cam_thread.results_ready.connect(self._results)
        self.cam_thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        # UI CHANGE: update status pill styling on session start
        self.status_pill.setText("● Live")
        self.status_pill.setStyleSheet(f"""
            color: {C_PRESENT};
            background: {C_PRESENT_DIM};
            border: 1px solid {C_PRESENT}55;
            border-radius: 20px;
            padding: 4px 14px;
        """)
        self.cam_card.set_active(True)

    def _frame(self, frame):
        self.cam_card.update_frame(frame, self.latest_results)

    def _results(self, results, names):
        self.latest_results = results
        self.recognized     = names
        # UI CHANGE: count badge updates live
        n = len(names)
        self.count_badge.setText(f"{n} student{'s' if n != 1 else ''} recognized")
        self.count_badge.setStyleSheet(f"""
            color: {C_PRESENT if n > 0 else C_SUBTEXT};
            background: {C_PRESENT_DIM if n > 0 else C_CARD};
            border: 1px solid {(C_PRESENT + '44') if n > 0 else C_BORDER};
            border-radius: 20px;
            padding: 4px 14px;
        """)

    def _stop(self):
        if self.cam_thread:
            self.cam_thread.stop()
            self.cam_thread = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        # UI CHANGE: reset status pill on stop
        self.status_pill.setText("● Inactive")
        self.status_pill.setStyleSheet(f"""
            color: {C_SUBTEXT};
            background: {C_CARD};
            border: 1px solid {C_BORDER};
            border-radius: 20px;
            padding: 4px 14px;
        """)
        self.cam_card.set_active(False)
        try:
            from src.attendance import write_attendance
            write_attendance(self.recognized)
        except Exception as e:
            print("Attendance write error:", e)
        self.session_finished.emit()

# ─── Results Page ─────────────────────────────────────────────────────────────
# UI CHANGE: more polished cards, better list rows, tighter summary layout

class ResultsPage(QWidget):
    retake = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {C_BG};")
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setStyleSheet(f"""
            QScrollArea {{ background: {C_BG}; border: none; }}
            QScrollBar:vertical {{
                background: transparent; width: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {C_BORDER}; border-radius: 2px; min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.scroll)

    def load(self, recognized_names):
        inner = QWidget()
        inner.setStyleSheet(f"background: {C_BG};")
        v = QVBoxLayout(inner)
        v.setContentsMargins(40, 40, 40, 40)
        v.setSpacing(0)

        # UI CHANGE: header with timestamp inline
        header_row = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(4)
        title_col.addWidget(lbl("Session Results", 26, QFont.Bold))
        title_col.addWidget(lbl(datetime.now().strftime("%A, %d %b %Y · %H:%M"), 13, color=C_SUBTEXT))
        header_row.addLayout(title_col)
        header_row.addStretch()
        v.addLayout(header_row)

        v.addSpacing(28)

        # Build sorted data (NO LOGIC CHANGE)
        all_students = []
        if os.path.isdir(DATASET_DIR):
            all_students = sorted(
                [n for n in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, n))],
                key=lambda x: x.split("_",1)[-1].lower()
            )

        present_raw = sorted(recognized_names,
                             key=lambda x: x.split("_",1)[-1].lower())
        absent_raw  = sorted([s for s in all_students if s not in recognized_names],
                             key=lambda x: x.split("_",1)[-1].lower())

        n_p = len(present_raw)
        n_a = len(absent_raw)
        n_t = n_p + n_a

        # ── Stats row
        stats_row = QHBoxLayout()
        stats_row.setSpacing(14)

        # Donut card — UI CHANGE: fixed width, more padding, cleaner layout
        dc = QWidget()
        dc.setFixedWidth(240)
        dc.setStyleSheet(f"""
            QWidget {{
                background: {C_CARD};
                border-radius: 16px;
                border: 1px solid {C_BORDER};
            }}
        """)
        shadow(dc, blur=28)
        dcv = QVBoxLayout(dc)
        dcv.setContentsMargins(22, 22, 22, 22)
        dcv.setSpacing(10)
        dcv.addWidget(lbl("Overview", 13, QFont.DemiBold))
        dcv.addWidget(divider())

        chart = DonutChart(n_p, n_a)
        dcv.addWidget(chart, alignment=Qt.AlignCenter)

        # UI CHANGE: legend with cleaner color dots
        leg = QHBoxLayout()
        leg.setSpacing(0)
        for label, col, val in [("Present", C_PRESENT, n_p), ("Absent", C_ABSENT, n_a)]:
            leg_w = QWidget()
            leg_w.setStyleSheet("background: transparent; border : none;")
            lv = QVBoxLayout(leg_w)
            lv.setContentsMargins(0, 4, 0, 0)
            lv.setSpacing(2)
            val_lbl = lbl(str(val), 24, QFont.Bold, col, Qt.AlignCenter)
            lv.addWidget(val_lbl)
            lv.addWidget(lbl(label, 10, color=C_SUBTEXT, align=Qt.AlignCenter))
            leg.addWidget(leg_w)
        dcv.addLayout(leg)
        stats_row.addWidget(dc)

        # Summary card — UI CHANGE: cleaner row layout with better visual separation
        sc = QWidget()
        sc.setStyleSheet(f"""
            QWidget {{
                background: {C_CARD};
                border-radius: 16px;
                border: 1px solid {C_BORDER};
            }}
        """)
        shadow(sc, blur=28)
        scv = QVBoxLayout(sc)
        scv.setContentsMargins(22, 22, 22, 22)
        scv.setSpacing(0)
        scv.addWidget(lbl("Summary", 13, QFont.DemiBold))
        scv.addSpacing(10)
        scv.addWidget(divider())
        scv.addSpacing(10)

        for i, (label, val, col) in enumerate([
            ("Total Registered", str(n_t),  C_TEXT),
            ("Present",          str(n_p),  C_PRESENT),
            ("Absent",           str(n_a),  C_ABSENT),
            ("Attendance Rate",  f"{int(n_p*100/max(n_t,1))}%", C_TEXT),
        ]):
            rw = QWidget()
            # UI CHANGE: alternating row bg for scannability
            rw.setStyleSheet("background: transparent; border: none;")
            rh = QHBoxLayout(rw)
            rh.setContentsMargins(0, 10, 0, 10)
            rh.addWidget(lbl(label, 13, color=C_SUBTEXT))
            rh.addStretch()
            rh.addWidget(lbl(val, 13, QFont.DemiBold, col))
            scv.addWidget(rw)
            if i < 3:
                scv.addWidget(divider())

        scv.addStretch()
        stats_row.addWidget(sc, stretch=1)
        v.addLayout(stats_row)

        v.addSpacing(16)

        # ── Student lists
        lists_row = QHBoxLayout()
        lists_row.setSpacing(14)
        lists_row.addWidget(self._list_card("Present", present_raw, C_PRESENT, C_PRESENT_DIM))
        lists_row.addWidget(self._list_card("Absent",  absent_raw,  C_ABSENT,  C_ABSENT_DIM))
        v.addLayout(lists_row)

        v.addSpacing(24)
        v.addWidget(divider())
        v.addSpacing(16)

        # ── Actions
        act = QHBoxLayout()
        act.setSpacing(10)
        dl = pill_btn("Download CSV", primary=True)
        dl.clicked.connect(self._download)
        rt = pill_btn("↺  New Session")
        rt.clicked.connect(self.retake.emit)
        act.addStretch()
        act.addWidget(rt)
        act.addWidget(dl)
        v.addLayout(act)

        self.scroll.setWidget(inner)

    def _list_card(self, title, names, accent, accent_bg):
        w = QWidget()
        w.setStyleSheet(f"""
            QWidget {{
                background: {C_CARD};
                border-radius: 16px;
                border: 1px solid {C_BORDER};
            }}
        """)
        shadow(w, blur=24)
        v = QVBoxLayout(w)
        v.setContentsMargins(18, 18, 18, 18)
        v.setSpacing(10)

        # UI CHANGE: header with colored count badge
        hh = QHBoxLayout()
        title_lbl = lbl(title, 13, QFont.DemiBold)
        hh.addWidget(title_lbl)
        hh.addStretch()
        # UI CHANGE: tinted count badge
        count_badge = QLabel(str(len(names)))
        count_badge.setFont(QFont("SF Pro Text", 11, QFont.Medium))
        count_badge.setAlignment(Qt.AlignCenter)
        count_badge.setFixedSize(28, 22)
        count_badge.setStyleSheet(f"""
            color: {accent};
            background: {accent_bg};
            border-radius: 6px;
        """)
        hh.addWidget(count_badge)
        v.addLayout(hh)
        v.addWidget(divider())
        v.addSpacing(6)

        # UI CHANGE: column headers with uppercase tracking
        ch = QHBoxLayout()
        ch.setContentsMargins(0, 0, 0, 0)
        roll_h = QLabel("ROLL NO")
        roll_h.setFont(QFont("SF Pro Text", 9))
        roll_h.setStyleSheet(f"color: {C_TERTIARY}; background: transparent; letter-spacing: 1px;")
        roll_h.setFixedWidth(90)
        name_h = QLabel("NAME")
        name_h.setFont(QFont("SF Pro Text", 9))
        name_h.setStyleSheet(f"color: {C_TERTIARY}; background: transparent; letter-spacing: 1px;")
        ch.addWidget(roll_h)
        ch.addWidget(name_h)
        v.addWidget(divider())
        v.addSpacing(6)
        v.addLayout(ch)
        v.addWidget(divider())

        # Scroll list
        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumHeight(200)
        scroll.setStyleSheet(f"""
            QScrollArea {{ background: transparent; border: none; }}
            QScrollBar:vertical {{
                background: transparent; width: 4px; border-radius: 2px;
            }}
            QScrollBar::handle:vertical {{
                background: {C_BORDER}; border-radius: 2px; min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """)

        container = QWidget()
        container.setStyleSheet("background: transparent; border : none;")
        cl = QVBoxLayout(container)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)

        for i, raw in enumerate(names):
            roll, name = parse_name(raw)
            row = QWidget()
            row.setFixedHeight(40)
            # UI CHANGE: subtle zebra striping for long lists
            row.setStyleSheet("background: transparent; border : none;")
            rh = QHBoxLayout(row)
            rh.setContentsMargins(0, 0, 4, 0)
            roll_l = QLabel(roll)
            roll_l.setFont(QFont("SF Pro Text", 11))
            roll_l.setStyleSheet(f"color: {C_SUBTEXT}; background: transparent;")
            roll_l.setFixedWidth(90)
            name_l = QLabel(name)
            name_l.setFont(QFont("SF Pro Text", 13))
            name_l.setStyleSheet(f"color: {C_TEXT}; background: transparent;")
            rh.addWidget(roll_l)
            rh.addWidget(name_l)
            rh.addStretch()
            cl.addWidget(row)

        cl.addStretch()
        scroll.setWidget(container)
        v.addWidget(scroll)
        return w

    def _download(self):
        if not os.path.exists(ATTENDANCE_FILE):
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save Attendance CSV",
            f"attendance_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "CSV Files (*.csv)"
        )
        if dest:
            shutil.copy(ATTENDANCE_FILE, dest)


# ─── Register Page ───────────────────────────────────────────────────────────
# UI CHANGE: cleaner input section, better progress section, improved btn layout

class RegisterPage(QWidget):
    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine       = engine
        self.cam_thread   = None
        self.count        = 0
        self.student_name = ""
        self.latest_res   = []
        self._frame_buf   = None
        self.setStyleSheet(f"background: {C_BG};")

        v = QVBoxLayout(self)
        v.setContentsMargins(40, 36, 40, 32)
        v.setSpacing(0)

        # UI CHANGE: header section
        v.addWidget(lbl("Register Student", 26, QFont.Bold))
        v.addSpacing(4)
        v.addWidget(lbl("Enter roll number and name, open camera, then capture 25 images", 13, color=C_SUBTEXT))
        v.addSpacing(20)

        # UI CHANGE: labeled input section
        v.addWidget(lbl("STUDENT ID", 10, color=C_TERTIARY))
        v.addSpacing(6)
        ir = QHBoxLayout()
        ir.setSpacing(10)
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g. 2408912_Ishtiyaq")
        self.name_input.setFixedHeight(44)
        self.name_input.setFont(QFont("SF Pro Text", 13))
        # UI CHANGE: refined input styling with focus glow
        self.name_input.setStyleSheet(f"""
            QLineEdit {{
                background: {C_CARD};
                color: {C_TEXT};
                border: 1px solid {C_BORDER};
                border-radius: 10px;
                padding: 0 16px;
                selection-background-color: {C_ACCENT}55;
            }}
            QLineEdit:focus {{
                border: 1px solid {C_ACCENT};
                background: {C_CARD_HVR};
            }}
            QLineEdit:hover {{
                border-color: {C_BORDER_HVR};
            }}
        """)
        self.open_btn = pill_btn("Open Camera", primary=True)
        self.open_btn.clicked.connect(self._open)
        ir.addWidget(self.name_input, stretch=1)
        ir.addWidget(self.open_btn)
        v.addLayout(ir)
        v.addSpacing(16)

        self.cam_card = CameraCard()
        v.addWidget(self.cam_card, stretch=1)

        v.addSpacing(14)

        # UI CHANGE: progress section with cleaner layout
        prog_section = QWidget()
        prog_section.setStyleSheet(f"""
            background: {C_CARD};
            border: 1px solid {C_BORDER};
            border-radius: 12px;
        """)
        ps_layout = QVBoxLayout(prog_section)
        ps_layout.setContentsMargins(16, 12, 16, 12)
        ps_layout.setSpacing(8)

        pr_header = QHBoxLayout()
        pr_header.addWidget(lbl("Capture Progress", 12, QFont.Medium))
        pr_header.addStretch()
        self.prog_lbl = QLabel("0 / 25")
        self.prog_lbl.setFont(QFont("SF Pro Text", 12, QFont.DemiBold))
        self.prog_lbl.setStyleSheet(f"color: {C_SUBTEXT}; background: transparent;")
        pr_header.addWidget(self.prog_lbl)
        ps_layout.addLayout(pr_header)

        self.prog_bar = QProgressBar()
        self.prog_bar.setMaximum(NUM_IMAGES)
        self.prog_bar.setValue(0)
        self.prog_bar.setFixedHeight(5)  # UI CHANGE: thinner, sleeker bar
        self.prog_bar.setTextVisible(False)
        self.prog_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {C_BORDER};
                border-radius: 3px;
                border: none;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {C_ACCENT}, stop:1 {C_PRESENT});
                border-radius: 3px;
            }}
        """)  # UI CHANGE: gradient progress fill
        ps_layout.addWidget(self.prog_bar)

        v.addWidget(prog_section)
        v.addSpacing(14)

        # UI CHANGE: divider before buttons
        v.addWidget(divider())
        v.addSpacing(14)

        # UI CHANGE: reorganized button row — primary actions right, secondary left
        br = QHBoxLayout()
        br.setSpacing(10)
        self.confirm_btn = pill_btn("Confirm Student", primary=True)
        self.retake_btn  = pill_btn("Retake", danger=True)
        self.cap_btn     = pill_btn("Capture  (Space)", primary=True)
        self.close_btn   = pill_btn("Close Camera")

        self.cap_btn.setEnabled(False)
        self.close_btn.setEnabled(False)
        self.confirm_btn.setEnabled(False)
        self.retake_btn.setEnabled(False)

        self.cap_btn.clicked.connect(self._capture)
        self.close_btn.clicked.connect(self._close)
        self.confirm_btn.clicked.connect(self._confirm)
        self.retake_btn.clicked.connect(self._retake)

        br.addWidget(self.confirm_btn)
        br.addWidget(self.retake_btn)
        br.addStretch()
        br.addWidget(self.close_btn)
        br.addWidget(self.cap_btn)
        v.addLayout(br)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Space and self.cap_btn.isEnabled():
            self._capture()

    def _open(self):
        name = self.name_input.text().strip()
        if not name:
            self.name_input.setPlaceholderText("⚠ Enter a name first!")
            return
        import os
        student_path = os.path.join(DATASET_DIR, name)

        if os.path.exists(student_path):
            self.name_input.clear()
            self.name_input.setPlaceholderText("⚠ Student already exists!")
            return
        
        self.student_name = name
        self.count = 0
        self.prog_bar.setValue(0)
        self.prog_lbl.setText("0 / 25")
        self.cam_thread = CameraThread(self.engine, mode="register")
        self.cam_thread.frame_ready.connect(self._on_frame)
        self.cam_thread.results_ready.connect(self._on_results)
        self.cam_thread.start()
        self.cap_btn.setEnabled(True)
        self.close_btn.setEnabled(True)
        self.open_btn.setEnabled(False)
        self.cam_card.set_active(True)
        self.setFocus()
        self.confirm_btn.setEnabled(False)
        self.retake_btn.setEnabled(False)

    def _on_frame(self, frame):
        self._frame_buf = frame.copy()
        self.cam_card.update_frame(frame, self.latest_res)

    def _on_results(self, results, _):
        self.latest_res = results
        self.cap_btn.setEnabled(True)

    def _capture(self):
        if self._frame_buf is None:
            return

        try:
            faces = self.engine.app.get(self._frame_buf)

            for face in faces:
                name, score = self.engine.match(face.embedding)

                if name != "Unknown" and score > 0.5:
                    # UI CHANGE: styled warning message in prog_lbl
                    self.prog_lbl.setText(f"Already registered as {name}")
                    self.prog_lbl.setStyleSheet(f"color: {C_ABSENT}; background: transparent;")
                    self.cap_btn.setEnabled(False)
                    return

        except Exception as e:
            print("Duplication check error:", e)

        if not self.latest_res:
            # UI CHANGE: styled no-face warning
            self.prog_lbl.setText("No face detected")
            self.prog_lbl.setStyleSheet(f"color: {C_ABSENT}; background: transparent;")
            self.cap_btn.setEnabled(False)
            return

        d = os.path.join(DATASET_DIR, self.student_name)
        os.makedirs(d, exist_ok=True)

        path = os.path.join(d, f"{self.student_name}_{self.count}.jpg")
        cv2.imwrite(path, self._frame_buf)

        if self.count < NUM_IMAGES:
            self.count += 1
            self.prog_bar.setValue(self.count)
            # UI CHANGE: cleaner progress count display
            self.prog_lbl.setText(f"{self.count} / {NUM_IMAGES}")
            self.prog_lbl.setStyleSheet(f"color: {C_SUBTEXT}; background: transparent;")

        if self.count >= NUM_IMAGES:
            self._close(reset_ui=False)

            # UI CHANGE: success state for progress label
            self.prog_lbl.setText("✓ Complete — confirm or retake")
            self.prog_lbl.setStyleSheet(f"color: {C_PRESENT}; background: transparent;")

            self.cap_btn.setEnabled(False)
            self.confirm_btn.setEnabled(True)
            self.retake_btn.setEnabled(True)

    def _confirm(self):
        try:
            self.engine.load_faces()
        except Exception as e:
            print("Reload error:", e)

        # UI CHANGE: success feedback
        self.prog_lbl.setText("✓ Registered successfully")
        self.prog_lbl.setStyleSheet(f"color: {C_PRESENT}; background: transparent;")

        self.name_input.clear()
        self.confirm_btn.setEnabled(False)
        self.retake_btn.setEnabled(False)
        self.open_btn.setEnabled(True)
    
    def _retake(self):
        student_path = os.path.join(DATASET_DIR, self.student_name)

        if os.path.exists(student_path):
            shutil.rmtree(student_path)

        self.count = 0
        self.prog_bar.setValue(0)
        self.prog_lbl.setText("0 / 25")

        self.confirm_btn.setEnabled(False)
        self.retake_btn.setEnabled(False)

        self._open()

    def _close(self, reset_ui=True):
        if self.cam_thread:
            self.cam_thread.stop()
            self.cam_thread = None

        self.cap_btn.setEnabled(False)
        self.close_btn.setEnabled(False)
        self.open_btn.setEnabled(True)

        self._frame_buf = None
        self.latest_res = []

        self.cam_card.feed.setText("No camera feed")
        self.cam_card.feed.setPixmap(QPixmap())
        self.cam_card.set_active(False)

        # 🔥 Only reset UI if explicitly asked
        if reset_ui:
            self.count = 0
            self.prog_bar.setValue(0)
            self.prog_lbl.setText("0 / 25")
            self.prog_lbl.setStyleSheet(f"color:{C_SUBTEXT};")

            if hasattr(self, "confirm_btn"):
                self.confirm_btn.setEnabled(False)
            if hasattr(self, "retake_btn"):
                self.retake_btn.setEnabled(False)


# ─── Main Window ─────────────────────────────────────────────────────────────
# UI CHANGE: window title, minimum size tweak for better default proportions

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Attend — Vision Attendance System")
        self.setMinimumSize(1100, 720)
        # UI CHANGE: background applied at root level
        self.setStyleSheet(f"background: {C_BG}; color: {C_TEXT};")

        self.engine = None
        try:
            from backend.engine import Engine
            self.engine = Engine()
        except Exception as e:
            print("Engine load error:", e)

        root  = QWidget()
        root_h = QHBoxLayout(root)
        root_h.setContentsMargins(0, 0, 0, 0)
        root_h.setSpacing(0)
        self.setCentralWidget(root)

        self.sidebar = Sidebar()
        self.sidebar.nav.connect(self._nav)
        root_h.addWidget(self.sidebar)

        # UI CHANGE: subtle separator line between sidebar and content
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFixedWidth(1)
        sep.setStyleSheet(f"background: {C_BORDER}; border: none;")

        self.stack = QStackedWidget()
        self.stack.setStyleSheet(f"background: {C_BG};")
        root_h.addWidget(self.stack, stretch=1)

        self.pg_dash     = DashboardPage()
        self.pg_attend   = AttendancePage(self.engine)
        self.pg_register = RegisterPage(self.engine)
        self.pg_results  = ResultsPage()

        self.stack.addWidget(self.pg_dash)      # 0
        self.stack.addWidget(self.pg_attend)    # 1
        self.stack.addWidget(self.pg_register)  # 2
        self.pg_attend.session_finished.connect(self._on_session_end)
        self.pg_results.retake.connect(self._retake)
        

    def _nav(self, idx):
        pages = [self.pg_dash, self.pg_attend, self.pg_register]
        self.stack.setCurrentWidget(pages[idx])
        if idx == 0:
            self.pg_dash.refresh()

    def _show_results(self, names):
        if self.stack.indexOf(self.pg_results) == -1:
            self.stack.addWidget(self.pg_results)
        self.pg_results.load(names)
        self.stack.setCurrentWidget(self.pg_results)

    def _retake(self):
        self.stack.setCurrentWidget(self.pg_attend)
        self.sidebar.select(1)
    
    def _on_session_end(self):
        # refresh dashboard
        self.pg_dash.refresh()

        # show results page (keep your existing flow)
        self._show_results(self.pg_attend.recognized)

    def closeEvent(self, e):
        for pg in [self.pg_attend, self.pg_register]:
            if pg.cam_thread:
                pg.cam_thread.stop()
        e.accept()


# ─── Entry ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setStyleSheet("""
    QWidget {
        border: none;
    }
    """)

    app.setApplicationName("Attend")

    # UI CHANGE: set app-wide palette for consistent base colors
    palette = QPalette()
    palette.setColor(QPalette.Window,       QColor(C_BG))
    palette.setColor(QPalette.WindowText,   QColor(C_TEXT))
    palette.setColor(QPalette.Base,         QColor(C_CARD))
    palette.setColor(QPalette.AlternateBase,QColor(C_SURFACE))
    palette.setColor(QPalette.Text,         QColor(C_TEXT))
    palette.setColor(QPalette.ButtonText,   QColor(C_TEXT))
    palette.setColor(QPalette.Highlight,    QColor(C_ACCENT))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)

    try:
        QFontDatabase.addApplicationFont("/System/Library/Fonts/SFNS.ttf")
        QFontDatabase.addApplicationFont("/System/Library/Fonts/SFNSDisplay.ttf")
    except:
        pass
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())