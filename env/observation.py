"""Image observation builder — lightweight PIL-based rendering.

Generates top-down intersection frames with:
- Lane occupancy bars (colour-coded by fill ratio)
- Phase colour overlay (green / red / yellow / all-red)
- Emergency vehicle markers (colour + symbol by type)
- Camera blur and weather overlays (rain, fog, night)
- Frame stacking support
"""
from __future__ import annotations

import io
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from config.env_config import EnvConfig
from env.schemas import (
    EmergencyType,
    IntersectionState,
    PhaseEnum,
    TrafficObservation,
    TrafficState,
    WeatherCondition,
)

# ---------------------------------------------------------------------------
# Colour constants
# ---------------------------------------------------------------------------
_PHASE_COLOUR = {
    PhaseEnum.NS_GREEN: (34, 197, 94),    # green
    PhaseEnum.EW_GREEN: (59, 130, 246),   # blue-green
    PhaseEnum.ALL_RED:  (239, 68, 68),    # red
    PhaseEnum.YELLOW:   (234, 179, 8),    # yellow
}
_EMERGENCY_COLOUR = {
    EmergencyType.NONE:      None,
    EmergencyType.POLICE:    (59, 130, 246),   # blue
    EmergencyType.FIRE:      (249, 115, 22),   # orange
    EmergencyType.AMBULANCE: (239, 68, 68),    # red
}
_EMERGENCY_SYMBOL = {
    EmergencyType.NONE:      "",
    EmergencyType.POLICE:    "P",
    EmergencyType.FIRE:      "F",
    EmergencyType.AMBULANCE: "A",
}
_LANE_COLOURS = {
    "low":    (34, 197, 94),    # green — low queue
    "medium": (234, 179, 8),    # yellow
    "high":   (239, 68, 68),    # red — near capacity
}
_ROAD_BG   = (30, 30, 35)
_ROAD_GREY = (80, 80, 90)


# ---------------------------------------------------------------------------
# Metadata feature indices (for documentation clarity)
# ---------------------------------------------------------------------------
# Per intersection, 11 features:
# [q0, q1, q2, q3, phase, phase_timer_norm, yellow_remaining_norm,
#  emergency_type_norm, emergency_lane_norm, weather_norm, spillback_flag]
N_META_FEATURES = 11


def _queue_colour(ratio: float) -> Tuple[int, int, int]:
    if ratio < 0.4:
        return _LANE_COLOURS["low"]
    if ratio < 0.75:
        return _LANE_COLOURS["medium"]
    return _LANE_COLOURS["high"]


class ObservationBuilder:
    """Builds stacked image observations for the traffic environment."""

    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg
        self.H, self.W = cfg.image_size
        self.frame_stack = cfg.frame_stack
        self._frame_buffer: Deque[np.ndarray] = deque(maxlen=self.frame_stack)
        self._rng = np.random.default_rng(cfg.sim.seed + 999)

    def reset(self) -> None:
        """Clear frame buffer."""
        self._rng = np.random.default_rng(self.cfg.sim.seed + 999)
        self._frame_buffer.clear()

    def build(self, state: TrafficState) -> TrafficObservation:
        """Render current state into a stacked observation."""
        frame = self._render_state(state)
        self._frame_buffer.append(frame)

        # Pad with copies of first frame if buffer not full yet
        while len(self._frame_buffer) < self.frame_stack:
            self._frame_buffer.appendleft(frame.copy())

        frames = np.stack(list(self._frame_buffer), axis=0)  # (T, H, W, 3)
        metadata = self._build_metadata(state)

        return TrafficObservation(
            frames=frames,
            metadata=metadata,
            step=state.step,
            intersections=state.intersections,
        )

    # ------------------------------------------------------------------
    # Frame rendering
    # ------------------------------------------------------------------

    def _render_state(self, state: TrafficState) -> np.ndarray:
        """Render all intersections into one image frame."""
        n = len(state.intersections)
        cols = max(1, int(np.ceil(np.sqrt(n))))
        rows = max(1, int(np.ceil(n / cols)))
        cell_h = self.H // rows
        cell_w = self.W // cols

        canvas = Image.new("RGB", (self.W, self.H), color=_ROAD_BG)

        for idx, inter in enumerate(state.intersections):
            r = idx // cols
            c = idx % cols
            x0 = c * cell_w
            y0 = r * cell_h
            cell_img = self._render_intersection(inter, cell_w, cell_h)
            canvas.paste(cell_img, (x0, y0))

        # Weather post-processing
        weather = state.intersections[0].weather if state.intersections else WeatherCondition.CLEAR
        canvas = self._apply_weather(canvas, weather)

        return np.array(canvas, dtype=np.uint8)

    def _render_intersection(
        self, inter: IntersectionState, W: int, H: int
    ) -> Image.Image:
        """Render a single intersection into a WxH image."""
        img = Image.new("RGB", (W, H), color=_ROAD_BG)
        draw = ImageDraw.Draw(img)

        cx, cy = W // 2, H // 2
        road_w = max(6, W // 8)

        # Road surface
        draw.rectangle([cx - road_w, 0, cx + road_w, H], fill=_ROAD_GREY)
        draw.rectangle([0, cy - road_w, W, cy + road_w], fill=_ROAD_GREY)

        # Phase colour on intersection box
        phase_col = _PHASE_COLOUR.get(inter.current_phase, (128, 128, 128))
        if inter.yellow_remaining > 0:
            phase_col = _PHASE_COLOUR[PhaseEnum.YELLOW]
        box_size = road_w + 2
        draw.rectangle(
            [cx - box_size, cy - box_size, cx + box_size, cy + box_size],
            fill=phase_col,
        )

        # Lane occupancy bars (one per lane direction)
        lane_rects = {
            "N": (cx - road_w, 0,       cx + road_w, cy - box_size),
            "S": (cx - road_w, cy + box_size, cx + road_w, H),
            "E": (cx + box_size, cy - road_w, W,            cy + road_w),
            "W": (0,            cy - road_w, cx - box_size, cy + road_w),
        }
        for lane in inter.lanes:
            rect = lane_rects.get(lane.direction)
            if rect is None:
                continue
            x1, y1, x2, y2 = rect
            # Background
            draw.rectangle([x1, y1, x2, y2], fill=_ROAD_GREY)

            ratio = lane.queue_length / max(self.cfg.sim.max_queue_per_lane, 1)
            ratio = min(1.0, ratio)
            col = _queue_colour(ratio)

            # Occluded lane — grey out
            if lane.is_occluded:
                col = (60, 60, 60)

            # Fill bar proportional to queue
            if lane.direction in ("N", "S"):
                bar_h = int((y2 - y1) * ratio)
                draw.rectangle([x1, y2 - bar_h, x2, y2], fill=col)
            else:
                bar_w = int((x2 - x1) * ratio)
                draw.rectangle([x1, y1, x1 + bar_w, y2], fill=col)

            # Emergency marker
            if lane.emergency != EmergencyType.NONE:
                ecol = _EMERGENCY_COLOUR[lane.emergency]
                sym  = _EMERGENCY_SYMBOL[lane.emergency]
                # Small marker square at midpoint of lane
                mx = (x1 + x2) // 2
                my = (y1 + y2) // 2
                ms = max(4, min(W, H) // 10)
                draw.rectangle([mx - ms, my - ms, mx + ms, my + ms], fill=ecol)
                # Symbol text (best-effort, no font file needed)
                try:
                    draw.text((mx - ms // 2, my - ms // 2), sym, fill=(255, 255, 255))
                except Exception:
                    pass

        # Phase timer indicator (small bar at top)
        max_timer = self.cfg.sim.phase_duration_max
        timer_ratio = min(1.0, inter.phase_timer / max_timer)
        draw.rectangle([0, 0, int(W * timer_ratio), 3], fill=(255, 255, 255))

        return img

    # ------------------------------------------------------------------
    # Weather overlays
    # ------------------------------------------------------------------

    def _apply_weather(
        self, img: Image.Image, weather: WeatherCondition
    ) -> Image.Image:
        if weather == WeatherCondition.CLEAR:
            return img
        if weather == WeatherCondition.CLOUDY:
            return self._overlay_fog(img, alpha=60)
        if weather == WeatherCondition.RAIN:
            img = self._overlay_rain(img)
            return img.filter(ImageFilter.GaussianBlur(0.5))
        if weather == WeatherCondition.FOG:
            return self._overlay_fog(img, alpha=140)
        if weather == WeatherCondition.NIGHT:
            return self._darken(img, factor=0.35)
        return img

    def _overlay_fog(self, img: Image.Image, alpha: int = 120) -> Image.Image:
        fog = Image.new("RGBA", img.size, (200, 200, 210, alpha))
        base = img.convert("RGBA")
        base.alpha_composite(fog)
        return base.convert("RGB")

    def _overlay_rain(self, img: Image.Image) -> Image.Image:
        arr = np.array(img, dtype=np.float32)
        H, W = arr.shape[:2]
        n_drops = max(10, (H * W) // 300)
        ys = self._rng.integers(0, H, size=n_drops)
        xs = self._rng.integers(0, W, size=n_drops)
        for y, x in zip(ys, xs):
            length = int(self._rng.integers(4, 12))
            for dy in range(length):
                ny = min(y + dy, H - 1)
                arr[ny, x] = np.clip(arr[ny, x] + 60, 0, 255)
        return Image.fromarray(arr.astype(np.uint8))

    def _darken(self, img: Image.Image, factor: float = 0.4) -> Image.Image:
        arr = (np.array(img, dtype=np.float32) * factor).astype(np.uint8)
        return Image.fromarray(arr)

    # ------------------------------------------------------------------
    # Metadata vector
    # ------------------------------------------------------------------

    def _build_metadata(self, state: TrafficState) -> np.ndarray:
        """Build normalised metadata tensor (n_intersections, N_META_FEATURES)."""
        rows = []
        for inter in state.intersections:
            q = [l.queue_length / self.cfg.sim.max_queue_per_lane
                 for l in inter.lanes[:4]]
            # Pad to 4 lanes if fewer
            while len(q) < 4:
                q.append(0.0)

            phase_norm = inter.current_phase.value / (len(PhaseEnum) - 1)
            timer_norm = min(1.0, inter.phase_timer / self.cfg.sim.phase_duration_max)
            yellow_norm = inter.yellow_remaining / max(self.cfg.sim.yellow_duration, 1)
            emerg_norm = inter.emergency_active.value / (len(EmergencyType) - 1)
            e_lane_norm = (inter.emergency_lane + 1) / self.cfg.lanes_per_intersection
            weather_norm = inter.weather.value / (len(WeatherCondition) - 1)
            spillback = float(inter.spillback_count > 0)

            row = q[:4] + [
                phase_norm,
                timer_norm,
                yellow_norm,
                emerg_norm,
                e_lane_norm,
                weather_norm,
                spillback,
            ]
            rows.append(row)

        return np.array(rows, dtype=np.float32)  # (n_inter, 11)
