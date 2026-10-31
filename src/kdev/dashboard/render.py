"""Render DashboardData to a self-contained static HTML page (inline CSS/JS, no network at view time,
light/dark via prefers-color-scheme). Owner: P8. Template: templates/index.html."""
from __future__ import annotations

from kdev.dashboard.collect import DashboardData


def render_html(data: DashboardData) -> str:
    raise NotImplementedError("dashboard.render.render_html (P8)")


def write(data: DashboardData, out_dir: str) -> dict[str, str]:
    """Write index.html + dashboard.json atomically; return {'html': path, 'json': path}."""
    raise NotImplementedError("dashboard.render.write (P8)")
