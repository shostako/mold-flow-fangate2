"""mold-flow-fangate2 core package."""

from .fill_player import (
    CONTROLS_HEIGHT_PX,
    build_fill_player_html,
    fill_player_height_px,
    wrap_standalone_html,
)
from .geometry import Geometry, build_demo_geometry
from .materials import MaterialDB, cross_wlf_viscosity
from .multilayer_solver import MultilayerFlowResult, MultilayerHeleShawSolver
from .solver import FlowResult, HeleShawSolver
from .visualizer import (
    draw_gate_markers,
    export_frames,
    fill_frame_fractions,
    fill_frame_times,
    render_core_layer_map,
    render_fill_animation,
    render_pressure_map,
    render_skin_layer_map,
    render_weldlines,
)
from .visualizer_3d import (
    render_3d_fill_time,
    render_3d_pressure,
    render_3d_thickness_map,
)

__all__ = [
    "MaterialDB",
    "cross_wlf_viscosity",
    "Geometry",
    "build_demo_geometry",
    "HeleShawSolver",
    "MultilayerHeleShawSolver",
    "MultilayerFlowResult",
    "FlowResult",
    "render_fill_animation",
    "render_pressure_map",
    "render_weldlines",
    "render_skin_layer_map",
    "render_core_layer_map",
    "draw_gate_markers",
    "export_frames",
    "fill_frame_times",
    "fill_frame_fractions",
    "build_fill_player_html",
    "CONTROLS_HEIGHT_PX",
    "fill_player_height_px",
    "wrap_standalone_html",
    "render_3d_thickness_map",
    "render_3d_fill_time",
    "render_3d_pressure",
]
