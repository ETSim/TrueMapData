#!/usr/bin/env python3
"""Model generation core functionality for TMD CLI."""

from pathlib import Path
from typing import Optional, Callable, Any
from enum import Enum
import logging
import psutil
import math

# Import CLI utilities
from tmd.cli.core import (
    console,
    print_warning,
    print_error,
    print_success,
    load_config,
    load_tmd_file
)

# Set up logging
logger = logging.getLogger(__name__)

BUILTIN_TEMPLATE_NAMES = {"plane", "sphere", "cube"}


def _resolve_map_slots(
    tmd_data: Any,
    output_root: Path,
    stem: str,
    *,
    compress: int = 75,
    normalize: bool = True,
) -> dict[str, str]:
    """Resolve map slot paths and lazily generate missing maps."""
    from tmd.image.export.exporter import MapExporter

    textures_dir = output_root / "textures"
    textures_dir.mkdir(parents=True, exist_ok=True)

    slot_types = {
        "map_kd": "height",
        "map_bump": "normal",
        "map_disp": "displacement",
        "map_pr": "roughness",
    }
    slots: dict[str, str] = {}
    metadata = dict(getattr(tmd_data, "metadata", {}) or {})

    for slot, map_type in slot_types.items():
        out_path = textures_dir / f"{stem}_{map_type}.png"
        if not out_path.exists():
            MapExporter.export_map(
                tmd_data.height_map,
                str(out_path),
                map_type,
                compress=compress,
                format="png",
                normalize=normalize,
                metadata=metadata,
            )
        if out_path.exists():
            slots[slot] = str(out_path)
    return slots


def _read_template_obj_metrics(template_mesh_path: Path, obj_units_to_mm: float) -> tuple[float, float, bool]:
    """Read template OBJ X/Z span in mm and whether UVs are present."""
    verts: list[tuple[float, float, float]] = []
    has_uv = False
    for line in template_mesh_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("v "):
            parts = stripped.split()
            if len(parts) >= 4:
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif stripped.startswith("vt "):
            has_uv = True

    if not verts:
        raise ValueError(f"Template OBJ has no vertices: {template_mesh_path}")

    xs = [v[0] for v in verts]
    zs = [v[2] for v in verts]
    span_x_mm = (max(xs) - min(xs)) * obj_units_to_mm
    span_z_mm = (max(zs) - min(zs)) * obj_units_to_mm
    if span_x_mm <= 0 or span_z_mm <= 0:
        raise ValueError(
            f"Template OBJ bounds are invalid (x={span_x_mm:.4f}mm, z={span_z_mm:.4f}mm)."
        )
    return span_x_mm, span_z_mm, has_uv


def _default_template_fixtures_dir() -> Path:
    """Return built-in template fixtures directory."""
    return Path(__file__).resolve().parents[2] / "fixtures" / "templates"


def _resolve_template_mesh_path(
    *,
    template_mesh_path: Optional[Path],
    template_plane_dir: Optional[Path],
    template_kind: str,
    template_fixtures_dir: Optional[Path],
) -> Path:
    """Resolve custom or built-in template mesh path."""
    if template_mesh_path is not None:
        return Path(template_mesh_path)

    if template_plane_dir is not None:
        return Path(template_plane_dir) / "plane.obj"

    kind = template_kind.lower().strip()
    if kind not in BUILTIN_TEMPLATE_NAMES:
        raise ValueError(
            f"template_kind must be one of {sorted(BUILTIN_TEMPLATE_NAMES)} when --template-mesh is not provided."
        )
    base = Path(template_fixtures_dir) if template_fixtures_dir else _default_template_fixtures_dir()
    return base / kind / f"{kind}.obj"


def _generate_uv_sphere_obj(path: Path, *, rings: int = 16, segments: int = 32, radius: float = 1.0) -> None:
    """Generate a UV sphere OBJ template with UVs and normals."""
    if rings < 3 or segments < 3:
        raise ValueError("rings and segments must be >= 3")

    lines: list[str] = ["mtllib sphere.mtl", "o sphere"]
    verts: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    norms: list[tuple[float, float, float]] = []

    for r in range(rings + 1):
        v = r / rings
        theta = math.pi * v
        y = math.cos(theta) * radius
        ring_radius = math.sin(theta) * radius

        for s in range(segments + 1):
            u = s / segments
            phi = 2.0 * math.pi * u
            x = math.cos(phi) * ring_radius
            z = math.sin(phi) * ring_radius
            verts.append((x, y, z))
            uvs.append((u, 1.0 - v))

            length = math.sqrt(x * x + y * y + z * z) or 1.0
            norms.append((x / length, y / length, z / length))

    for x, y, z in verts:
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
    for u, v in uvs:
        lines.append(f"vt {u:.6f} {v:.6f}")
    for nx, ny, nz in norms:
        lines.append(f"vn {nx:.6f} {ny:.6f} {nz:.6f}")

    lines.append("usemtl TemplateMaterial")
    row = segments + 1
    for r in range(rings):
        for s in range(segments):
            a = r * row + s + 1
            b = a + 1
            c = (r + 1) * row + s + 1
            d = c + 1
            lines.append(f"f {a}/{a}/{a} {c}/{c}/{c} {b}/{b}/{b}")
            lines.append(f"f {b}/{b}/{b} {c}/{c}/{c} {d}/{d}/{d}")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _ensure_template_quality(template_mesh_path: Path, template_kind: str) -> Path:
    """Upgrade low-fidelity built-in templates when needed."""
    if template_kind.lower().strip() != "sphere":
        return template_mesh_path

    try:
        contents = template_mesh_path.read_text(encoding="utf-8")
    except Exception:
        return template_mesh_path

    # Legacy fixture is an octahedron (8 faces). Replace with UV sphere.
    if contents.count("\nf ") <= 12:
        _generate_uv_sphere_obj(template_mesh_path)
    return template_mesh_path


def _require_mm_length(metadata: dict[str, Any], key: str) -> float:
    raw = metadata.get(key, None)
    if raw is None:
        raise ValueError(
            f"Missing required TMD metadata '{key}' for physical tiling. "
            "Provide TMD with x_length/y_length metadata or add an override pathway."
        )
    value = float(raw)
    if value <= 0:
        raise ValueError(f"TMD metadata '{key}' must be > 0, got {value}")
    return value


def _resolve_mm_per_pixel(
    metadata: dict[str, Any],
    override: Optional[float],
    *,
    fallback_mm_per_pixel: float = 0.06,
) -> float:
    """Resolve mm-per-pixel from override or common metadata keys."""
    if override is not None:
        value = float(override)
        if value <= 0:
            raise ValueError(f"tmd_mm_per_pixel must be > 0, got {value}")
        return value

    # Primary key used by apply-on-mesh, plus compatibility aliases.
    for key in ("mm_per_pixel", "mmpp"):
        raw = metadata.get(key, None)
        if raw is None:
            continue
        value = float(raw)
        if value > 0:
            return value

    # Fallback for TMDs that only carry dimensions and physical lengths.
    width = metadata.get("width", None)
    x_length = metadata.get("x_length", None)
    if width not in (None, 0) and x_length is not None:
        value = float(x_length) / float(width)
        if value > 0:
            return value

    height = metadata.get("height", None)
    y_length = metadata.get("y_length", None)
    if height not in (None, 0) and y_length is not None:
        value = float(y_length) / float(height)
        if value > 0:
            return value

    if fallback_mm_per_pixel > 0:
        return float(fallback_mm_per_pixel)

    raise ValueError(
        "Missing or invalid physical scale for apply-on-mesh. Provide --tmd-mm-per-pixel "
        "or include positive 'mm_per_pixel'/'mmpp' in TMD metadata."
    )


def apply_maps_to_mesh(
    tmd_file: Path,
    output_root: Path,
    *,
    template_mesh_path: Optional[Path] = None,
    template_plane_dir: Optional[Path] = None,
    template_kind: str = "plane",
    template_fixtures_dir: Optional[Path] = None,
    output_prefix: str = "applied_mesh",
    application_mode: str = "uv",
    uv_alignment_mode: str = "preserve",
    compress: int = 75,
    normalize: bool = True,
    obj_units_to_mm: float = 1000.0,
    tmd_mm_per_pixel: Optional[float] = None,
    max_texture_edge: Optional[int] = 8192,
) -> dict[str, str]:
    """
    Apply TMD-derived maps to an external template mesh and emit OBJ+MTL+textures.
    """
    from tmd.core import TMD
    from tmd.model.formats.obj import create_mtl_file
    from tmd.image.export.exporter import MapExporter

    mode = application_mode.lower().strip()
    if mode not in {"uv", "displace"}:
        raise ValueError("application_mode must be 'uv' or 'displace'")

    uv_mode = uv_alignment_mode.lower().strip()
    if uv_mode not in {"preserve", "remap_bbox"}:
        raise ValueError("uv_alignment_mode must be 'preserve' or 'remap_bbox'")

    if obj_units_to_mm <= 0:
        raise ValueError(f"obj_units_to_mm must be positive, got {obj_units_to_mm}")

    template_mesh_path = _resolve_template_mesh_path(
        template_mesh_path=template_mesh_path,
        template_plane_dir=template_plane_dir,
        template_kind=template_kind,
        template_fixtures_dir=template_fixtures_dir,
    )
    if not template_mesh_path.is_file():
        raise FileNotFoundError(f"Template mesh not found: {template_mesh_path}")
    template_mesh_path = _ensure_template_quality(template_mesh_path, template_kind)

    output_root.mkdir(parents=True, exist_ok=True)
    tmd_data = TMD.load(str(tmd_file))
    metadata = dict(getattr(tmd_data, "metadata", {}) or {})
    mm_per_pixel = _resolve_mm_per_pixel(metadata, tmd_mm_per_pixel, fallback_mm_per_pixel=0.06)

    span_x_mm, span_z_mm, has_uv = _read_template_obj_metrics(template_mesh_path, obj_units_to_mm)
    if mode == "uv" and not has_uv:
        raise ValueError(
            f"Template mesh has no UVs: {template_mesh_path}. UV mode requires vt coordinates."
        )

    x_length_mm = _require_mm_length(metadata, "x_length")
    y_length_mm = _require_mm_length(metadata, "y_length")

    target_w_px = max(1, int(round(span_x_mm / mm_per_pixel)))
    target_h_px = max(1, int(round(span_z_mm / mm_per_pixel)))
    tile_w_px = max(1, int(round(x_length_mm / mm_per_pixel)))
    tile_h_px = max(1, int(round(y_length_mm / mm_per_pixel)))
    scale_cap = 1.0
    if max_texture_edge is not None and max_texture_edge > 0:
        max_edge = max(target_w_px, target_h_px)
        if max_edge > max_texture_edge:
            scale_cap = float(max_texture_edge) / float(max_edge)
            target_w_px = max(1, int(round(target_w_px * scale_cap)))
            target_h_px = max(1, int(round(target_h_px * scale_cap)))
            tile_w_px = max(1, int(round(tile_w_px * scale_cap)))
            tile_h_px = max(1, int(round(tile_h_px * scale_cap)))

    slots = MapExporter.export_material_binding_maps_with_physical_tiling(
        tmd_data.height_map,
        str(output_root / "textures"),
        output_prefix,
        tile_size_px=(tile_w_px, tile_h_px),
        target_size_px=(target_w_px, target_h_px),
        compress=compress,
        normalize=normalize,
        metadata=metadata,
    )

    out_obj = output_root / f"{output_prefix}_bundle.obj"
    out_mtl = output_root / f"{output_prefix}_bundle.mtl"
    material_name = f"Material_measured_{output_prefix}"

    create_mtl_file(
        str(out_mtl),
        material_map_bindings=slots,
        obj_dir=str(output_root),
        material_name=material_name,
    )

    src_lines = template_mesh_path.read_text(encoding="utf-8").splitlines()
    out_lines: list[str] = []
    saw_mtllib = False
    saw_usemtl = False
    for line in src_lines:
        if line.startswith("mtllib "):
            out_lines.append(f"mtllib {out_mtl.name}")
            saw_mtllib = True
        elif line.startswith("usemtl "):
            out_lines.append(f"usemtl {material_name}")
            saw_usemtl = True
        else:
            out_lines.append(line)

    if uv_mode == "remap_bbox":
        # Optional troubleshooting mode; default remains preserving template UVs.
        from tmd.model.formats.obj import apply_uv_margin_to_obj_lines

        out_lines = apply_uv_margin_to_obj_lines(out_lines, margin=0.05)
    if not saw_mtllib:
        out_lines.insert(0, f"mtllib {out_mtl.name}")
    if not saw_usemtl:
        # Keep geometry untouched; both modes rely on material maps.
        out_lines.insert(1, f"usemtl {material_name}")
    out_obj.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")

    result = {
        "obj": str(out_obj),
        "mtl": str(out_mtl),
        "textures_dir": str(output_root / "textures"),
        "template_mesh_path": str(template_mesh_path),
        "template_kind": template_kind,
        "application_mode": mode,
        "uv_alignment_mode": uv_mode,
        "obj_units_to_mm": obj_units_to_mm,
        "tmd_mm_per_pixel": mm_per_pixel,
        "target_size_px": f"{target_w_px}x{target_h_px}",
        "tile_size_px": f"{tile_w_px}x{tile_h_px}",
        "scale_cap": scale_cap,
        "max_texture_edge": max_texture_edge if max_texture_edge is not None else "None",
    }
    if mode == "displace":
        # Displacement mode is material-driven for OBJ (map_disp); geometry remains template-based.
        result["displacement_behavior"] = "material_map_disp"
    return result

# Add enum definitions
class MeshMethod(str, Enum):
    """Mesh generation methods."""
    ADAPTIVE = "adaptive"
    QUADTREE = "quadtree"

class QualityPreset(str, Enum):
    """Quality presets for mesh generation."""
    DRAFT = "draft"
    NORMAL = "normal"
    HIGH = "high"
    ULTRA = "ultra"

def get_quality_params(preset: QualityPreset) -> dict:
    """Get parameters for a quality preset."""
    presets = {
        QualityPreset.DRAFT: {
            'error_threshold': 0.1,
            'max_triangles': 10000,
            'min_quad_size': 8,
            'max_quad_size': 64,
            'max_subdivisions': 6
        },
        QualityPreset.NORMAL: {
            'error_threshold': 0.05,
            'max_triangles': 50000,
            'min_quad_size': 4,
            'max_quad_size': 32,
            'max_subdivisions': 8
        },
        QualityPreset.HIGH: {
            'error_threshold': 0.01,
            'max_triangles': 200000,
            'min_quad_size': 2,
            'max_quad_size': 16,
            'max_subdivisions': 10
        },
        QualityPreset.ULTRA: {
            'error_threshold': 0.005,
            'max_triangles': 500000,
            'min_quad_size': 1,
            'max_quad_size': 8,
            'max_subdivisions': 12
        }
    }
    return presets[preset]

def export_model(
    input_file: Path,
    output_file: Path,
    format: str,
    **kwargs
) -> bool:
    """Export model in specified format."""
    try:
        from time import time
        from rich.panel import Panel
        from rich.table import Table
        from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
        
        start_time = time()
        
        # Load TMD file
        from tmd.core import TMD
        tmd_data = TMD.load(str(input_file))
        
        # TMD File Info Table
        info_table = Table(title="TMD File Information", show_header=True, expand=True)
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="green")
        
        info_table.add_row("Size", f"{tmd_data.height_map.shape[0]} × {tmd_data.height_map.shape[1]}")
        info_table.add_row("Height Range", f"{tmd_data.height_map.min():.2f} to {tmd_data.height_map.max():.2f}")
        info_table.add_row("Memory Usage", f"{tmd_data.height_map.nbytes / 1024:.1f} KB")
        
        # Add metadata to the table
        for key, value in tmd_data.metadata.items():
            info_table.add_row(key, str(value))
        
        console.print(info_table)
        console.print()

        # Show detailed export parameters
        param_table = Table(title="Export Parameters", show_header=True, expand=True)
        param_table.add_column("Parameter", style="cyan")
        param_table.add_column("Value", style="green")
        param_table.add_column("Description", style="yellow")

        parameter_info = {
            'format': (format.upper(), "Output format"),
            'method': (kwargs.get('method', 'adaptive'), "Mesh generation method"),
            'scale': (kwargs.get('scale', 1.0), "Height scale factor"),
            'error_threshold': (kwargs.get('error_threshold', 0.05), "Max error for mesh simplification"),
            'max_triangles': (kwargs.get('max_triangles', 50000), "Maximum triangle count"),
            'binary': (kwargs.get('binary', True), "Use binary format if supported"),
            'min_quad_size': (kwargs.get('min_quad_size', 4), "Minimum quad size for subdivision"),
            'max_quad_size': (kwargs.get('max_quad_size', 64), "Maximum quad size for subdivision"),
            'max_subdivisions': (kwargs.get('max_subdivisions', 4), "Maximum subdivision depth"),
            'detail_boost': (kwargs.get('detail_boost', 1.0), "Detail enhancement factor"),
            'coordinate_system': (kwargs.get('coordinate_system', 'right-handed'), "Coordinate system orientation"),
            'uv_method': (kwargs.get('uv_method', 'planar'), "UV mapping method"),
            'optimize': (kwargs.get('optimize', True), "Optimize mesh after generation"),
            'calculate_normals': (kwargs.get('calculate_normals', True), "Generate vertex normals"),
            'texture': (kwargs.get('texture', False), "Generate texture from heightmap")
        }

        for param, (value, desc) in parameter_info.items():
            param_table.add_row(param, str(value), desc)
        
        console.print(param_table)
        console.print()
        
        # Map CLI parameters to ExportConfig parameters
        config_params = {
            'triangulation_method': str(kwargs.get('method', 'adaptive')).replace('MeshMethod.', '').lower(),
            'error_threshold': kwargs.get('error_threshold', 0.05),
            'min_quad_size': kwargs.get('min_quad_size', 4),
            'max_quad_size': kwargs.get('max_quad_size', 64),
            'max_triangles': kwargs.get('max_triangles', 50000),
            'simplify_ratio': kwargs.get('simplify_ratio', 0.25),
            'z_scale': kwargs.get('scale', 1.0),
            'max_subdivisions': kwargs.get('max_subdivisions', 4),
            'detail_boost': kwargs.get('detail_boost', 1.0),
            'binary': kwargs.get('binary', True),
            'x_length': tmd_data.metadata.get('x_length', 1.0),
            'y_length': tmd_data.metadata.get('y_length', 1.0),
            'x_offset': tmd_data.metadata.get('x_offset', 0.0),
            'y_offset': tmd_data.metadata.get('y_offset', 0.0),
        }

        # Progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="green"),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False
        ) as progress:
            # Create tasks for each stage
            main_task = progress.add_task("[cyan]Generating mesh...", total=100)
            
            def progress_callback(percent):
                progress.update(main_task, completed=percent)
            
            # Create config and export
            from tmd.model.config import ExportConfig
            from tmd.model.factory import ModelExporterFactory
            
            config = ExportConfig(**config_params)
            config.progress_callback = progress_callback
            config.bind_material_maps = bool(kwargs.get("bind_material_maps", False))
            if config.bind_material_maps and format.lower() in {"obj", "gltf", "glb"}:
                slots = _resolve_map_slots(
                    tmd_data,
                    output_file.parent,
                    output_file.stem,
                    compress=int(kwargs.get("material_map_compress", 75)),
                    normalize=bool(kwargs.get("material_map_normalize", True)),
                )
                config.material_map_bindings = slots
                config.extra["material_map_bindings"] = slots
            if "base_height" in kwargs:
                config.base_height = float(kwargs["base_height"])
            if "save_heightmap" in kwargs:
                config.extra["save_heightmap"] = bool(kwargs["save_heightmap"])
            if "colormap" in kwargs:
                config.extra["colormap"] = kwargs["colormap"]

            factory = ModelExporterFactory()
            result = factory.export(
                input_file=str(input_file),  # Ensure string path
                output_file=str(output_file), # Ensure string path
                format_name=format.lower(),   # Ensure lowercase format
                config=config
            )

        # Show completion message with timing
        elapsed = time() - start_time
        if result:
            success_panel = Panel.fit(
                f"Successfully exported [cyan]{input_file.name}[/] to [green]{output_file}[/]\n"
                f"Format: [yellow]{format.upper()}[/]\n"
                f"Time: [cyan]{elapsed:.1f}[/] seconds",
                title="Export Complete",
                border_style="green"
            )
            console.print(success_panel)
            return True

        print_error("Export failed")
        return False
        
    except Exception as e:
        print_error(f"Failed to export model: {e}")
        return False

def batch_export_models(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    format: str = "stl",
    pattern: str = "*.tmd",
    quality: QualityPreset = QualityPreset.NORMAL,
    scale: float = 5.0,
    max_workers: int = 1,
    recursive: bool = False,
    **kwargs
) -> bool:
    """Batch process multiple TMD files."""
    try:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        # Find all TMD files
        if recursive:
            tmd_files = list(input_dir.rglob(pattern))
        else:
            tmd_files = list(input_dir.glob(pattern))
        
        if not tmd_files:
            print_error(f"No TMD files found matching pattern '{pattern}' in {input_dir}")
            return False
        
        # Set up output directory
        if output_dir is None:
            output_dir = input_dir / "models"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        console.print(f"[cyan]Found {len(tmd_files)} TMD files to process[/cyan]")
        console.print(f"[yellow]Output directory: {output_dir}[/yellow]")
        
        # Get quality parameters
        quality_params = get_quality_params(quality)
        quality_params.update(kwargs)  # Include any additional kwargs
        
        # Process files
        from rich.progress import Progress
        
        with Progress(console=console) as progress:
            main_task = progress.add_task("[cyan]Processing files...", total=len(tmd_files))
            
            if max_workers == 1:
                # Sequential processing
                for tmd_file in tmd_files:
                    output_file = output_dir / f"{tmd_file.stem}.{format.lower()}"
                    success = export_model(
                        input_file=tmd_file,
                        output_file=output_file,
                        format=format,
                        scale=scale,
                        **quality_params
                    )
                    progress.advance(main_task)
                    if not success:
                        print_warning(f"Failed to process: {tmd_file}")
            else:
                # Parallel processing
                def process_file(tmd_file):
                    output_file = output_dir / f"{tmd_file.stem}.{format.lower()}"
                    return export_model(
                        input_file=tmd_file,
                        output_file=output_file,
                        format=format,
                        scale=scale,
                        **quality_params
                    )
                
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(process_file, f): f for f in tmd_files}
                    for future in as_completed(futures):
                        tmd_file = futures[future]
                        try:
                            success = future.result()
                            if not success:
                                print_warning(f"Failed to process: {tmd_file}")
                        except Exception as e:
                            print_error(f"Error processing {tmd_file}: {e}")
                        progress.advance(main_task)
        
        print_success(f"Batch processing completed. Check {output_dir} for results.")
        return True
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}", exc_info=True)
        print_error(f"Batch processing failed: {e}")
        return False

def list_model_formats():
    """List all available model export formats."""
    try:
        from tmd.model.formats import get_available_formats
        formats = get_available_formats()
        
        from rich.table import Table
        table = Table(title="Available 3D Model Formats", show_header=True, expand=True)
        table.add_column("Format", style="cyan")
        table.add_column("Extension", style="green")
        table.add_column("Description", style="yellow")
        table.add_column("Features", style="blue")
        
        format_info = {
            'stl': ('.stl', 'Stereolithography format', 'Binary/ASCII, widely supported'),
            'obj': ('.obj', 'Wavefront OBJ format', 'Text-based, materials support'),
            'ply': ('.ply', 'Polygon File Format', 'Binary/ASCII, efficient'),
            'gltf': ('.gltf/.glb', 'GL Transmission Format', 'Modern, web-friendly'),
            'usd': ('.usd/.usda', 'Universal Scene Description', 'Advanced features, Pixar standard')
        }
        
        for format_name in formats:
            if format_name in format_info:
                ext, desc, features = format_info[format_name]
                table.add_row(format_name.upper(), ext, desc, features)
            else:
                table.add_row(format_name.upper(), f".{format_name}", "", "")
        
        console.print(table)
        
    except ImportError:
        # Fallback if get_available_formats is not available
        console.print("[cyan]Available 3D model formats:[/cyan]")
        formats = ["stl", "obj", "ply", "gltf", "usd"]
        for format_name in formats:
            console.print(f"  - {format_name}")

def generate_model_command(
    tmd_file: Path,
    output_file: Optional[Path] = None,
    z_scale: float = 1.0,
    base_height: float = 0.0,
    max_triangles: Optional[int] = None,
    error_threshold: float = 0.01,
    coordinate_system: str = "right-handed",
    origin_at_zero: bool = True,
    invert_base: bool = False,
    progress_callback: Optional[Callable[[float], None]] = None
) -> bool:
    """Generate a 3D model from a TMD file (for backwards compatibility)."""
    try:
        # Validate input file
        if not tmd_file.exists():
            print_error(f"Input file does not exist: {tmd_file}")
            return False

        # Check available memory
        try:
            file_size = tmd_file.stat().st_size
            available_memory = psutil.virtual_memory().available
            required_memory = file_size * 4  # Rough estimate
            
            if required_memory > available_memory:
                print_error(f"Not enough memory available. Need {required_memory/(1024**3):.1f}GB but only {available_memory/(1024**3):.1f}GB available")
                return False
        except Exception as e:
            logger.warning(f"Could not check memory requirements: {e}")

        # Load TMD file with progress reporting
        with console.status(f"Loading {tmd_file.name}..."):
            try:
                data = load_tmd_file(tmd_file)
                if not data or not hasattr(data, 'height_map') or data.height_map is None:
                    raise RuntimeError("Invalid or missing height map data")
                
                height_map = data.height_map
                shape = height_map.shape
                
                # Validate dimensions
                if len(shape) != 2:
                    raise ValueError(f"Expected 2D height map, got {len(shape)}D")
                if any(dim > 10000 for dim in shape):
                    raise ValueError(f"Height map too large: {shape}")
                    
                logger.info(f"Loaded heightmap with shape {shape}")
                
            except Exception as e:
                raise RuntimeError(f"Failed to load TMD file: {e}")

        # Create output filename if not specified
        if output_file is None:
            # Load config for default format
            config = load_config()
            default_format = config.get("model", {}).get("default_format", "stl")
            output_format = default_format.lower()
            
            # Create output filename
            try:
                from tmd.cli.core.io import create_output_dir
                output_dir = create_output_dir(subdir="models")
                output_file = output_dir / f"{tmd_file.stem}.{output_format}"
            except ImportError:
                # Fallback if create_output_dir not available
                output_file = tmd_file.with_suffix(f".{output_format}")

        # Import and run model generation
        try:
            from tmd.model.adaptive_mesh import convert_heightmap_to_adaptive_mesh
            
            with console.status("Generating 3D model..."):
                result = convert_heightmap_to_adaptive_mesh(
                    height_map=height_map,
                    output_file=str(output_file),
                    z_scale=z_scale,
                    base_height=base_height,
                    error_threshold=error_threshold,
                    max_triangles=max_triangles,
                    progress_callback=progress_callback,
                    coordinate_system=coordinate_system,
                    origin_at_zero=origin_at_zero,
                    invert_base=invert_base
                )

            if result is None:
                raise RuntimeError("Model generation failed")

            vertices, faces = result
            print_success(f"Generated mesh with {len(vertices)} vertices and {len(faces)} triangles")
            print_success(f"Model saved to {output_file}")
            return True
            
        except ImportError as e:
            # Fallback to using export_model if adaptive_mesh not available
            logger.warning(f"Could not import adaptive_mesh, using fallback: {e}")
            return export_model(
                input_file=tmd_file,
                output_file=output_file,
                format=output_file.suffix[1:] if output_file.suffix else "stl",
                scale=z_scale,
                base_height=base_height,
                max_triangles=max_triangles,
                error_threshold=error_threshold,
                coordinate_system=coordinate_system
            )

    except ImportError as e:
        print_error(f"Missing dependencies: {e}")
        print_warning("Make sure SciPy, NumPy and OpenCV are installed")
        return False
        
    except Exception as e:
        logger.error(f"Model generation failed: {e}", exc_info=True)
        print_error(f"Error generating 3D model: {e}")
        return False