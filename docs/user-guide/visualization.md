# TMD visualization

The CLI offers several visualization backends (matplotlib, plotly, seaborn, and optional Polyscope) and multiple modes. Subcommands: `basic`, `3d`, `profile`, `contour`, `fancy`, `compare`, Polyscope helpers (`ps-3d`, `ps-pointcloud`, `ps-mesh`), `backends`, and `examples`.

Replace `path/to/file.tmd` with your own TMD file (none are committed to this repository). For a consolidated list of CLI examples, run:

```bash
python tmd_cli.py visualize examples
# or, if installed from PyPI:
tmd-process visualize examples
```

## 2D heatmap

```bash
python tmd_cli.py visualize basic path/to/file.tmd
python tmd_cli.py visualize basic path/to/file.tmd --plotter plotly --colormap plasma
```

## 3D surface

```bash
python tmd_cli.py visualize 3d path/to/file.tmd
python tmd_cli.py visualize 3d path/to/file.tmd --z-scale 2.5 --plotter plotly --wireframe
```

## Profile

```bash
python tmd_cli.py visualize profile path/to/file.tmd
python tmd_cli.py visualize profile path/to/file.tmd --row 50 --plotter seaborn
python tmd_cli.py visualize profile path/to/file.tmd --no-show-markers --colormap coolwarm
```

## Contour

```bash
python tmd_cli.py visualize contour path/to/file.tmd
python tmd_cli.py visualize contour path/to/file.tmd --levels 15 --plotter plotly --colormap terrain
```

## Enhanced (`fancy`)

Requires seaborn for the best experience.

```bash
python tmd_cli.py visualize fancy path/to/file.tmd --plotter seaborn
python tmd_cli.py visualize fancy path/to/file.tmd --mode distribution --plotter seaborn
python tmd_cli.py visualize fancy path/to/file.tmd --mode joint --plotter seaborn
```

## Compare

```bash
python tmd_cli.py visualize compare path/to/file.tmd
python tmd_cli.py visualize compare path/to/file.tmd --second-file path/to/other.tmd
python tmd_cli.py visualize compare path/to/file.tmd --plotter seaborn
```

## Polyscope

Interactive 3D when Polyscope is installed.

```bash
python tmd_cli.py visualize ps-3d path/to/file.tmd --z-scale 3.0
python tmd_cli.py visualize ps-3d path/to/file.tmd --wireframe
python tmd_cli.py visualize ps-pointcloud path/to/file.tmd --point-size 4.0 --sample-rate 2
python tmd_cli.py visualize ps-pointcloud path/to/file.tmd --output points.png --no-interactive
python tmd_cli.py visualize ps-mesh path/to/file.tmd --wireframe --no-smooth
```

### Polyscope tips

- Mouse: rotate, pan, zoom.
- Right-click for viewport options.
- `a` resets the camera; space toggles the settings panel; `p` saves a screenshot.

## Backends and output

| Backend    | Notes                                      |
|------------|---------------------------------------------|
| matplotlib | PNG/PDF; good default for scripts          |
| plotly     | Interactive HTML when saved as `.html`     |
| seaborn    | Statistical / aesthetic plots              |
| polyscope  | Interactive 3D (optional dependency)       |

```bash
python tmd_cli.py visualize backends
python tmd_cli.py visualize basic path/to/file.tmd --output out.png
python tmd_cli.py visualize 3d path/to/file.tmd --plotter plotly --output out.html
python tmd_cli.py visualize profile path/to/file.tmd --output profile.png --auto-open
```

Use `--plotter auto` to pick a sensible default when multiple backends are available.
