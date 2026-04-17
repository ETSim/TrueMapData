# To view example output as users would: pytest tests/cli/test_examples.py -vv -s

from rich.console import Console

from typer.testing import CliRunner

from tmd.cli.commands.examples import EXAMPLES_MD, show_examples
from tmd.cli.main import app


def test_examples_md_has_expected_anchors():
    assert "# Mesh export" in EXAMPLES_MD
    assert "mesh formats" in EXAMPLES_MD
    assert "visualize backends" in EXAMPLES_MD
    assert "path/to/file.tmd" in EXAMPLES_MD
    assert "polyscope-animate" not in EXAMPLES_MD
    assert "check-polyscope" not in EXAMPLES_MD


def test_visualize_examples_cli():
    runner = CliRunner(env={"TERM": "dumb"}, mix_stderr=False)
    result = runner.invoke(app, ["visualize", "examples"])
    assert result.exit_code == 0, (result.stdout or "") + (result.stderr or "")
    out = (result.stdout or "") + (result.stderr or "")
    assert "Mesh export" in out
    assert "mesh formats" in out
    assert "TMD Command-Line Tool Examples" in out


def test_examples_markdown_renders_in_rich():
    from rich.markdown import Markdown

    console = Console(record=True, force_terminal=False, width=120)
    console.print(Markdown(EXAMPLES_MD))
    text = console.export_text()
    assert "mesh formats" in text
    assert "visualize backends" in text
