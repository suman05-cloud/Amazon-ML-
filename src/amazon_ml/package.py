from pathlib import Path
import zipfile

from .data import log
from .validate import validate


def package(data_dir, output_dir, destination, project_root, documentation):
    root = Path(project_root)
    documentation = Path(documentation)
    if not documentation.is_file():
        raise FileNotFoundError(documentation)
    text = documentation.read_text(encoding="utf-8")
    if "[Your Team" in text or "[List all" in text:
        raise ValueError("Fill in team details before creating the final package")
    validate(data_dir, output_dir)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(Path(output_dir) / name, "output/" + name)
        z.write(documentation, "Documentation_template.md")
        prefix = "code/business_entity_resolution/"
        for path in sorted((root / "src").rglob("*.py")):
            z.write(path, prefix + path.relative_to(root).as_posix())
        for name in ("README.md", "requirements.txt", "pyproject.toml", "uv.lock", "LICENSE", "run_local.ps1"):
            z.write(root / name, prefix + name)
        for path in sorted((root / "tests").rglob("*.py")):
            z.write(path, prefix + path.relative_to(root).as_posix())
    temporary.replace(destination)
    log(f"Created {destination}")
