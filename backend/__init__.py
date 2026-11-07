

def _load_dotenv() -> None:
    """Tiny .env loader (skeptic/.env). Real environment variables win; empty values are ignored."""
    import os
    from pathlib import Path
    f = Path(__file__).resolve().parents[1] / ".env"
    if not f.is_file():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


_load_dotenv()
