from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the public URL benchmark only for PDFs without existing outputs.")
    parser.add_argument("--data-dir", default="benchmark/data", help="Folder containing final report PDFs.")
    parser.add_argument("--out-dir", default="benchmark/output_public_url", help="Folder where benchmark outputs are saved.")
    parser.add_argument("--model", default=None, help="Optional Vertex model override.")
    parser.add_argument("--location", default=None, help="Optional Vertex location override.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    script_dir = Path(__file__).resolve().parent
    benchmark_script = script_dir / "public_url_benchmark.py"

    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")

    missing = []
    for pdf in pdfs:
        expected = out_dir / pdf.stem / f"{pdf.stem}_public_url_benchmark.xlsx"
        if expected.exists():
            print(f"Already done: {pdf.name}")
        else:
            print(f"Missing: {pdf.name}")
            missing.append(pdf)

    print(f"\nMissing companies: {len(missing)}")

    for pdf in missing:
        cmd = [
            sys.executable,
            str(benchmark_script),
            "--pdf", str(pdf),
            "--out-dir", str(out_dir),
        ]
        if args.model:
            cmd.extend(["--model", args.model])
        if args.location:
            cmd.extend(["--location", args.location])

        print("\n" + "=" * 100)
        print(f"Running benchmark for: {pdf.name}")
        print("=" * 100)
        subprocess.run(cmd, check=True)

    print("\nAll missing companies processed.")


if __name__ == "__main__":
    main()
