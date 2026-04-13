from pathlib import Path
import shutil


def clear_directory_contents(directory: Path) -> tuple[int, int]:
	"""Delete all files/subfolders inside a directory, keeping the directory itself."""
	removed_files = 0
	removed_dirs = 0

	if not directory.exists():
		directory.mkdir(parents=True, exist_ok=True)
		return removed_files, removed_dirs

	for item in directory.iterdir():
		if item.is_dir():
			shutil.rmtree(item)
			removed_dirs += 1
		else:
			item.unlink()
			removed_files += 1

	return removed_files, removed_dirs


def main() -> int:
	base_dir = Path(__file__).parent
	targets = [
		base_dir / "charts",
		base_dir / "data",
	]

	print("=" * 60)
	print("Cleaner: clear charts and data")
	print("=" * 60)

	for target in targets:
		files, dirs = clear_directory_contents(target)
		print(f"Cleared {target.name}: removed {files} files, {dirs} folders")

	print("Done.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
