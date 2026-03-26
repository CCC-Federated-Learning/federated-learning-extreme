from pathlib import Path
import shutil


def clear_directory_contents(directory: Path) -> tuple[int, int]:
	"""Delete all files/subfolders inside a directory, keeping the directory itself and .gitkeep."""
	removed_files = 0
	removed_dirs = 0

	if not directory.exists():
		directory.mkdir(parents=True, exist_ok=True)
		return removed_files, removed_dirs

	for item in directory.iterdir():
		if item.name == '.gitkeep':
			continue  # Always preserve .gitkeep
		if item.is_dir():
			shutil.rmtree(item)
			removed_dirs += 1
		else:
			item.unlink()
			removed_files += 1

	return removed_files, removed_dirs


def main() -> int:
	base_dir = Path(__file__).parent
	
	# Directories to preserve
	safe_dirs = {
		"PUT-DATA-THERE",
		"generate_charts",
		"clearner.py",  # Script itself
		"generator.py",  # Script itself
		"move_to_stage02.py",  # Script itself
		".gitkeep",  # Git marker for empty directories
		"__pycache__",  # Python cache (should remain clean)
	}

	print("=" * 60)
	print("Cleaner: Clear contents + Remove everything except core folders")
	print("=" * 60)

	# First: Clear contents of PUT-DATA-THERE and generate_charts
	print("\n🧹 Clearing directory contents...")
	targets_to_clear = [
		base_dir / "PUT-DATA-THERE",
		base_dir / "generate_charts",
	]
	
	for target in targets_to_clear:
		if target.exists():
			files, dirs = clear_directory_contents(target)
			print(f"  ✓ Cleared {target.name}: removed {files} files, {dirs} folders")

	# Second: Remove other files/folders in stage01
	print("\n🗑️  Removing non-essential folders...")
	removed_files = 0
	removed_dirs = 0
	
	for item in base_dir.iterdir():
		if item.name not in safe_dirs:
			try:
				if item.is_dir():
					shutil.rmtree(item)
					removed_dirs += 1
					print(f"  ✓ Removed folder: {item.name}")
				else:
					item.unlink()
					removed_files += 1
					print(f"  ✓ Removed file: {item.name}")
			except Exception as e:
				print(f"  ❌ Error removing {item.name}: {e}")

	print(f"\n✅ Total removed: {removed_files} files, {removed_dirs} folders")
	print("✅ Done.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
