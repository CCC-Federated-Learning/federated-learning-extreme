"""Move first-station outputs into second-station with normalized batch naming."""

import shutil
from pathlib import Path


FIRST_STATION_NAME = "res-drag-set-to-there----first-station"
SECOND_STATION_NAME = "res-happy-ending--------second-station"


def extract_metadata(metadata_path):
    """Extract key metadata from metadata.txt file."""
    metadata = {}
    try:
        with open(metadata_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    metadata[key.strip()] = value.strip()
    except Exception as e:
        print(f"Error reading metadata: {e}")
    return metadata


def get_experiment_config(data_dir: Path):
    """Extract configuration from first available metadata.txt file."""
    if not data_dir.exists():
        print("Error: PUT-DATA-THERE directory not found")
        return None
    
    # Find first metadata file
    for subdir in sorted(data_dir.iterdir()):
        if subdir.is_dir():
            metadata_path = subdir / 'metadata.txt'
            if metadata_path.exists():
                metadata = extract_metadata(metadata_path)
                
                # Extract required fields
                if all(key in metadata for key in ['data_distribution', 'num_rounds', 'local_epochs', 'run_id']):
                    return {
                        'data_distribution': metadata['data_distribution'],
                        'num_rounds': metadata['num_rounds'],
                        'local_epochs': metadata['local_epochs'],
                        'run_id': metadata['run_id']
                    }
    
    print("Error: Could not find valid metadata.txt files (directory may be empty)")
    return None


def create_directory_name(config):
    """Create directory name from configuration."""
    return (f"{config['data_distribution']}-Round-{config['num_rounds']}-"
            f"Epoch-{config['local_epochs']}-{config['run_id']}")


def organize_results():
    """Main function to organize and move results."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    if script_dir.name != FIRST_STATION_NAME:
        print(f"\n⚠️  Warning: script is not under expected first station '{FIRST_STATION_NAME}'")
        print(f"Current script dir: {script_dir}")

    print("\n" + "="*70)
    print("Experiment Results Organization Script")
    print("="*70)
    
    # Check required directories
    put_data_dir = script_dir / 'PUT-DATA-THERE'
    generate_charts_dir = script_dir / 'generate_charts'
    
    if not put_data_dir.exists():
        print("\n❌ Error: PUT-DATA-THERE directory not found")
        return False
    
    if not generate_charts_dir.exists():
        print("\n❌ Error: generate_charts directory not found")
        return False
    
    print(f"\n✓ Found PUT-DATA-THERE directory")
    print(f"✓ Found generate_charts directory")
    
    # Get experiment configuration
    print("\nExtracting experiment configuration...")
    config = get_experiment_config(put_data_dir)
    
    if not config:
        print("❌ Failed to extract experiment configuration")
        return False
    
    print(f"  Data Distribution: {config['data_distribution']}")
    print(f"  Rounds: {config['num_rounds']}")
    print(f"  Local Epochs: {config['local_epochs']}")
    print(f"  Run ID: {config['run_id']}")
    
    # Create directory name
    dir_name = create_directory_name(config)
    print(f"\n📁 Creating directory: {dir_name}")
    
    # Check if second-station exists, create if not
    second_station = project_root / SECOND_STATION_NAME
    second_station.mkdir(exist_ok=True)
    print(f"✓ Ensured {SECOND_STATION_NAME} directory exists")
    
    # Create target directory
    target_dir = second_station / dir_name
    
    if target_dir.exists():
        print(f"\n⚠️  Directory already exists: {target_dir}")
        response = input("Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            print("Cancelled.")
            return False
        shutil.rmtree(target_dir)
        print("Removed existing directory")
    
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created target directory: {target_dir}")
    
    # Move PUT-DATA-THERE
    print("\nMoving PUT-DATA-THERE...")
    src_put_data = put_data_dir
    dst_put_data = target_dir / 'PUT-DATA-THERE'
    
    try:
        shutil.copytree(src_put_data, dst_put_data, dirs_exist_ok=True)
        print(f"✓ Copied PUT-DATA-THERE to {dst_put_data.relative_to(second_station)}")
    except Exception as e:
        print(f"❌ Error copying PUT-DATA-THERE: {e}")
        return False
    
    # Move generate_charts
    print("\nMoving generate_charts...")
    src_charts = generate_charts_dir
    dst_charts = target_dir / 'generate_charts'
    
    try:
        shutil.copytree(src_charts, dst_charts, dirs_exist_ok=True)
        print(f"✓ Copied generate_charts to {dst_charts.relative_to(second_station)}")
    except Exception as e:
        print(f"❌ Error copying generate_charts: {e}")
        return False
    
    # Create summary file
    summary_path = target_dir / 'SUMMARY.txt'
    try:
        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("EXPERIMENT RESULTS SUMMARY\n")
            f.write("="*70 + "\n\n")
            f.write(f"Directory: {dir_name}\n\n")
            f.write("Configuration:\n")
            f.write(f"  Data Distribution: {config['data_distribution']}\n")
            f.write(f"  Training Rounds: {config['num_rounds']}\n")
            f.write(f"  Local Epochs: {config['local_epochs']}\n")
            f.write(f"  Run ID: {config['run_id']}\n\n")
            f.write("Contents:\n")
            f.write("  - PUT-DATA-THERE/: Raw experimental data from all strategies\n")
            f.write("  - generate_charts/: Generated comparison charts and visualizations\n")
        print(f"✓ Created SUMMARY.txt")
    except Exception as e:
        print(f"⚠️  Warning: Could not create SUMMARY.txt: {e}")
    
    print("\n" + "="*70)
    print("✅ SUCCESS: Results organized and moved!")
    print("="*70)
    print(f"\nLocation: {target_dir.relative_to(project_root)}")
    print(f"Full path: {target_dir}\n")
    
    return True


if __name__ == '__main__':
    success = organize_results()
    exit(0 if success else 1)
