"""Move second-station outputs into final-station with first-station-like naming."""

import shutil
import subprocess
from pathlib import Path


SECOND_STATION_NAME = "res-happy-ending--------second-station"
FINAL_STATION_NAME = "RESULT_Finally_fan_tasty"


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
    """Extract configuration from available metadata.txt files under PUT-DATA-THERE."""
    if not data_dir.exists():
        print("Error: PUT-DATA-THERE directory not found")
        return None

    required = ['data_distribution', 'num_rounds', 'local_epochs', 'run_id']
    valid_configs = []

    for metadata_path in sorted(data_dir.rglob('metadata.txt')):
        metadata = extract_metadata(metadata_path)
        if all(key in metadata for key in required):
            valid_configs.append({
                'data_distribution': metadata['data_distribution'],
                'num_rounds': metadata['num_rounds'],
                'local_epochs': metadata['local_epochs'],
                'run_id': metadata['run_id']
            })

    if not valid_configs:
        print("Error: Could not find valid metadata.txt files (directory may be empty)")
        return None

    first = valid_configs[0]
    distributions = sorted({cfg['data_distribution'] for cfg in valid_configs})
    run_ids = sorted({cfg['run_id'] for cfg in valid_configs})
    rounds = sorted({cfg['num_rounds'] for cfg in valid_configs})
    epochs = sorted({cfg['local_epochs'] for cfg in valid_configs})

    if len(run_ids) > 1 or len(rounds) > 1 or len(epochs) > 1:
        print("Warning: Inconsistent metadata found; using first valid config for run/round/epoch")

    return {
        'data_distribution': '-'.join(distributions),
        'num_rounds': first['num_rounds'],
        'local_epochs': first['local_epochs'],
        'run_id': first['run_id']
    }


def create_directory_name(config):
    """Create directory name from configuration (same format as first-station)."""
    return (f"{config['data_distribution']}-Round-{config['num_rounds']}-"
            f"Epoch-{config['local_epochs']}-{config['run_id']}")


def aggregate_results():
    """Main function to copy second-station outputs into final-station."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    if script_dir.name != SECOND_STATION_NAME:
        print(f"\n⚠️  Warning: script is not under expected second station '{SECOND_STATION_NAME}'")
        print(f"Current script dir: {script_dir}")

    print("\n" + "="*70)
    print("Second-Station Results Organization Script")
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

    # Create directory name using first-station format
    output_dir_name = create_directory_name(config)
    print(f"\n📁 Creating directory: {output_dir_name}")

    # Ensure final-station exists
    final_station = project_root / FINAL_STATION_NAME
    final_station.mkdir(exist_ok=True)
    print(f"✓ Ensured {FINAL_STATION_NAME} directory exists")
    
    # Create target directory
    target_dir = final_station / output_dir_name
    
    if target_dir.exists():
        print(f"\n⚠️  Directory already exists: {target_dir}")
        print("Removing existing directory...")
        shutil.rmtree(target_dir)
        print("✓ Removed existing directory")
    print("\nMoving PUT-DATA-THERE...")
    src_put_data = put_data_dir
    dst_put_data = target_dir / 'PUT-DATA-THERE'

    try:
        # Use robocopy for better Windows long-path support
        result = subprocess.run(
            ['robocopy', str(src_put_data), str(dst_put_data), '/E', '/MT:4'],
            capture_output=True,
            text=True,
            timeout=120
        )
        # robocopy return codes: 0-7 are success, 8+ are errors
        if result.returncode >= 8:
            print(f"❌ Error copying PUT-DATA-THERE: {result.stderr}")
            return False
        print(f"✓ Copied PUT-DATA-THERE")
    except Exception as e:
        print(f"❌ Error copying PUT-DATA-THERE: {e}")
        return False

    # Copy generate_charts
    print("\nMoving generate_charts...")
    src_charts = generate_charts_dir
    dst_charts = target_dir / 'generate_charts'

    try:
        # Use robocopy for better Windows long-path support
        result = subprocess.run(
            ['robocopy', str(src_charts), str(dst_charts), '/E', '/MT:4'],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode >= 8:
            print(f"❌ Error copying generate_charts: {result.stderr}")
            return False
        print(f"✓ Copied generate_charts")
    except Exception as e:
        print(f"❌ Error copying generate_charts: {e}")
        return False
    
    # Create summary file
    summary_path = target_dir / 'SUMMARY.txt'
    try:
        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("SECOND-STATION RESULTS SUMMARY\n")
            f.write("="*70 + "\n\n")
            f.write(f"Directory: {output_dir_name}\n\n")
            f.write("Configuration:\n")
            f.write(f"  Data Distribution: {config['data_distribution']}\n")
            f.write(f"  Training Rounds: {config['num_rounds']}\n")
            f.write(f"  Local Epochs: {config['local_epochs']}\n")
            f.write(f"  Run ID: {config['run_id']}\n\n")
            f.write("Contents:\n")
            f.write("  - PUT-DATA-THERE/: Raw experimental data\n")
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
    success = aggregate_results()
    exit(0 if success else 1)
