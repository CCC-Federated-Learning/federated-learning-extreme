"""Move stage02 outputs to finally_result for final aggregation."""

import shutil
from pathlib import Path


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
    """从第一个 metadata.txt 提取配置信息。"""
    if not data_dir.exists():
        return None
    
    # 找到第一个 metadata.txt 文件
    for subdir in sorted(data_dir.iterdir()):
        if subdir.is_dir() and subdir.name != '.gitkeep':
            metadata_path = subdir / 'metadata.txt'
            if metadata_path.exists():
                metadata = extract_metadata(metadata_path)
                
                # 提取必需的字段
                if all(key in metadata for key in ['data_distribution', 'num_rounds', 'local_epochs', 'run_id']):
                    return {
                        'data_distribution': metadata['data_distribution'],
                        'num_rounds': metadata['num_rounds'],
                        'local_epochs': metadata['local_epochs'],
                        'run_id': metadata['run_id']
                    }
    
    return None


def create_directory_name(config):
    """Create directory name from configuration."""
    return (f"{config['data_distribution']}-Round-{config['num_rounds']}-"
            f"Epoch-{config['local_epochs']}-{config['run_id']}")


def move_to_finally_result():
    """将 stage02 的 PUT-DATA-THERE 和 generate_charts 移动到 finally_result。"""
    script_dir = Path(__file__).resolve().parent  # stage02 目录
    project_root = script_dir.parent  # draw_chart 目录
    
    print("\n" + "="*70)
    print("Stage02 → Finally_Result: 打包最终结果")
    print("="*70)
    
    # 检查来源目录
    put_data_dir = script_dir / 'PUT-DATA-THERE'
    generate_charts_dir = script_dir / 'generate_charts'
    
    if not put_data_dir.exists():
        print("\n❌ 错误: PUT-DATA-THERE 目录不存在")
        return False
    
    print(f"✓ 找到 PUT-DATA-THERE 目录")
    if generate_charts_dir.exists():
        print(f"✓ 找到 generate_charts 目录")
    
    # 从 PUT-DATA-THERE 的第一个实验获取配置
    print("\n提取实验配置...")
    config = get_experiment_config(put_data_dir)
    
    if not config:
        print("❌ 无法提取实验配置")
        return False
    
    print(f"  数据分布: {config['data_distribution']}")
    print(f"  轮数: {config['num_rounds']}")
    print(f"  本地轮数: {config['local_epochs']}")
    print(f"  运行 ID: {config['run_id']}")
    
    # 生成标准化目录名
    dir_name = create_directory_name(config)
    print(f"\n📁 在 finally_result 中创建: {dir_name}")
    
    # 创建 finally_result 目录
    finally_result_dir = project_root / 'finally_result'
    finally_result_dir.mkdir(exist_ok=True)
    
    # 创建目标目录
    target_dir = finally_result_dir / dir_name
    
    if target_dir.exists():
        print(f"\n⚠️  目录已存在: {target_dir.name}")
        response = input("要覆盖吗? (y/n): ").strip().lower()
        if response != 'y':
            print("取消。")
            return False
        shutil.rmtree(target_dir)
    
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ 创建目录: {target_dir.name}")
    
    # 复制 PUT-DATA-THERE
    print("\n📋 复制 PUT-DATA-THERE...")
    try:
        shutil.copytree(put_data_dir, target_dir / 'PUT-DATA-THERE', dirs_exist_ok=True)
        print(f"✓ 已复制 PUT-DATA-THERE")
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False
    
    # 复制 generate_charts
    if generate_charts_dir.exists():
        print("\n📊 复制 generate_charts...")
        try:
            shutil.copytree(generate_charts_dir, target_dir / 'generate_charts', dirs_exist_ok=True)
            print(f"✓ 已复制 generate_charts")
        except Exception as e:
            print(f"⚠️  警告: {e}")
    
    print("\n" + "="*70)
    print("✅ 成功! 结果已保存到:")
    print(f"   {target_dir}")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    success = move_to_finally_result()
    exit(0 if success else 1)
