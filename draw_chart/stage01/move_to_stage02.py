"""Copy stage01 experiments to stage02/PUT-DATA-THERE."""

import shutil
from pathlib import Path


def copy_experiments_to_stage02():
    """将 stage01 的所有实验复制到 stage02/PUT-DATA-THERE。"""
    script_dir = Path(__file__).resolve().parent  # stage01 目录
    project_root = script_dir.parent  # draw_chart 目录
    
    print("\n" + "="*70)
    print("Stage01 → Stage02: 复制所有实验")
    print("="*70)
    
    # 检查来源
    put_data_dir = script_dir / 'PUT-DATA-THERE'
    
    if not put_data_dir.exists():
        print("\n❌ 错误: PUT-DATA-THERE 目录不存在")
        return False
    
    # 获取所有实验目录
    experiments = [d for d in put_data_dir.iterdir() if d.is_dir() and d.name != '.gitkeep']
    
    if not experiments:
        print("❌ 错误: PUT-DATA-THERE 中没有实验")
        return False
    
    print(f"✓ 找到 {len(experiments)} 个实验")
    
    # 复制到 stage02
    stage02_dir = project_root / 'stage02'
    stage02_put_data = stage02_dir / 'PUT-DATA-THERE'
    
    stage02_put_data.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📋 复制实验到 stage02/PUT-DATA-THERE...")
    
    copied_count = 0
    for exp_dir in experiments:
        dst_path = stage02_put_data / exp_dir.name
        
        if dst_path.exists():
            print(f"  ⚠️  {exp_dir.name} 已存在，跳过")
            continue
        
        try:
            shutil.copytree(exp_dir, dst_path)
            print(f"  ✓ 已复制 {exp_dir.name}")
            copied_count += 1
        except Exception as e:
            print(f"  ❌ 错误复制 {exp_dir.name}: {e}")
            return False
    
    # 也复制 generate_charts
    generate_charts_dir = script_dir / 'generate_charts'
    if generate_charts_dir.exists():
        stage02_charts = stage02_dir / 'generate_charts'
        stage02_charts.mkdir(exist_ok=True)
        
        print(f"\n📊 复制图表...")
        try:
            for chart_file in generate_charts_dir.glob('*'):
                if chart_file.is_file():
                    dst_file = stage02_charts / chart_file.name
                    if not dst_file.exists():
                        shutil.copy2(chart_file, dst_file)
            print(f"  ✓ 图表已复制")
        except Exception as e:
            print(f"  ⚠️  警告: {e}")
    
    print("\n" + "="*70)
    print(f"✅ 成功! 已复制 {copied_count} 个实验到 stage02")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    success = copy_experiments_to_stage02()
    exit(0 if success else 1)
