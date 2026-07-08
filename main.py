import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from evolutionary_algorithm import test_encoding, run_experiments, plot_convergence_curves

if __name__ == "__main__":
    print("=== 测试编码解码 ===")
    test_encoding()
    print("\n=== 运行参数实验 ===")
    results = run_experiments()
    print("\n=== 绘制收敛曲线 ===")
    plot_convergence_curves(results, output_dir='outputs')
    print("完成！收敛曲线已保存至 outputs/convergence_curves.png")