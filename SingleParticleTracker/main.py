import argparse
from SingleParticleTracker import SingleParticleTracker

def parse_args():
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="初始化 MyClass 的参数。")
    parser.add_argument('-w', type=float, default=3, help='w: 邻域窗口半径（用于背景去除和最大值检测），默认值为 3')
    parser.add_argument('--r_percent', type=float, default=0.1, help='r_percent 参数，默认值为 0.1')
    parser.add_argument('--sigma', type=float, default=1.0, help='sigma 参数，默认值为 1.0')
    parser.add_argument('--Ts', type=float, default=2.0, help='Ts 参数，默认值为 2.0')
    parser.add_argument('--max_step', type=float, default=1.5, help='max_step 参数，默认值为 100.0')
    parser.add_argument('--r_particle', type=float, default=3, help='r_particle 参数，默认值为 2.1')
    parser.add_argument('--dpi', type=int, default=400, help='dpi 参数，默认值为 400')
    parser.add_argument('--save_path', type=str, default='./result/', help='save_path 参数，默认值为 ./result')
    parser.add_argument('--sample_count', type=int, default=100, help='统计速率分布的采样点数量，默认值为 100')
    parser.add_argument('--limit', type=list, default=None, help="限制采样范围的列表，例如 [y_start, y_end,x_start,x_end]，默认值为 None")
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()

    # 使用命令行参数初始化 MyClass
    tracker = SingleParticleTracker(
        w=args.w,
        r_percent=args.r_percent,
        sigma=args.sigma,
        Ts=args.Ts,
        max_step=args.max_step,
        r_particle=args.r_particle
    )
    
    path = []
    for i in range(1,10001):
        path.append(f"VimbaImage_{i}.tiff")
    tracker.run(file_paths = path , limit = args.limit,dpi = args.dpi , save_path = args.save_path,sample_count = args.sample_count  )

if __name__ == "__main__":
    main()