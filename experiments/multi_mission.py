import subprocess
from multiprocessing import Pool

def run_program(command):
    # 运行命令
    subprocess.run(command, check=True)

if __name__ == "__main__":
    # 定义要运行的命令和参数
    commands = [
        ["python", "experiments/train_class2.py",
         '--num_of_episodes', '1',
         '--net_path', r"sumo_rl/nets/2way-single-intersection/single-intersection-2.net.xml",
         '--total_timesteps', '5000',
         '--proportion_of_saturations', '0.75,0.75,0.75,0.75',  
         '--note', '2-stage with dqn',
        '--training_fraction', '1',
        '--tripinfo',
                  '--batch_size', '256',
        '--replay_buffer_size', '200000'],   
    ]
    
    
    # 使用多进程池并行运行命令
    with Pool(processes=len(commands)) as pool:
        pool.map(run_program, commands)
        
        