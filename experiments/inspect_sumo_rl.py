import sumo_rl
import inspect

# 获取 SumoEnvironment 类
SumoEnvironment = sumo_rl.SumoEnvironment

# 打印 SumoEnvironment 类的文件路径
print(inspect.getfile(SumoEnvironment))
