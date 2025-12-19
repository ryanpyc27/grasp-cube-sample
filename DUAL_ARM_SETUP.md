# 双机械臂环境设置说明

## 概述

`SortCubeSO101-v1` 是一个使用两个 SO101 机器人的协作环境，两个机器人需要独立地将各自的方块移动到指定区域。

## 环境配置

### 机器人设置
- **Robot 1** (右侧): 位置 `[0.481, 0.003, 0]`
  - 负责: 红色方块
  - 目标区域: `(0.479, 0.25)`

- **Robot 2** (左侧): 位置 `[0.121, 0.003, 0]`
  - 负责: 绿色方块
  - 目标区域: `(0.121, 0.25)`

### 方块设置
- **红色方块**: 在中心区域 `(0.30, 0.25)` 附近随机生成
- **绿色方块**: 在中心区域 `(0.30, 0.25)` 附近随机生成

## 关键代码模式

### 1. 环境注册
```python
SUPPORTED_ROBOTS = [
    ("so101", "so101"),  # 两个机器人的元组
]

def __init__(self, *args, robot_uids=("so101", "so101"), **kwargs):
    super().__init__(*args, robot_uids=robot_uids, **kwargs)
```

### 2. 加载机器人
```python
def _load_agent(self, options: dict):
    super()._load_agent(
        options, 
        [sapien.Pose(p=self.robot1_position), sapien.Pose(p=self.robot2_position)]
    )
```
**注意**: 位姿必须作为列表传递！

### 3. 初始化机器人
```python
def initialize_agent(self, env_idx: torch.Tensor):
    # 为每个机器人生成独立的 qpos
    qpos1 = ...  # Robot 1 的初始关节位置
    qpos2 = ...  # Robot 2 的初始关节位置
    
    # 分别重置每个机器人
    self.agent.agents[0].reset(qpos1)
    self.agent.agents[0].robot.set_pose(sapien.Pose(self.robot1_position))
    
    self.agent.agents[1].reset(qpos2)
    self.agent.agents[1].robot.set_pose(sapien.Pose(self.robot2_position))
```

### 4. 访问机器人
在所有方法中，通过 `self.agent.agents[i]` 访问单个机器人：

```python
# 获取 TCP 位置
robot1_tcp = self.agent.agents[0].tcp_pose.p
robot2_tcp = self.agent.agents[1].tcp_pose.p

# 检查抓取状态
is_red_grasped = self.agent.agents[0].is_grasping(self.red_cube)
is_green_grasped = self.agent.agents[1].is_grasping(self.green_cube)

# 检查静止状态
is_robot1_static = self.agent.agents[0].is_static(0.2)
is_robot2_static = self.agent.agents[1].is_static(0.2)
```

## 奖励设计

奖励函数分别为每个机器人计算奖励，然后求和：

### Robot 1 (红色方块) 奖励
1. **接近奖励** (0-1): 鼓励 TCP 接近红色方块
2. **抓取奖励** (1): 成功抓住红色方块
3. **移动奖励** (0-2): 抓住后将方块移向红色目标区域
4. **分拣奖励** (2): 红色方块进入红色目标区域

### Robot 2 (绿色方块) 奖励
1. **接近奖励** (0-1): 鼓励 TCP 接近绿色方块
2. **抓取奖励** (1): 成功抓住绿色方块
3. **移动奖励** (0-2): 抓住后将方块移向绿色目标区域
4. **分拣奖励** (2): 绿色方块进入绿色目标区域

### 额外奖励
- **静止奖励** (1): 两个方块都到达目标且机器人静止
- **成功奖励** (20): 完成任务

**总奖励**: 最高 20 分

## 动作空间

### 原始动作空间
多机器人环境的原始动作空间是 Dict 类型：
```python
Dict('so101-0': Box(-1.0, 1.0, (6,), float32), 
     'so101-1': Box(-1.0, 1.0, (6,), float32))
```

### 展平后的动作空间
使用 `FlattenActionSpaceWrapper` 后，动作空间变为：
```python
Box(-1.0, 1.0, (12,), float32)
```

- 每个机器人有 6 个自由度
- 总动作维度: 12 (6 + 6)
- 动作格式: `[robot1_action (6), robot2_action (6)]`

**重要**: RL 训练时必须使用 `FlattenActionSpaceWrapper`，否则大多数 RL 算法无法处理 Dict 类型的动作空间。这个 wrapper 已经自动添加到 `rl/tdmpc2/envs/maniskill.py` 中。

## 观测空间

关键观测包括：
- `is_red_grasped`: Robot 1 是否抓住红色方块
- `is_green_grasped`: Robot 2 是否抓住绿色方块
- `robot1_tcp_pose`: Robot 1 的末端执行器位姿
- `robot2_tcp_pose`: Robot 2 的末端执行器位姿
- `red_target_pos`: 红色目标区域位置
- `green_target_pos`: 绿色目标区域位置
- `red_cube_pose`: 红色方块位姿 (state 模式)
- `green_cube_pose`: 绿色方块位姿 (state 模式)

## 测试环境

运行测试脚本验证环境：
```bash
python test_sort_cube.py
```

## 常见错误

### 1. `'MultiAgent' object has no attribute 'robot'`
**原因**: 试图直接访问 `self.agent.robot`  
**解决**: 使用 `self.agent.agents[0].robot` 或 `self.agent.agents[1].robot`

### 2. `IndexError: list index out of range`
**原因**: 在 `_load_agent` 中没有将位姿放入列表  
**解决**: 使用 `[pose1, pose2]` 而不是 `pose1, pose2`

### 3. 机器人位置重叠
**原因**: 两个机器人使用相同的位置  
**解决**: 确保 `robot1_position` 和 `robot2_position` 不同

### 4. `TypeError: 'NoneType' object is not subscriptable` (访问 action_space.shape[1])
**原因**: 多机器人环境的 action_space 是 Dict 类型，没有 shape 属性  
**解决**: 使用 `FlattenActionSpaceWrapper` 将 Dict action space 转换为 Box。已在 `rl/tdmpc2/envs/maniskill.py` 中自动添加。

## 训练配置

在使用 TD-MPC2 或其他 RL 算法时，需要注意：
- 动作维度会自动扩展为 12
- 观测空间包含两个机器人的状态
- 建议使用更长的训练时间，因为协调两个机器人更复杂

## 参考

- ManiSkill 多机器人示例: `grasp_cube/envs/tasks/table.py` (第 216-252 行)
- 单机器人环境: `grasp_cube/envs/tasks/pick_cube_so101.py`










