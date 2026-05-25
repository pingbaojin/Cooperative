# Cooperative

双车合作目标建联 ROS1 catkin 工作空间。核心包 `cooperative_link` 从 CenterPoint `data_generate` 迁移：雷达动态目标识别、队友轨迹匹配、建联 FSM 与 ROS/离线工具。

## 功能特性

- **先验门控**：双车 NAV 瞬时几何 → 雷达扇区内的动态点云聚类与关联
- **多目标**：先验扇区内全部动态簇发布 `/cooperative_dynamic/*`；仅与队友 NAV 轨迹对齐的 track 进入建联链
- **建联 FSM**：`locking` → `locked`，融合 NAV 与雷达检测，短时丢失可速度外推
- **滤波**：CV 卡尔曼 + 后处理（观测来自检测，NAV 仅作对齐评估）
- **多伙伴**：按 `link_target_id` 发布 `/cooperative_target/{id}/*`

坐标系：载机雷达系（FLU），`x` 前、`y` 左、`z` 上。

## 环境要求

| 依赖 | 说明 |
|------|------|
| ROS1 Noetic | `rospy`、`sensor_msgs`、`geometry_msgs`、`visualization_msgs` |
| Python 3 | numpy、pyyaml、matplotlib（离线绘图） |
| 可选 sklearn | DBSCAN 聚类；无则自动网格聚类 |
| 可选 open3d | 仅离线 `replay_dynamic_assoc` 可视化 |

## 目录结构

```text
Cooperative/                    # catkin 工作空间根目录
├── src/cooperative_link/       # ROS 包
│   ├── config/default.yaml     # 主配置
│   ├── launch/                 # cooperative_link.launch, play_bag_with_nav.launch
│   ├── rviz/                   # cooperative_link.rviz / cooperative_link_multi.rviz
│   ├── docs/使用说明.md         # 详细文档
│   └── matlab/                 # 参考算法脚本
├── build/  devel/              # catkin 产物（已 .gitignore）
└── README.md
```

## 编译

```bash
cd /docker_ws/ubuntu20_04/Cooperative
catkin_make
source devel/setup.bash
```

## 快速开始

### 推荐：两终端播 bag

**终端 1** — launch（默认 `use_sim_time=true`，含 RViz）：

```bash
source devel/setup.bash
roslaunch cooperative_link play_bag_with_nav.launch
```

**终端 2** — 播放 bag（须带仿真时钟）：

```bash
rosbag play YOUR.bag --clock
```

常用 launch 参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `use_sim_time` | `true` | bag 回放；实车/在线设为 `false` |
| `rviz` | `true` | 无图形界面设为 `false` |
| `rviz_config` | 多伙伴 rviz | 单伙伴：`$(rospack find cooperative_link)/rviz/cooperative_link.rviz` |
| `config` | `default.yaml` | 节点与话题配置 |

### 仅建联节点（bag 内已有 NAV topic）

```bash
roslaunch cooperative_link cooperative_link.launch
```

### 分步启动

```bash
rosrun cooperative_link nav_file_to_ros_publisher _config:=$(rospack find cooperative_link)/config/default.yaml
rosrun cooperative_link cooperative_link_node _config:=$(rospack find cooperative_link)/config/default.yaml
```

配置文件路径：`src/cooperative_link/config/default.yaml`。

## 主要话题

| 话题 | 说明 |
|------|------|
| `/cooperative_dynamic/targets` | 先验扇区内动态目标（`DynamicTargetArray`） |
| `/cooperative_dynamic/markers` | 多目标 KF 轨迹 RViz（稳定轨、带 lifetime） |
| `/cooperative_targets` | 合作目标注册表（数量、伙伴 ID） |
| `/cooperative_target/{id}/relative_pose*` | 建联链：NAV / 检测 / 融合 / 滤波 |
| `/cooperative_target/{id}/link_state` | FSM 状态 |
| `/cooperative_link/state` | 节点级状态 |

多伙伴模式下扁平 legacy 话题（如 `/cooperative_target/relative_pose`）默认不发布；详见 [`使用说明.md`](src/cooperative_link/docs/使用说明.md)。

## RViz 标记颜色

| 颜色 | 含义 | 来源 |
|------|------|------|
| 绿 + 绿线 | NAV 参考 | 建联链 `coop_nav` |
| 品红 | 雷达单帧检测 | 建联链 `coop_det` |
| 深橙 / 金黄 | FSM 融合输出 | 建联链 `coop_fused`（locking / locked） |
| 紫 | 滤波输出 | 建联链 `coop_filtered` |
| 金 / 灰 / 灰蓝 | 多目标动态轨 | `/cooperative_dynamic/markers`（队友 / 其它 / coasted 预测） |

不要在 RViz 中用 **Pose** 显示订阅 `relative_pose_*`（会显示坐标轴）；数值请用 `rostopic echo`。

## 离线分析

录包后使用 `plot_coop_targets` 生成轨迹、误差、极坐标与建联状态图：

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
rosrun cooperative_link plot_coop_targets \
  --bag /path/to/coop_targets.bag \
  --out-dir /path/to/coop_targets_plots \
  --partner-id 1
```

输出：`01_xy_trajectories.png` … `05_link_state.png` 与 `summary.txt`。

### 录制 coop_targets.bag

```bash
rosbag record --regex -O coop_targets.bag \
  /cooperative_targets \
  /cooperative_target/relative_pose \
  /cooperative_target/relative_pose_nav \
  /cooperative_target/relative_pose_det \
  /cooperative_target/relative_pose_filtered \
  /cooperative_target/relative_pose_path \
  /cooperative_target/relative_polar \
  /cooperative_target/markers \
  /cooperative_link/state \
  /cooperative_target/.*/relative_pose \
  /cooperative_target/.*/relative_pose_nav \
  /cooperative_target/.*/relative_pose_det \
  /cooperative_target/.*/relative_pose_filtered \
  /cooperative_target/.*/relative_pose_path \
  /cooperative_target/.*/relative_polar \
  /cooperative_target/.*/link_state \
  /cooperative_target/.*/markers
```

多伙伴场景可精简为：

```bash
rosbag record --regex -O coop_targets.bag \
  /cooperative_targets \
  /cooperative_target/.*/relative_pose_nav \
  /cooperative_target/.*/relative_pose_det \
  /cooperative_target/.*/relative_pose \
  /cooperative_target/.*/relative_pose_filtered \
  /cooperative_target/.*/relative_pose_path \
  /cooperative_target/.*/relative_polar \
  /cooperative_target/.*/link_state \
  /cooperative_target/.*/markers
```

## 文档

- **详细使用说明**：[src/cooperative_link/docs/使用说明.md](src/cooperative_link/docs/使用说明.md)（话题表、配置项、FAQ、离线脚本）
- **MATLAB 参考**：`src/cooperative_link/matlab/`

## 许可证

`cooperative_link` 包声明为 MIT（见 `package.xml`）。
