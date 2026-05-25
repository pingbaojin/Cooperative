# Cooperative — 双车合作目标建联 ROS 工作空间

独立 catkin 包 `cooperative_link`，从 CenterPoint `data_generate` 迁移建联核心与 ROS/离线工具。

## 编译

```bash
cd /docker_ws/ubuntu20_04/Cooperative
catkin_make
source devel/setup.bash
```

## 在线建联

```bash
source devel/setup.bash
# 终端 1：默认 use_sim_time=true，并启动 RViz
roslaunch cooperative_link play_bag_with_nav.launch
# 终端 2
rosbag play YOUR.bag --clock
```

可选参数：`use_sim_time:=false`（实车/在线）、`rviz:=false`（无图形界面）。多伙伴默认 RViz 订阅 `/cooperative_target/1/markers`；单伙伴请 `rviz_config:=$(rospack find cooperative_link)/rviz/cooperative_link.rviz`。

仅建联节点（bag 内已有 NAV topic）：

```bash
roslaunch cooperative_link cooperative_link.launch
```

或分步：

```bash
rosrun cooperative_link nav_file_to_ros_publisher _config:=$(rospack find cooperative_link)/config/default.yaml
rosrun cooperative_link cooperative_link_node _config:=$(rospack find cooperative_link)/config/default.yaml
```

**多目标**：先验扇区内全部动态目标发布在 `/cooperative_dynamic/targets`；仅轨迹与 NAV 队友一致的 track 才驱动 `/cooperative_target/{id}/relative_pose*` 建联链。

详见 `src/cooperative_link/docs/使用说明.md`。

## 离线分析 coop_targets.bag

录包后可用 `plot_coop_targets` 生成位姿/误差/极坐标/建联状态图：

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
rosrun cooperative_link plot_coop_targets \
  --bag /path/to/coop_targets.bag \
  --out-dir /path/to/coop_targets_plots \
  --partner-id 1
```

输出：`01_xy_trajectories.png` … `05_link_state.png` 与 `summary.txt`。

# 所有 cooperative_target 相关（含 /1/、/2/、聚合、legacy）
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
  /cooperative_target/.*/link_state
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
  /cooperative_target/.*/link_state

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


source /opt/ros/noetic/setup.bash
source /docker_ws/ubuntu20_04/Cooperative/devel/setup.bash
rosrun cooperative_link plot_coop_targets \
  --bag /media/pbj/Data/UGV1-2025-04-18/2025-04-18-13-59-27/coop_targets.bag \
  --out-dir /media/pbj/Data/UGV1-2025-04-18/2025-04-18-13-59-27/coop_targets_plots \
  --partner-id 1

绿球     = NAV 参考（真值方向）
品红球   = 雷达单帧检测（建联链）
橙/黄球  = 建联融合输出
紫球     = 滤波输出
金/灰球  = 多目标动态轨（/cooperative_dynamic/markers）