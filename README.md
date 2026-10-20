# gs-merger

基于人工对应地标和 Sim(3) 的多模型 3D Gaussian Splatting 拼接工具。

一个 shell 命令依次执行：读取高斯 PLY → 导出密集 RGB 中心点云 → Open3D
人工选择对应点 → 估计 Sim(3) → 查看对齐 → 变换完整高斯属性并合并 → 下一轮。

本项目可以独立安装和部署，所有运行代码都位于 `src/gs_merger/`。
只依赖 NumPy、plyfile 和 Open3D，不需要上一级工程、配准网络或训练权重。
输入和输出均为标准 Graphdeco 3DGS PLY；保留所有高斯及完整属性。
最终直接报告最后一轮 `merged.ply` 的实际路径，不创建软链接。

安装、平台要求、离线打包和 GitHub 发布见 [部署说明](INSTALL.md)。
已有命令和会话的迁移方法见 [迁移说明](docs/MIGRATION.md)。

关于 10 对点为什么能约束完整模型、Sim(3) 的 SVD 求解、高斯与球谐属性怎样变换，
见 [原理详解：从控制点到完整 3DGS 拼接](docs/SIM3_AND_GAUSSIANS.md)。

## 开源使用、作者与联系

本项目采用 [MIT License](LICENSE)。你可以按许可证直接使用、修改和分发本项目，
包括商业用途；应保留许可证要求的版权和许可声明。使用前无需联系作者，
也无需逐项申请书面授权。第三方运行库适用其各自的许可证，见
[依赖与算法参考](THIRD_PARTY_NOTICES.md)。

作者：[**thsdgsq**](https://github.com/thsdgsq)。
联系邮箱：[1609303328@qq.com](mailto:1609303328@qq.com)。
欢迎通过 [GitHub Issues](https://github.com/thsdgsq/gs-merger/issues) 反馈问题、分享应用场景或讨论合作。
提问时请附操作系统、Python/Open3D 版本、执行命令、完整报错和简要复现步骤。

如果这个工具帮助了你的研究或工程工作，欢迎在论文或项目说明中引用它。
引用元数据见 [CITATION.cff](CITATION.cff)；发布到 GitHub 后可通过仓库侧栏的
“Cite this repository”查看引用格式。引用和联系是自愿的，不额外限制 MIT 许可授予的使用权。

## 启动

推荐在 Linux 图形桌面上使用 Python 3.11，在本项目根目录安装：

```bash
conda create -n gs-merger python=3.11 pip -y
conda activate gs-merger
python -m pip install .
```

运行依赖也直接列在 [requirements.txt](requirements.txt) 中；
`pip install .` 会读取相同的依赖并安装命令入口。

在项目根目录直接指定模型顺序和每轮对应点数：

```bash
bash stitch.sh \
  --models data/model1.ply data/model2.ply \
           data/model3.ply data/model4.ply \
  --pairs 10 \
  --output-dir results/site
```

安装后也可以使用 `gs-merger`，参数与 Shell 入口相同：

```bash
gs-merger --help
python -m gs_merger --help
```

将示例模型路径替换为真实路径。可将 `--models` 后面增加到任意数量，至少两个。
带空格的文件路径必须加引号。命令行中的相对路径以当前工作目录为基准。
可从任意目录运行脚本，不要求先 `cd` 到本目录。

也可以先复制并编辑配置：

```bash
cp configs/models.example.json configs/local.json
# 编辑 configs/local.json 中的模型路径、顺序、输出目录和参数后执行：
bash stitch.sh --config configs/local.json
```

配置文件内的相对路径**以配置文件所在目录为基准**。
`models` 数组即拼接顺序；`pairs_per_step` 可以是整数（每轮相同），
也可以是长度等于模型数减一的数组，例如四个模型使用 `[6, 8, 6]`。
`--pairs 6` 可以覆盖配置中的全部轮次点数。

可先检查路径、字段、配置和依赖，检查模式不打开窗口、不写模型：

```bash
bash stitch.sh --config configs/local.json --check
```

`--check` 不验证桌面/OpenGL，也不执行完整模型的所有数值检查；实际写入前会检查
高斯属性的有限性和四元数。需要指定解释器时：

```bash
bash stitch.sh \
  --python /absolute/path/to/venv/bin/python \
  --config configs/local.json
```

## 每轮如何操作

四个模型的顺序为：

```text
第 1 轮：target=model1，source=model2 → steps/001/merged.ply（merge1）
第 2 轮：target=merge1，source=model3 → steps/002/merged.ply（merge2）
第 3 轮：target=merge2，source=model4 → steps/003/merged.ply（最终结果）
```

**始终固定第一个模型的坐标系。** 每轮只变换新加入的 source，已有 target 保持不动。
相邻模型不必全部互相重叠，但每个新模型必须与当时的累计模型有足够的可辨认共同地物。

1. 自动导出本轮 target 和 source 的密集彩色中心点云。
2. **同时打开两个窗口**：左侧 target（已有模型），右侧 source（新模型）。
   可以分别旋转、平移和缩放，来回对照；推荐左侧选 A 后立即在右侧选 A，再选 B、C……。
3. 两侧分别选够指定数量。关闭任一选点窗口，在终端输入 `y` 确认本侧（仅在内存中），
   或输入 `r` 重选。**两侧窗口都关闭且有一侧点数不足时，只输入一次 `r`，就同时
   重开 target 和 source，两侧从零选取。** 如果另一侧窗口仍开着，`r` 只重开当前侧。
4. 如果一侧已确认、另一侧仍在窗口中选取，已确认侧会以只读对照窗口重新打开。
   对照窗口标题含 `comparison only`，不修改已确认点；两侧已关闭并选满时，
   依次确认即可，不额外重开第一个模型。
5. **只有两侧都选满要求数量并确认后，才一次性保存完整点对 JSON**，然后求 Sim(3)、
   打印每对点的残差和内点标记，并关闭只读对照窗口。
6. 自动显示对齐预览：橙色是新模型，蓝色是累计模型。关闭窗口后回到终端。
7. 输入 `a` 写入本轮完整高斯合并结果，然后自动进入下一轮。

Open3D 选点窗口操作：

- `Shift + 左键`：加入一个点；顺序就是对应点顺序。
- `Shift + 右键`：撤销最后一个点。
- 窗口顶部 **Point size 滑块**或键盘 `+` / `-`：实时调整 RGB 点云大小（1～20 像素）。
- **Marker size 滑块**：独立调整彩色选点标记大小（1～30 像素），不影响点云大小。
- 已选点显示高亮颜色和深色底编号；两窗的第 N 个点颜色相同，不同点对使用不同颜色。
- 顶部 `next #N` 提示下一个编号，便于发现某次点击未选中导致两窗点数不同。
- `Undo`：撤销最后一个点；`Reset`：清空这一侧当前选点。
- `Finish`：选满后返回终端；不足指定数量时该按钮禁用。
- `Q`、`Esc` 或窗口关闭按钮：返回终端，**不会自动确认或合并**。
- 鼠标旋转、缩放和平移视角；按住 Shift 点击用于选点或撤销。

**例如 `--pairs 10` 时，任一侧不足 10 点就关闭：本侧点全部丢弃，不保存坐标，
如果另一侧窗口仍开着，输入 `r` 重新打开这一侧，从零选取。
如果两侧都关了，输入一次 `r` 同时重新打开两侧。** 另一侧开着时可以继续作为对照。
如果此时退出整个流程，未凑齐的整组都不保存；下次运行重新选择两侧。
重复坐标不加入选择，选满后不能继续添加，需先 Undo 或 Reset。
选点窗口使用独立的 OpenGL 事件循环，因此在终端确认一侧时，另一侧窗口仍可操作。
本界面基于 Open3D GUI 的
[深度拾取接口](https://github.com/isl-org/Open3D/blob/main/examples/python/visualization/mouse_and_point_coord.py)，
拾取后吸附到点云中的最近中心，保存该中心的原始 XYZ 与点击顺序，不直接保存深度反投影坐标。

### 点大小、标记和窗口尺寸

选点及只读对照窗口直接使用查看 PLY 中已有的 RGB，关闭新渲染器默认的颜色后处理
（包括色调映射、颜色分级）和多重采样，避免额外提亮、改色或混合点的颜色。
使用无光照材质，不添加亮度、对比度、Gamma 补偿，也不修改 PLY、内存点云中的 RGB
或高斯模型的颜色属性。两条滑块分别改变点云和标记覆盖的像素范围。
渲染设置参见 Open3D 的
[View 接口说明](https://www.open3d.org/html/python_api/open3d.visualization.rendering.View.html)。

**标记使用彩色点图元，大小由独立的 Marker size 滑块控制，不再使用三维球体。**
相机放大、缩小或旋转后，标记仍保持设定的屏幕像素大小；标记的微小深度偏移仅用于显示，
不改变保存的控制点坐标。每对点按编号固定颜色，例如第 1 对红、第 2 对青、第 3 对黄；
撤销、重选、只读对照时，同一编号使用相同颜色。前 10 对采用预设高亮色，更多点对
继续生成不同颜色，不循环使用同一种黄点。深色底数字放在点旁，并随视角移动；
编号是屏幕叠加元素，不加入拾取深度图，也不改写原始点云 RGB 或保存坐标。
默认点云大小 2 像素、标记大小 6 像素；两者可分别调整。设置两侧窗口的初始值，
在原命令末尾加：

```bash
--point-size 4 --marker-point-size 10
```

确认的完整点对会附带记录两种显示尺寸，重新打开只读对照窗口时复用。
`marker_point_size` / `--marker-point-size` 使用像素单位；旧的 `marker_size` /
`--marker-size` 保留解析并忽略，以兼容旧配置中使用世界坐标单位的数值。
两窗默认每个 `800 × 700`，可拖动和调整尺寸。需要更小的初始窗口时加：

```bash
--pick-window-width 640 --pick-window-height 600
```

修改代码后请先退出原来正在运行的脚本，再执行原命令。旧会话可以直接继续，
不需要删除输出目录；完整保存的点对和已完成轮次会保留。
旧版留下的单侧选点、数量不足的点对会自动丢弃，并重新打开两侧选取。

在对齐预览之后，终端支持：

| 按键 | 动作 |
| --- | --- |
| `a` | 确认本轮对齐并合并；仅在内点数量达标时可用 |
| `t` | 重选 target（已有模型）这一侧 |
| `s` | 重选 source（新模型）这一侧 |
| `b` | 两侧都重新选 |
| `v` | 再打开一次对齐预览 |
| `q` | 退出并保留已有进度 |

选点至少三对不共线，推荐 6～10 对、覆盖较大范围。共面但不共线可以使用。
选择同一物理地标的同一位置，例如同一个路沿转角或灯杆底座；平坦地面上随意选点
不能建立对应关系。至少留意几个未参与拟合的地物是否也对齐，避免只看拟合残差。

## 参数

| 配置字段 | 默认值 | 含义 |
| --- | --- | --- |
| `pairs_per_step` | `6` | 每轮选几对点，或按轮次指定数组 |
| `control_threshold` | `0.1` | 对应点最大内点误差，单位为**第一个模型的坐标单位**，不一定是米 |
| `min_inlier_ratio` | `0.8` | 至少需要 `max(3, ceil(点对数 × 比例))` 个内点才能合并 |
| `view_max_points` | `0` | 每侧选点时最多显示多少中心点；0 表示全部有效中心 |
| `preview_max_points` | `300000` | 对齐预览每侧最大点数；不影响选点精度或最终高斯数量 |
| `min_opacity` | `0.05` | 只过滤查看点云，不裁剪合并的高斯模型 |
| `point_size` | `2.0` | RGB 点云初始像素大小（1～20）；支持滑块、`+` / `-` 实时调整 |
| `marker_point_size` | `6.0` | 彩色选点标记初始像素大小（1～30）；通过 Marker size 独立调整 |
| `marker_size` | 兼容旧配置 | 新选点界面忽略；请使用 `marker_point_size` |
| `pick_window_width` | `800` | 两侧选点窗各自的初始宽度，像素 |
| `pick_window_height` | `700` | 两侧选点窗各自的初始高度，像素 |
| `seed` | `123` | 随机抽样和控制点 RANSAC 的随机种子 |

尺度未知会自动估计，不需要人工填写尺度或矩阵，不使用固定尺度的 ICP。
`control_threshold` 应与模型单位、选点误差匹配；没有通用的米制阈值。
几何支持不足或点共线时可直接重选，不会把失败候选应用到模型。

选点默认导出全部有效中心，Open3D 会为窗口保存几何副本，随着累计模型增大，
内存、空间索引和显示开销也会增大。需要限制时设置 `view_max_points: 2000000`，或运行时加
`--max-view-points 2000000`。RGB 使用 DC 项近似，并不是完整 3DGS 渲染。
无共同可见地物时，本流程也无法从孤立的模型中推断正确摆放关系。

## 输出与中断恢复

```text
results/site/
├── session.json                 # 配置、输入签名、已经完成的轮次
├── workflow.log                 # 主要操作与误差日志
├── steps/
│   ├── 001/
│   │   ├── context.json         # 本轮原始 source 和累计 target
│   │   ├── target_view.ply      # 密集 RGB 选点代理
│   │   ├── source_view.ply
│   │   ├── selections.json     # 两侧都选满并确认后的完整点对；退出后可复用
│   │   ├── correspondences.json # source/target 三维坐标及变换方向
│   │   ├── alignment.json      # Sim(3)、内点、残差与质量检查
│   │   ├── transform_source_to_target.npy
│   │   ├── transform_source_to_target.txt
│   │   ├── merged.ply          # 本轮完整高斯模型
│   │   └── step.json           # 本轮完成记录
│   ├── 002/
│   └── 003/
└── final.json                   # 最终模型位置、每个原模型到第一个模型的变换
```

每轮保留中间结果。最终模型就是最后一轮的 `steps/NNN/merged.ply`，
终端的“全部完成”直接输出这个真实文件的绝对路径；不创建 `final.ply` 软链接。
`final.json` 的 `merged` 字段也记录这个路径。旧版本已生成的软链接可以忽略。
需要在查看器中导入或单独复制模型时，使用最后一轮的真实 `merged.ply`。
仅在 `step.json` 存在且 `session.json` 记录该轮完成后，该轮输出才是已提交的结果；
未完成轮次中的矩阵和报告仅供诊断。

如果 SuperSplat 报 `Array buffer allocation failed while loading`，说明加载时
数组内存分配失败；这个错误本身不表示模型格式错误。完整合并模型可能包含几千万
个高斯、占用数 GB，导入和渲染还需要额外内存、显存。先检查 SuperSplat 版本：
[官方 3.0 更新说明](https://blog.playcanvas.com/new-in-supersplat-editor-3-0-rebuilt-on-webgpu/)
介绍了分块 GPU 存储和内存占用改进；旧版用户可先尝试新版编辑器。
若仍受内存限制，可另行导出压缩或 LOD 查看版本；原始 `merged.ply` 保留完整属性。
压缩通常包含量化误差，且文件变小不保证加载后的内存一定足够。
具体排查见 [SuperSplat 官方说明](https://developer.playcanvas.com/user-manual/supersplat/editor/troubleshooting/)。

输入 `q` 或 `Ctrl+C` 后，用**完全相同的命令**再次运行即自动恢复：
已完成轮次跳过，完整点对可以复用，仍可在预览后重选。
未选够整组的轮次不会保存选点，下次重新选择两侧。
退出码 130 表示用户退出。主动重选任一侧时会使旧整组点对失效，
另一侧仅在当前运行中保留，直到重新凑齐整组才保存。
输出目录有锁，防止两个进程同时写入同一个会话。

恢复会检查输入文件的路径、大小和修改时间，以及已经完成的输出文件签名。
更改模型、顺序、点对数量、拟合阈值等计算参数，或删除中间结果后，不会继续混用旧结果；
这时使用新的输出目录。标记大小、窗口尺寸、RGB 点大小、查看点数和透明度过滤等
显示参数可在恢复时调整，不会重做已经完成的合并，已有完整 XYZ 点对也会保留。
这不是文件内容的完整加密校验，请勿在运行过程中修改原始模型。
若写入中断，尚未提交的临时 PLY 不会替换完整结果；再次运行可能重写当前未提交轮次。
强制杀死进程留下的 `.merge_*` 临时文件不参与恢复，可在确认没有进程运行后自行清理。

合并调用本项目内置的 `gs_merger.gaussian_io.merge_gaussian_files()`，
采用分块变换、磁盘映射缓冲区和标准 PLY 写入实现。
多模型适配层只在临时目录中执行，成功后把完整 `merged.ply` 移到本轮结果路径。
保留所有高斯，使用 `minimum_opacity=0`；显示代理的过滤不作用于最终模型。

合并期间先显示全部属性校验，然后显示已有模型分块复制、新模型分块变换的数量、
高斯准备进度、吞吐量和本阶段预计剩余秒数，约每 2 秒更新，同时写入 `workflow.log`。
**高斯准备进度 100% 后还要写入 PLY**，终端会明确显示写入阶段；只有看到“合并完成”
才表示本轮完整模型已经提交。最终 PLY 写入使用 `plyfile`，显示阶段开始和结束。
临时目录为本轮目录中的 `.merge_*/`，其中包括映射缓冲区、对齐源模型和合并 PLY；
日志列出缓冲区位置。正常完成或异常退出会清理临时目录，不提交半成品，不覆盖原始模型。
完整属性合并需要临时磁盘空间，启动合并时会检查并显示预算；对齐源模型仅在临时目录中存在。
完整高斯的变换和写入由独立 Python 进程执行，该进程不加载 Open3D，数值库线程数固定为 1，
用于隔离界面和数值库的线程环境；不改父进程环境或 Gaussian 变换公式。
连续 60 秒没有进度消息时，计算进程会自动向终端和 `workflow.log` 输出调用栈，
保留现场继续等待，便于定位具体停在读取、数值运算还是写入。
`Timeout (0:01:00)!` 是这个诊断计时器的提示，不会取消写入；大文件正常写入也可能触发。
以最终完成日志和已提交结果为准。

每轮合并按块读取和写入，完整保留位置、透明度、wxyz 四元数、对数尺度、球谐系数、
法线与额外标量字段。要求全部模型的 SH 阶数、字段和类型一致，保留所有高斯，
不做接缝优化、重复高斯融合或曝光统一。累计误差和独立重建的非刚性畸变需要另行处理。
所有中间 `merged.ply` 会占磁盘，启动时显示它们的体积估算，此外还需存储 RGB 查看点云。

## 验证

```bash
bash -n stitch.sh
python -m unittest discover -s tests -t . -v
```

可选的无窗口渲染检查（需要 EGL，不读取实际模型）：

```bash
EGL_PLATFORM=surfaceless GS_MERGER_TEST_RENDERING=1 \
  python -m unittest tests.test_rgb_rendering -v
```

该检查对全部 256 级灰度和 32 组彩色 RGB 逐像素比对输入值与渲染值，
同时覆盖点大小调整，检查内存中的 RGB、XYZ 未被改写。普通测试默认跳过 EGL 检查。

测试使用小型合成高斯与模拟选点，覆盖四模型顺序、完整属性、固定参考坐标系、
误选重试、双窗口启动和退出协调、整组保存门槛、点云和标记尺寸独立调整、深度拾取、
窗口关闭重试、旧会话恢复、
输入变更保护、写入失败保护和含空格路径。
测试不会启动真实桌面窗口，也不会对用户的大模型执行拼接。

## 项目结构

```text
gs-merger/
├── stitch.sh                      # 一个命令执行多模型顺序拼接
├── pyproject.toml                 # 安装、依赖、命令入口
├── CITATION.cff                   # 作者、版本和软件引用信息
├── configs/models.example.json    # 按顺序列出全部模型
├── src/gs_merger/
│   ├── pipeline.py                # 轮次、配置、恢复和最终路径
│   ├── interaction.py             # 双窗口协调与终端确认
│   ├── point_picker.py            # RGB 点云、对应点和尺寸滑块
│   ├── picker_worker.py           # 独立选点窗口进程
│   ├── merge_worker.py            # 独立高斯变换与保存进程
│   ├── gaussian_io.py             # 标准 3DGS PLY、协方差与 SH 变换
│   └── similarity.py              # Umeyama 与控制点 RANSAC
├── tests/                         # 合成数据和模拟交互测试
├── examples/                      # 可复算的十对点示例
├── docs/SIM3_AND_GAUSSIANS.md      # 完整原理和实现介绍
├── INSTALL.md                     # 独立部署与发布
└── .github/workflows/test.yml      # Linux / Python 3.10～3.12 CI
```

只复制或上传整个本项目目录即可；真实模型和结果不随代码分发。
许可和来源见 [LICENSE](LICENSE) 与 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
