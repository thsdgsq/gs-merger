# 从原工作目录迁移

本项目保留原人工顺序拼接的选点、Sim(3)、完整高斯变换和检查点逻辑，
将必需的数学及 PLY 实现搬进自己的 Python 包。旧工作流仍可继续使用；
本项目可以整体复制到其他位置，不需要复制原工程。

## 名称和入口对应关系

| 原工作目录中的名称 | 独立项目中的名称 |
| --- | --- |
| `multi_gs_stitch/run.sh` | 根目录 `stitch.sh` |
| `multi_gs_stitch/pipeline.py` | `src/gs_merger/pipeline.py` |
| 直接运行旧 Python 脚本 | `python -m gs_merger` 或 `gs-merger` |
| `multi_gs_stitch/config.example.json` | `configs/models.example.json` |
| `utils/gaussian_stitch.py` | `src/gs_merger/gaussian_io.py` |
| `utils/similarity.py` 中的手工控制点求解 | `src/gs_merger/similarity.py` |
| 旧原理文档 | `docs/SIM3_AND_GAUSSIANS.md` |

`--models`、`--pairs`、`--output-dir`、选点滑块和各轮终端操作保持相同。
可从现有 Python 环境直接试运行源码：

```bash
bash /absolute/path/to/gs-merger/stitch.sh \
  --python /absolute/path/to/python \
  --models /data/model1.ply /data/model2.ply /data/model3.ply \
  --pairs 10 --output-dir /data/merged_result --check
```

检查通过后移除 `--check` 执行人工拼接。安装项目后可使用相同参数调用
`gs-merger`；只有 Shell 入口解析 `--python`。

## 接着已有会话执行

已有会话不保存 Python 包名，因此在输入模型、路径、输出目录和计算参数
均保持一致时，可以由新入口继续读取已有 `session.json` 和各轮记录。
模型顺序、点数、误差阈值、内点比例和随机种子应与原会话一致；使用显式配置可避免
因漏写参数而被恢复检查拒绝。显示参数可调整，不需要重做已提交轮次。

把模型移动到新机器、改变输入路径或文件修改时间，会触发原有的签名保护。
这种情况应使用新输出目录，不能直接把旧会话当作新输入的完成记录。
会话数据与真实 PLY 不应上传到代码仓库中。

## 最终模型

最后一轮 `steps/NNN/merged.ply` 是独立的标准 3DGS 模型文件。
程序完成时直接打印其绝对路径，并写入 `final.json` 的 `merged` 字段。
没有额外的最终软链接，也不会自动压缩、减少 SH 阶数、抽稀或改写输出格式。
查看器中的加载内存限制应另行处理。
