# 依赖与算法参考

本项目的人工选点、多模型顺序拼接、Sim(3) 求解和高斯属性变换代码，
是在实现这套工作流时新增的代码。开发期间曾位于另一个工程的工作目录中，
但没有使用该工程的网络、训练代码、预训练权重或自动特征配准实现。
本项目自己的许可位于根目录 [LICENSE](LICENSE)。

标准高斯字段、wxyz 四元数与 SH 约定参考
[Graphdeco Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)；
本项目不分发其训练器或渲染器代码。
点选界面使用 [Open3D](https://github.com/isl-org/Open3D) 的 GUI 和渲染 API。
第三方运行库通过 Python 包管理器安装，其许可由相应软件包提供。

运行依赖为 NumPy、plyfile 和 Open3D，版本范围列在
[requirements.txt](requirements.txt) 与 [pyproject.toml](pyproject.toml) 中。

人工对应点与自动特征匹配是不同的输入条件。本项目的拼接结果不能被标注为
自动配准方法的性能结果。算法及论文参考见
[原理说明](docs/SIM3_AND_GAUSSIANS.md)。
