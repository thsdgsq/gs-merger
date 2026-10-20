# 安装、部署与发布

## 支持范围

面向有图形桌面的 Linux，Python 3.10～3.12，推荐 Python 3.11。
运行依赖为 NumPy、plyfile 和 Open3D 0.18/0.19。
原生窗口需要可用的 OpenGL 和显示会话；对应点交互需要终端输入。
当前工作流使用 POSIX 文件锁，不支持原生 Windows。
macOS 和远程桌面环境尚未实际验证。

## 在新机器安装

把整个 `gs-merger` 目录或 GitHub 仓库放到新机器上，在项目根目录执行：

```bash
conda create -n gs-merger python=3.11 pip -y
conda activate gs-merger
python -m pip install .
gs-merger --help
```

`requirements.txt` 直接列出三项运行依赖及版本范围。若只安装依赖，
随后通过源码的 Shell 入口运行，可以执行：

```bash
python -m pip install -r requirements.txt
bash stitch.sh --help
```

`python -m pip install .` 会同时安装这些依赖和 `gs-merger` 命令，
因此完整安装时不必先单独安装 requirements。

也可以使用系统 Python 3.10～3.12 创建虚拟环境：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

已装依赖且直接运行源码时可使用 `bash stitch.sh`。此入口会定位本项目自己的
`src/`，因此从其他工作目录启动、项目路径带空格或中文也可使用。
通过 `--python /绝对路径/python` 或 `PYTHON` 环境变量指定解释器。
安装后的 `gs-merger` 命令绑定安装时的解释器，不使用 `--python` 参数。

Ubuntu 如果提示缺少 GL/EGL 动态库，可安装系统依赖：

```bash
sudo apt-get update
sudo apt-get install libgl1 libegl1 libgomp1
```

无需安装 BUFFER-X、PyTorch、CUDA 工具链或下载配准权重。
Open3D 可以通过桌面 GPU 加速窗口显示；这与 CUDA 配准没有关系。
终端中先执行 `--check` 检查依赖、输入路径和 PLY 字段。
检查模式不会验证图形驱动，也不会打开窗口或运行完整拼接。
纯无显示服务器只能做这些检查；人工选点需要连接可用的图形桌面。

## 开发与测试

```bash
python -m pip install -e '.[dev]'
bash -n stitch.sh
ruff check src tests
python -m unittest discover -s tests -t . -v
```

测试使用合成 PLY、模拟窗口和小规模子进程，不运行真实桌面选点或用户的大模型。
可选的 EGL 离屏渲染验证见 [使用说明](README.md#验证)。

## 打包部署

在安装了 setuptools 77 及以上版本和 wheel 的构建环境中，可以离线构建工程自身：

```bash
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir dist
```

新机器可安装这个 wheel，pip 将解析其声明的运行依赖：

```bash
python -m pip install dist/gs_merger-0.1.0-py3-none-any.whl
```

如需完全离线部署，在相同操作系统、架构和 Python 小版本的联网环境中下载
项目 wheel 及全部传递依赖，再复制 `wheelhouse/` 到目标机器：

```bash
python -m pip download dist/gs_merger-0.1.0-py3-none-any.whl --dest wheelhouse
python -m pip install --no-index --find-links wheelhouse gs-merger
```

Open3D 等库包含平台相关二进制文件，不能把不同平台的 wheelhouse 直接混用。
打包配置遵循 [setuptools 的 pyproject 配置说明](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html)。

## 上传到自己的 GitHub 仓库

本项目按 MIT 许可公开发布，使用者无需先取得单独授权。创建 GitHub 仓库时选择
Public，并使用本项目已经提供的 `LICENSE`。

公开作者署名为 [`thsdgsq`](https://github.com/thsdgsq)，联系邮箱为
[1609303328@qq.com](mailto:1609303328@qq.com)。作者、版本、MIT 许可和仓库地址已在
`LICENSE`、`CITATION.cff`、`pyproject.toml` 及 README 中保持一致。
`CITATION.cff` 使用 GitHub 署名作为引用作者名。

目标仓库地址为 [https://github.com/thsdgsq/gs-merger](https://github.com/thsdgsq/gs-merger)。如果仓库尚未创建，先在 `thsdgsq` 账户下
创建名为 `gs-merger` 的公开空仓库，然后上传本项目。用户无需先发邮件申请使用许可；
联系、反馈和引用均为自愿。

只上传本项目目录内的文件，不需要上传上一级 BUFFER-X 工程。
项目不附带真实 PLY、大模型、配准权重或运行输出，`.gitignore` 已排除这些内容。
README、配置示例和文档使用相对路径，部署时编辑配置里的本地模型路径即可。

在你把项目目录复制到独立位置、并创建自己的空 GitHub 仓库之后，可执行：

```bash
git init
git add .
git commit -m "Initial standalone landmark-based 3DGS merger"
git branch -M main
git remote add origin https://github.com/thsdgsq/gs-merger.git
git push -u origin main
```

这些 Git 命令在独立的 `gs-merger` 项目根目录中执行。已有本地仓库时继续使用现有
Git 配置；上方地址是本项目的目标远程仓库。
项目自身的 MIT 许可位于根目录 `LICENSE`，运行库与算法参考见
`THIRD_PARTY_NOTICES.md`。
