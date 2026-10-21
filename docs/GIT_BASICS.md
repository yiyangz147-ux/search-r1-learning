# 用这个项目学习 Git

Git 在本地记录代码的历史快照；GitHub 托管远程仓库。执行 `git commit` 会产生本地提交，
执行 `git push` 才会把提交发送到远程仓库。

## 四个位置

```text
工作目录            暂存区              本地提交历史             GitHub
编辑代码 --git add--> 准备本次内容 --git commit--> 保存快照 --git push--> 上传
```

暂存区决定下次提交包含哪些版本。`git add` 后又修改文件，新修改还需要再次 `git add`。
详见 Git 官方教材 [记录每次更新](https://git-scm.com/book/en/v2/Git-Basics-Recording-Changes-to-the-Repository)。

## 每天最常用的命令

先在终端进入自己克隆的 `search-r1-learning` 目录。

| 命令 | 用途 |
|---|---|
| `git status` | 查看当前分支、修改和暂存状态 |
| `git diff` | 看尚未暂存的修改 |
| `git add README.md` | 将指定文件的当前版本放入暂存区 |
| `git diff --cached` | 看下一次提交的具体内容 |
| `git commit -m "Explain the RAG workflow"` | 在本地保存一次快照 |
| `git log --oneline -5` | 查看最近五次提交 |
| `git remote -v` | 查看远程仓库地址 |
| `git push` | 上传本地提交到已设置的上游分支 |
| `git pull --ff-only` | 仅在能快进时接收远程更新；发生分叉则停止 |
| `git restore --staged README.md` | 取消该文件的暂存，保留工作目录中的修改 |

## 第一次建立仓库

下面展示命令的含义。已经存在 `.git/` 的项目可以直接使用现有仓库。
提交作者可以在当前仓库单独设置，避免影响电脑上的其他项目：

```bash
git init -b main
git config user.name "你的提交显示名"
git config user.email "你的 GitHub noreply 邮箱"
```

先创建 `.gitignore`，再明确选择要提交的文件：

```bash
git add .gitignore README.md NOTICE requirements*.txt scripts/ tests/ configs/ docs/
python3 scripts/verify_code_only.py
git diff --cached --stat
git commit -m "Add code-only RAG, SFT and GRPO workflow"
```

`git init` 建立本地仓库；它不创建 GitHub 仓库。创建好 GitHub 仓库后，连接和首次上传：

```bash
git remote add origin https://github.com/YOUR_ACCOUNT/search-r1-learning.git
git push -u origin main
```

`origin` 是远程地址的本地别名，`main` 是分支名，`-u` 为当前分支记录上游；以后可直接 `git push`。
远程设置已经存在时，先 `git remote -v` 检查，再决定是否需要修改。
官方说明见 [使用远程仓库](https://git-scm.com/book/en/v2/Git-Basics-Working-with-Remotes)。

## 做一次小练习

在已有首次提交的仓库中，给 README 增加一行自己的学习笔记，然后依次执行：

```bash
git status --short
git diff -- README.md
git add README.md
git diff --cached -- README.md
python3 scripts/verify_code_only.py
git commit -m "Add a learning note"
git log --oneline -3
git push
```

看完差异再提交。提交信息应描述本次改动，例如 `Fix answer extraction`，方便以后查找原因。

## 分支是什么

分支让你单独记录一项改动。主分支保留已确认版本，新实验从新分支开始：

```bash
git switch -c experiment/new-reward
```

在新分支中编辑、测试、提交；准备分享时首次上传该分支：

```bash
git push -u origin experiment/new-reward
```

之后可以在 GitHub 创建 Pull Request，对比和讨论改动。切换已有分支用 `git switch main`；
切换前用 `git status` 检查尚未提交的工作。

## 这个项目的上传边界

代码、配置示例、依赖和教程进入 Git。真实数据、模型、adapter、checkpoint、轨迹、
结果、日志、缓存与凭证保存在本地或独立存储中。`data/`、`work/` 等目录没有加入许可列表。

`.gitignore` 影响尚未跟踪的文件，已经提交的内容仍留在历史中。发现误暂存时，在提交前执行
`git restore --staged 路径` 并修正忽略规则。发现已提交的敏感内容时，应先停止推送并检查历史；
泄露的凭证需要撤销，删除工作目录里的文件不能撤回已公开的凭证。

## 登录和作者的区别

`user.name`／`user.email` 是提交署名；GitHub 登录或 SSH key 决定你是否有推送权限。
两者各自独立。私有仓库的拉取和推送同样需要认证。

GitHub CLI 的官方浏览器登录流程为 `gh auth login --web --git-protocol https`，
凭证由 CLI 管理，不写进项目源码。详见 [官方登录文档](https://cli.github.com/manual/gh_auth_login)。
