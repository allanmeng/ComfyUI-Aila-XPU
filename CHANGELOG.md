# Changelog

## v0.1.2 (2026-05-04)

### 插件
- 节点名改为 `Aila Model Loader (XPU)` / `Aila Engine (XPU)`
- `memory_cleanup` 改为下拉选项：`persistent (不释放)` / `full_cleanup (释放显存, 清理缓存)`
- 引擎销毁后自动重建，不再报"模型未加载"
- 纯文本模式支持（不接图片即可使用）
- mode 下拉菜单增加中文说明
- AilaShared.dll 纳入 git 追踪，其他运行时 DLL 通过 Release 分发

### 引擎
- Aila v0.1.1
- 修复输出中的思维链（CoT）问题
- 新增 `/think` 后缀命令
- 修复 B580 上 subgroup size 兼容性问题

---

## v0.1.1 (2026-05-02)

### 插件
- `memory_cleanup` 功能（布尔开关）
- IMAGE 输入改为可选，支持纯文本模式
- 添加 Intel Arc 用户安装说明

### 引擎
- Aila v0.1.0
- Qwen3.5-4B / 0.8B NF4 量化推理
- bitsandbytes XPU 后端

---

## v0.1.0 (2026-05-02)

### 插件
- 首个公开发布版本
- 支持 Aila Model Loader + Aila Captioner 两个节点
- 三种 mode：prompt / caption / danbooru
- 支持单张/批量图片处理
- 支持 Intel Arc B580 (12GB)

### 引擎
- Aila v0.1.0
