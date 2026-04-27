# 自然资源无损张量化底座 - 系统操作手册

本系统由 6 个核心模块构成，实现了海量地理空间矢量数据（`.shp`）到高维张量（`.pt`）的**铸造、存储、检索、解码与无损审计**全生命周期管理。

## 核心处理工作流 (Execution Pipeline)
为了保证底座的正常构建，首次运行请严格遵守以下流转顺序：
1. `data_builder.py` (构建字典与维度) -> 2. `data_loader.py` (大规模张量落盘) -> 3. `matrix_query_engine.py` (按需业务查询) -> 4. `vector_decoder.py` (人类可读解析)。
*(注：`lossless_auditor.py` 可在第 2 步后随时运行以核验精度)*

---

### 1. `config.py` (全局中枢配置)
* **作用**：系统的配置大脑。定义了物理特征与语义特征的初始维度、Transformer 模型架构参数（如嵌入维度、注意力头数）、以及全局的文件路径规划。它包含核心的动态维度对齐算法（`get_optimal_dim`），确保输出矩阵满足 Tensor Core 的 64 倍数硬件加速要求。
* **输入**：无独立输入，被其他脚本调用。
* **输出**：在程序运行时，可被序列化保存为 `config.json` 供全局读取。

### 2. `data_builder.py` (特征探测与词表构建)
* **作用**：数据入库前的“侦察兵”。它会遍历所有原始 `.shp` 文件，执行两项核心任务：一是收集所有的文本特征（如村名、权属单位）训练生成自然资源专属的 BPE 分词器（Tokenizer）；二是全局嗅探最长文本序列，自动计算并覆盖 `config.py` 中的维度上限，杜绝张量截断。
* **输入**：原始的 `.shp` 文件目录。
* **输出**：
    * 专属词表模型：`zrzy_tokenizer.json`
    * 自适应更新后的配置文件：`config.json`
* **运行指令**：
    ```bash
    python data_builder.py \
      --data_dir /mnt/data/yqmeng/ZRZYB/NRE_GIT_V8/raw_data/ \
      --output_dir /mnt/data/yqmeng/ZRZYB/NRE_GIT_V8/output \
      --config_path /mnt/data/yqmeng/ZRZYB/NRE_GIT_V8/output/config.json
    ```

### 3. `data_loader.py` (张量铸造兵工厂)
* **作用**：系统的“心脏”。负责吞吐海量 `.shp` 文件，将文本映射为 Token ID，将高精度浮点数利用一拆三算子（Float64 -> 3xFloat32）转化为无损张量。它采用“流式定长分块（Chunking）”技术，每攒满指定行数即封存为一个 `.pt` 文件，并巧妙地将 GID 剥离成独立索引文件以保证 AI 训练集的纯净。
* **输入**：原始 `.shp` 目录、`zrzy_tokenizer.json`、`config.json`。
* **输出**：
    * 定长张量数据桶：`cache_chunk_0.pt`, `cache_chunk_1.pt` ...
    * 伴生哈希主键桶：`cache_chunk_0_gids.pt`, `cache_chunk_1_gids.pt` ...
    * 全局物理血缘追踪表：`schema_registry.json`
* **运行指令**：
    ```bash
    python data_loader.py --config_path /mnt/data/yqmeng/ZRZYB/NRE_GIT_V8/output/config.json
    ```

### 4. `matrix_query_engine.py` (毫秒级张量检索引擎)
* **作用**：跨越物理文件边界的极速雷达。利用 `global_gid_index.pt` 构建的内存级哈希映射，能够以 O(1) 的时间复杂度瞬间定位指定 GID 或复合条件数据所在的数据桶与局部行号。支持静默 API 通信，直接把查到的张量切片导出，不需要读取任何庞大的原始文件。
* **输入**：所有底座张量文件（`.pt`）、`schema_registry.json`、查询条件或外部传入的 GID 列表。
* **输出**：包含目标数据的碎片化张量文件及其元数据（如 `xxx.pt` 和 `xxx_meta.json`），存放于指定的导出目录中。
* **运行指令** (以 API 静默传入 GID 为例)：
    ```bash
    python matrix_query_engine.py \
      --tensor_dir /mnt/data/yqmeng/ZRZYB/NRE_GIT_V8/output \
      --export_dir ./export \
      --gids "11729083,11729084"
    ```

### 5. `vector_decoder.py` (高维张量反演解码器)
* **作用**：张量的“翻译官”。读取 `matrix_query_engine.py` 导出的张量碎片，结合 Tokenizer 词表和 Schema 注册表，将冰冷的数字矩阵精准反解成人类可读的原始明细，并自动还原一拆三的数值精度，按照数据归属的 `LAYER` 整理成标准的报表。
* **输入**：引擎导出的张量碎片目录（`./export`）、`schema_registry.json`、`zrzy_tokenizer.json`。
* **输出**：完美对齐原始数据的明细报表文件（例如 CSV 格式）。
* **运行指令**：
    ```bash
    python vector_decoder.py \
      --input_dir ./export \
      --schema_path /mnt/data/yqmeng/ZRZYB/NRE_GIT_V8/output/schema_registry.json \
      --tokenizer_path /mnt/data/yqmeng/ZRZYB/NRE_GIT_V8/output/zrzy_tokenizer.json
    ```

### 6. `lossless_auditor.py` (自动化无损审计官)
* **作用**：端到端物理级精度的终极裁判。独立读取原始的物理源文件（`.shp`）和底座生成的张量系统，逐个细胞（单元格）进行数值精度和字符串文本的比对。自动出具包含千万级比对次数的闭环审计报告，自证“100%物理无损”的可靠性。
* **输入**：所有张量底层目录、`schema_registry.json`、`config.json` 以及原始 `.shp` 文件。
* **输出**：终端双向日志报告、`lossless_audit_report.log` 审计存证文件。
* **运行指令**：
    ```bash
    python lossless_auditor.py
    ```
