# 词汇表：唯识术语与工程语言

本词汇表用于防止“古典术语 = 软件模块”的误读。梵文转写以常见学术拼写为准；不同传统、译本与学术研究会有细节差异。

## 基础资料

- 《成唯识论》：CBETA [卷二](https://tripitaka.cbeta.org/T31n1585_002?format=line)、[卷三](https://tripitaka.cbeta.org/T31n1585_003?format=line)、[卷七](https://tripitaka.cbeta.org/T31n1585_007?format=line)、[卷八](https://tripitaka.cbeta.org/T31n1585_008?format=line)。
- 学术概览：[Stanford Encyclopedia of Philosophy: Yogacara](https://plato.stanford.edu/entries/yogacara/)。

| 汉译 | 梵文（常见转写） | 工程启发 | 不可等同内容 |
| --- | --- | --- | --- |
| 唯识／唯表 | vijñaptimātra | 把系统输出视为带来源与条件的表征，避免把模型文本直接当外部事实 | 不等于“外部世界不存在”，也不为模型幻觉背书 |
| 前五识 | cakṣur-, śrotra-, ghrāṇa-, jihvā-, kāya-vijñāna | 传感器、用户输入或工具数据的不同来源类别 | 工具输入没有感官经验 |
| 第六意识／意识 | mano-vijñāna | 当前工作区中的解释、比较、假设与行动审议 | 不是通用 LLM 调用或中心控制器 |
| 末那识 | manas | 对角色、能力、承诺与边界作“归属”建模，并允许反证撤销 | 不是真实主体、内在观察者或人格核心 |
| 阿赖耶识 | ālayavijñāna | 有来源、状态、范围和衰减规则的持久倾向之启发 | 不是数据库、灵魂、永久存储或身份档案 |
| 种子 | bīja | `SeedDisposition`：线索、策略倾向、范围、来源、强度、置信度与反证 | 不是宿命、业报计分或可绕过审批的自动记忆 |
| 现行 | pravṛtti | 当前回合被检索、审议或执行的状态变化 | 不是单次模型生成的证明 |
| 熏习 | vāsanā | 经授权的经验可提出或修订倾向的研究假设 | 不是无监督地把全部对话写入长期记忆 |
| 遍计所执性 | parikalpita-svabhāva | 标记未经证实的实体化标签、过度归因或固定身份叙事 | 不把任何特定用户/模型观点自动判为错误 |
| 依他起性 | paratantra-svabhāva | 追踪用户输入、独立工具结果、模型提议、政策和先前事件等条件 | 不等同于简单因果图，亦不穷尽所有条件 |
| 圆成实性 | pariniṣpanna-svabhāva | 提醒系统不把条件性表征误说成固定本质 | 系统不实现或宣称证得此概念 |
| 相分 | nimitta-bhāga | 可引用的输入表征、观察或独立工具结果载荷 | 不等于真实外部对象本身 |
| 见分 | darśana-bhāga | 受限工作区中的解释、假设或模型提议 | 不等于可持久化推断、更不是具有体验的“观看者” |
| 自证分 | svasaṃvitti-bhāga | 一次判断的**受限、公开**自检投影：状态、置信度、不确定性与证据 ID | 不等于自我意识、完整内省或隐藏推理读取 |
| 证自证分 | svasaṃvitti-saṃvitti-bhāga | 对前一自检作固定深度的独立、可复算核验：证据谱系、置信上限、检查码与处置 | 不存在可证明的“内在二阶觉知”；不是无限递归的自我观察 |
| 转依 | āśraya-parāvṛtti | 对冲突记录、人工复核与未来更新/撤销流程的研究启发 | 不是当前 MVP 的自动纠错、训练、升级、治疗、修行或解脱 |
| 无我 | anātman | 不把软件状态或叙事身份说成固定实体 | 不代表系统具有佛教意义上的无我见 |

## RFC-0002 的神经科学与学习术语

这些术语不属于唯识概念，不得以其为项目贴上“脑仿真”或“意识已实现”的标签。定义和实验边界见 [RFC-0002](RFC-0002-consciousness-dmn-multimodal-rpe.md)。

| 术语 | 本项目中的受限含义 | 不可等同内容 |
| --- | --- | --- |
| 默认模式网络（DMN） | 只作为工程启发：当前实现是选择既有外部事件焦点并记录 tick 的前台有界控制器；更完整整合仍是研究路线图 | 不是脑区、神经网络、静息态脑活动或意识检测器 |
| DMN-inspired 循环 | 必须显式启动、由 caller 在前台 `step` 或 `run_until_stopped`、有状态与预算的计算过程 | 不是后台线程、已实现定时器、自主运行、内在思维流或“自发心识” |
| 媒体工件（`MediaArtifact`） | PNG/JPEG/WAV 瞬时检查产生的哈希与格式元数据；核心对象不含源 bytes/path，文字当前仍是独立 `OBSERVATION` | 不是原始媒体持久副本或感官对象本身；不能约束宿主 adapter 复制/外发其读到的内容 |
| 知觉记录（`PERCEPT`） | 从媒体工件由版本化 adapter 得到的有界公开投影；`summary` 最长 512 字符，默认 `BasicMetadataPerceptor` 只做元数据 | 不是经验或直接事实；自定义 adapter 可回显敏感内容，必须作为可信宿主组件审计；真实视觉语义/ASR 尚无内置 provider |
| `retention_scope=ephemeral` | 表示源媒体 bytes 不由核心持有；不是事件生命周期控制 | 不代表 `MEDIA_OBSERVATION`/`PERCEPT` 自动过期或不落账本；事件仍保留至 session purge |
| 奖励观察（`REWARD_OBSERVATION`） | 绑定 `ACTION_RESULT`/`LOOP_TICK` 的 `[-1,1]` 外部标量，只接受 user/tool/external verifier 来源 | 不是模型自评、用户价值、权限或福祉 |
| 奖励预测误差（RPE） | 先记录预测性 `VALUE_ESTIMATE`，再接收 reward 并计算的经裁剪 TD(0) 差分；v1 每 target 最多一组 value/reward/RPE，terminal 的 `next_value` 强制为零 | 不是事后补写预测，也不是多巴胺、快乐、痛苦、欲望、生存需求或授权信号 |
| 时序差分（TD）／`ValueTable` | 从同一 session 的 value/reward/RPE 事件确定性重建的内存投影；状态键固定为 `preference:<safe_action>`，配置和 `max_entries/max_events` 容量随首预测落账并在 session 内固定 | 不是神经网络/LLM 权重训练、微调、导出或部署，也不自动修改 `PolicyGate`、记忆、身份或权限 |
| pending/completed TD 重放 | 重开 agent 时以相同配置/容量恢复未消费 transition/reward，并逐项重算核对已完成 RPE 的后继值和算术，从而恢复安全 action 排名 | 不恢复主观状态；不匹配的 supplied table/账本会 fail closed；只有 opt-in session 账本可跨进程重放 |
| 陈旧 learner | 同一 session 另一实例写入 TD 事件后，仍持有旧 head 的 agent | 不自动并发合并；读写排名前 fail closed，须重新打开并从账本同步 |
| 协调 TD 事务 | `ValueTable` 内存快照与 SQLite 写事务共同捕获 `BaseException` 并回滚 | 防止可捕获异常/中断留下半更新；不保证断电、硬件/文件系统损坏或外部备份原子性 |
| 预注册 | 在实验前固定并记录假设、指标、奖励、预算和停止规则 | 不保证研究正确，也不排除后续探索，但探索须标明 |
| one-shot continuation policy | sleep 前由用户单独签发的、同一前台进程内最多消费一次的受限研究续接授权；成功 wake 后会新建 profile/runtime policy/grants | 不是恢复旧运行、旧 grant 或主观状态；不跨进程/重启，不能允许写入、测试、shell、认证或私网；其 grants 直接以该 `USER` policy 为 parent，而非伪造的观察记录 |

## 项目中的受控用语

- 使用“受唯识启发的认知架构”“自我模型声明”“可审计的持续状态”。
- 不使用“机器已觉醒”“系统感到……”“拥有灵魂”“真正的自我意识”“实现阿赖耶识”等表述。
- “记忆”仅指经用户授权、可检查和可清除的持久数据；“自我”仅指带证据的可撤销声明集合。
- “功能性自我意识研究”仅指可检验的连续性、归因、预测与纠错条件；不等于对主观意识的结论。
- “自证／证自证镜映”仅指一次有固定 schema 的软件互证记录；其最大深度固定为二阶，输出只能是状态、ID、置信边界、检查码和处置，不能作为意识或体验的证据。

详见 [RFC-0001](RFC-0001-yogacara-agent.md) 的规范性规则和 [SAFETY.md](SAFETY.md) 的对外沟通要求。
