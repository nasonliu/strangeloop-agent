# RFC-0006：账本派生记忆图谱

## 状态

已实现 v1：SQLite 会话账本上的可重建、只读关系投影。

## 目的与边界

本 RFC 把 Yogacara 的“缘起/条件关系”只作为工程问题的启发：软件记录一个可核验结果依赖哪些观察、纠错、种子版本和实验记录。它不表示主体体验、记忆体验、固有自我或宗教结论。

`MemoryGraph` 不是第二个真相源。`cognitive_events` 的 hash-chain 是唯一权威；图谱仅由其中同一 session 的公开结构化事件重建。

## 数据模型

- `memory_graph_nodes`：不透明 event ID、固定 node type、来源 event ID、snapshot digest、created time。
- `memory_graph_edges`：固定 relation、source/target node ID、产生该边的 evidence event ID。
- `memory_graph_meta`：图谱版本和已投影的账本 head。

v1 关系仅为：`derived_from`、`contradicts`、`supports`、`represented_by`、`activates`、`supersedes`、`executes`、`result_of`、`verified_by`。

不抽取实体、不保存网页正文、URL、用户原文、cue、模型推理、embedding 或任意模型生成关系。未知事件只保留其既有 parent 的 `derived_from` 关系。

## 权限与生命周期

- 图谱不得批准、激活、强化、退役或复活种子；种子状态仍由事件账本与 standing policy 校验决定。
- 图谱不得影响 reward/TD、frontier ranking、capability/grant、quota、sleep/wake、stop、purge 或自我模型。
- 每次 ledger head 前进时，`ensure_current()` 原子替换该 session 的派生行；无新事件时不写入。
- `/graph status` 与 `/graph explain EVENT_ID` 只返回元数据和一跳关系，不返回 payload。
- 图表与账本共用同一 session SQLite 容器；确认 `/purge` 删除容器时，图谱表、WAL 和 SHM 一并消失。图谱可从仍存在的账本重建，不承诺清除备份、快照或外部副本。

## 非目标

v1 不是开放世界知识图谱、语义检索、向量数据库或自动事实判断。后续若要用图谱影响检索或调度，必须新 RFC 明确说明来源、冲突、撤销、用户控制、审计和奖励隔离。
