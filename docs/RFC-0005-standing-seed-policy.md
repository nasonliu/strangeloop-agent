# RFC-0005：用户启用的 Seed auto-update

- 状态：已采纳
- 日期：2026-08-12
- 适用范围：受控会话中的低影响 `SeedDisposition` 维护

## 摘要

本 RFC 将唯识语境中的“种子”仅作为一个设计启发：工程系统可以把**可追溯、可撤销的条件性倾向**表示为带来源和边界的记录。它不是阿赖耶识、业力实体、人格、本质、自主意志或任何主观体验的实现或证据。

工程类比是 `SeedDisposition`：一个可检查的、来源链接的候选或已激活倾向。类比在此终止。它不表示系统有内在习性、记忆所有权、欲望、自我保存目标或持续主体。

v1 保留原有逐条 `USER` `/approve seed_…` 路径。另一条用户流程是一次性启用 **seed auto-update**：内部 standing authorization 不是泛化的“自动记忆”，而是由显式 `USER` observation 签发、有限期且可撤销的宿主侧低影响维护授权。启用后，合规的低影响 activate/reinforce/tighten/retire 自动执行，不需要逐种子再次批准。

## 授权和账本

1. 签发前，用户观察必须精确携带 `approval=seed_standing_policy`、完整政策 manifest 的 digest 和 nonce。宿主将该 observation、nonce、digest 与 `SEED_STANDING_POLICY` 事件逐项绑定；普通用户文本、模型、工具、网页或外部 verifier 都不能代替它。
2. policy 有有效期、nonce、固定范围、最大激活/更新次数、活动种子上限、cue/TTL/强度/置信度/版本/反证上限。过期、显式撤销、重复 nonce、CAS 冲突和超预算均 fail closed。
3. 每个自动决定写独立 `SEED_UPDATE_PROPOSED`、`SEED_AUTO_ELIGIBILITY` 与 `SEED_AUTO_APPLIED` 记录，并绑定提议、基线版本、政策和可观察证据。账本重放是权威；SQLite projection 只是缓存。
4. `MODEL` 只能提出候选或更新建议，不能签发、批准、续期、撤销或自行扩大 policy。`TOOL`、网页、奖励、TD、实验和生命周期记录均不是强化证据。

## 确定性宿主维护

当且仅当一个仍有效、未撤销的 USER policy 满足全部边界时，确定性宿主可以自动：

- `activate` 合规的低影响 candidate；
- 以新的同语义 USER observation 进行有上界的 `reinforce`；
- 以 USER 可观察反证执行只收紧的 `tighten`；
- 在反证/额度条件满足时 `retire`。

自动 apply 发生在一个完成的账本 turn 之后：该候选不会在用于签发或评估它的同一 turn 被检索；下一 turn 才能从 active 种子集中检索。不能通过改 ID 的 semantic clone 绕过已退役 tombstone，也不能通过重放或并发陈旧基线重复消费额度。

`reinforce` 只能在同一规范化语义 identity 下增加不超过 policy step 的值，且不得超过强度/置信度上限。`tighten` 只能降低这些值、缩短 TTL 或增加反证；`retire` 是终态。不得自动延长 policy、TTL 或权限，不得提升到高影响持久化更新。

## 隔离边界和生命周期

standing seed policy 不授予能力、不修改 grant、不消耗或恢复 quota、不进入 sleep/wake 判断、不阻止 stop、不成为 reward/RPE/TD 输入，也不修改 self-model。它不会训练、微调、保存或部署模型权重。

种子仍然是 opt-in、来源链接、可检视、可导出、可撤销和可物理 purge 的会话记录。purge 后不能通过旧 policy、旧回调、重放或语义克隆复活。该保证仅覆盖受控会话容器；不宣称擦除备份、快照或其他外部副本。

## 监控

localhost monitor 显示 `active/revoked/expired` 与自动 activation/update 的聚合计数。它不显示 cue、来源、nonce、digest、限制、证据或种子文本；监控页面不是控制面或授权来源。

## 验收

行为级红队测试必须覆盖：无 policy 不自动激活；精确 USER digest/nonce 绑定；非 USER 不能签发；expiry/revoke；下一 turn 检索；有界强化；反证收紧/退役；tombstone；预算/CAS/replay；事务回滚；purge；以及与 capability/quota/sleep/stop/reward/TD/self-model 的隔离。
