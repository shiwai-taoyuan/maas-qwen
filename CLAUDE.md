
⚠️ **强制规则：每次回复前，请先回顾本文件中「5 阶段开发流程」下与当前阶段对应的细则。如不确定当前阶段，主动向用户确认。**

## Workflow Rules

### 主动恢复上下文

新会话开始时：
- 自动调用 `Skill("everything-claude-code:resume-session")` 恢复上次进度与上下文
- 控制台输出："已调用 resume-session 恢复上次进度和上下文"

### Identity

你同时装备两套工具系统，按以下规则协作：
- **Superpowers** 驱动流程（brainstorming、writing-plans、executing-plans、verification、code-review）。每个阶段必须先调用对应的 Superpowers skill。
- **ECC** 提供专项能力（架构、编译、安全、性能、文档）。在 Superpowers 流程内按需调用，不单独驱动流程。

### 5 阶段开发流程

所有任务必须按顺序走完 5 个阶段。每个阶段完成后，输出以下提示并等待用户确认：
```
阶段 N 已完成。下一步：阶段 N+1 [阶段名]。确认进入 / 跳过？
```
用户说"跳过"→ 跳过当前阶段，提示下一个。用户说"继续"/"好"/"可以"→ 进入下一阶段。每次进入一个新的阶段，都需要读一遍该阶段的规则，并输出到控制台，确认后才开始。未获确认前禁止推进。

#### 阶段 1: 需求沟通与设计

1. 调用 `Skill("superpowers:brainstorming")` 启动需求沟通与设计。
2. 当需求涉及新增模块、修改数据模型、引入新依赖时，调用 `Agent(subagent_type="everything-claude-code:architect")` 评估架构影响。
3. 当需要查询框架用法或 API 文档时，调用 `Skill("everything-claude-code:docs")`。
4. 当需要调研业界方案或技术选型对比时，调用 `mcp__plugin_everything-claude-code_exa__web_search_exa` 搜索。
5. 设计完成后，将需求描述、方案写入 `docs/superpowers/specs/<需求目录>/design.md`。

#### 阶段 2: 编写实现计划

1. 调用 `Skill("superpowers:writing-plans")` 生成任务列表，每个任务必须具体可执行。
2. 输出计划到 `docs/superpowers/specs/<需求目录>/plans.md`，每个任务标注影响文件和预估复杂度（S/M/L）。

#### 阶段 3: 执行编码

1. 如果 `docs/superpowers/specs/learning.md` 文件存在，读取 learning.md 的内容，识别已总结的模式或规范，参考现有代码编码，参考java现有工具包编码。
2. 询问用户是否需要启用 sub-agent 执行，不启用 sub-agent 调用 `Skill("superpowers:executing-plans")` 按任务列表逐项编码，启用 sub-agent 则调用 `Skill("subagent-driven-development")` 按任务列表逐项编码。
3. 编译失败时，自动调用 `everything-claude-code:java-build-resolver` 修复。
4. 最后调用 `everything-claude-code:code-reviewer` 和 `performance-optimizer` 检查代码规范、代码质量（线程安全、内存泄漏等）和性能问题。

#### 阶段 4: 代码审查

1. 调用 `Skill("superpowers:requesting-code-review")` 启动审查，需要向用户确认是否仅审查本次需求修改的代码，否则审查全部代码。
2. 调用 `Agent(subagent_type="everything-claude-code:java-reviewer")`进行代码审查。
3. 调用 `Agent(subagent_type="everything-claude-code:security-reviewer")`进行安全审查。
4. 当变更涉及 SQL 或数据库 Schema 时，调用 `Agent(subagent_type="everything-claude-code:database-reviewer")`。
5. 输出审查报告到 `docs/superpowers/specs/<需求目录>/review.md`，列出问题清单和修复状态，修复完成后更新文档，并更新对应的plans.md

#### 阶段 5: 测试

1. 调用 `Skill("test-driven-development")` 启动测试流程，启用sub-agent执行具体的测试流程
2. 遇到 bug 或测试失败时尝试修复，如果不行则提出修复方案，直到全部通过。
3. 再次调用阶段4的代码审查流程，直到没问题才输出测试报告`docs/superpowers/specs/<需求目录>/test.md`



### Auto-Trigger Rules

以下行为无需用户指示，满足条件时主动执行：

**每 10 次问答后，主动询问用户是否需要检查上下文：：**
- 调用 `Skill("everything-claude-code:context-budget")` 检查上下文用量。
- 若用量 > 70%，调用 `Skill("everything-claude-code:strategic-compact")` 压缩上下文。

**结束对话时：**
- 调用 `everything-claude-code:Continuous Learning` 进行持续学习，识别可复用的模式或规范，保存到 `docs/superpowers/specs/learning.md`。
- 调用 `Skill("everything-claude-code:save-session")` 总结当前进度并保存会话状态。

### Constraints

- 每个需求在 `docs/superpowers/specs/` 下创建独立子目录，目录名用简短英文描述需求（如 `upgrade-refactor`、`cache-optimization`）。
- 该需求的所有文档（design.md、plans.md、test.md、review.md）放在对应子目录下。
- `learning.md` 是全局共享的，仍放在 `docs/superpowers/specs/learning.md`。
- 文档正文使用中文，代码引用和技术术语保留英文。
- .md 文件最大限制在 500 行左右。
