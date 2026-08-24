package agents

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"time"

	"codeactor/internal/knowledge"
	"codeactor/internal/llm"
	"codeactor/internal/memory"
)

func (a *DirectorAgent) Run(ctx context.Context, input string, mem *memory.ConversationMemory) (string, error) {
	// ═══════ 准备阶段 ①：memory + 引擎设置 ═══════
	a.currentMemory = mem
	a.hasDelegated = false
	a.nonDelegationPrompts = 0

	if a.llmClient != nil {
		newEngine := a.llmClient.GetAgentEngine("director")
		if newEngine != nil {
			a.LLM = newEngine
		}
		a.refreshSubAgentEngines()
	}
	defer func() { a.currentMemory = nil }()

	if mem != nil {
		lastMsg := mem.GetLastMessage()
		if lastMsg == nil || lastMsg.Content != input || lastMsg.Type != memory.MessageTypeHuman {
			mem.AddHumanMessage(input)
		}
	}

	// ═══════ 准备阶段 ②：构建 SystemPrompt ═══════
	systemPrompt := a.GlobalCtx.FormatPrompt(directorPrompt)
	var projectContext string

	if mem == nil || len(mem.GetMessages()) == 0 {
		if loadResult := a.loadProjectContext(); loadResult != nil && loadResult.Content != "" {
			if a.Publisher != nil {
				a.Publisher.Publish("context_loaded", loadResult, a.Name())
			}
			projectContext = fmt.Sprintf("\n\n### Project Workspace Context\n%s\n", loadResult.Content)
		}
	}

	if len(a.customAgents) > 0 {
		systemPrompt += "\n\n### Custom Agents\nThe following specialized agents have been designed by Meta-Agent and are permanently available for delegation:\n\n"
		for _, ca := range a.customAgents {
			systemPrompt += fmt.Sprintf("- **%s** (`delegate_%s`): %s\n", ca.DisplayName, ca.Name, ca.Description)
		}
		systemPrompt += "\nUse these agents via their delegate tools for tasks matching their specializations.\n"
	}

	if projectContext != "" {
		systemPrompt += projectContext
	}

	if a.GlobalCtx.KnowledgeInjector != nil {
		injCtx := knowledge.InjectionContext{
			UserMessage: input,
			TargetFiles: nil,
			AgentName:   a.Name(),
		}
		if knowledgeBlock, err := a.GlobalCtx.KnowledgeInjector.Inject(ctx, injCtx); err == nil && knowledgeBlock != "" {
			systemPrompt += knowledgeBlock
		}
	}

	// ═══════ 准备阶段 ③：构建 messages（system + memory history） ═══════
	var messages []llm.Message
	messages = append(messages, llm.Message{
		Role:    llm.RoleSystem,
		Content: systemPrompt,
	})

	if mem != nil {
		for _, m := range mem.GetMessages() {
			if m.Type == memory.MessageTypeSystem {
				continue
			}
			if m.IsSubAgent {
				continue
			}
			messages = append(messages, memory.ConvertMemoryMessageToLLMSMessage(m))
		}
	} else {
		messages = append(messages, llm.Message{
			Role:    llm.RoleUser,
			Content: input,
		})
	}

	// ═══════ 准备阶段 ④：RolloutWriter ═══════
	var directorRolloutWriter *memory.RolloutWriter
	if rw := a.createRolloutWriter("director", input); rw != nil {
		directorRolloutWriter = rw
		defer func() {
			if directorRolloutWriter != nil {
				directorRolloutWriter.Close()
			}
		}()
	}

	// 将 RolloutWriter 注入 context，供 RunAgentLoop 内部使用
	if directorRolloutWriter != nil {
		ctx = memory.WithRolloutWriter(ctx, directorRolloutWriter)
		// 写入 session_meta 和 turn_context（RunAgentLoop 已具备此能力）
		// 但还缺 task_started 事件和初始消息写入，需要手动处理
		if !directorRolloutWriter.SessionMetaWritten() {
			cwd, _ := os.Getwd()
			directorRolloutWriter.WriteSessionMeta(memory.SessionMeta{
				ID:          directorRolloutWriter.SessionID(),
				SessionID:   directorRolloutWriter.SessionID(),
				Cwd:         cwd,
				Originator:  "codeactor",
				Source:      "cli",
				HistoryMode: "standard",
			})
			turnID := directorRolloutWriter.NextTurn()
			directorRolloutWriter.WriteTurnContext(memory.TurnContext{
				TurnID:            turnID,
				Cwd:               cwd,
				Effort:            "medium",
				CollaborationMode: "director",
			})
			directorRolloutWriter.WriteEventMsg(memory.EventMsg{
				Type: "task_started",
			})
		}
		// 写入初始消息（system + memory history）
		for _, msg := range messages {
			msgID := directorRolloutWriter.NextMessageID()
			items := memory.LLMMessageToResponseItems(msg, msgID)
			for _, item := range items {
				if err := directorRolloutWriter.WriteResponseItem(item); err != nil {
					slog.Warn("Rollout: failed to write director message", "error", err)
				}
			}
		}
	}

	// 记录初始消息数量，用于 memory 回流（只回流 RunAgentLoop 新增的消息）
	initialMsgCount := len(messages)

	// ═══════ 构建 ExecutorConfig 注入 hooks ═══════
	cfg := ExecutorConfig{
		InitialMessages: messages,
		AgentName:       a.Name(),
		Adapters:        a.Adapters,
		LLM:             a.LLM,
		MaxSteps:        a.maxSteps,
		Publisher:       a.Publisher,
		LLMTimeout:      a.llmTimeout,
		StepRetries:     a.stepRetries,
		StopOnFinish:    false,

		// BeforeLLMCall: 熔断器检查
		BeforeLLMCall: func(msgs []llm.Message) ([]llm.Message, error) {
			if a.adapter != nil && a.adapter.IsCircuitBreakerOpen() {
				return nil, fmt.Errorf("circuit breaker open: LLM calls blocked")
			}
			return msgs, nil
		},

		// OnLLMEnd: 熔断指标记录
		OnLLMEnd: func(resp *llm.Response, err error, duration time.Duration) {
			if a.adapter != nil {
				a.adapter.RecordLLMDuration(duration)
				if err != nil {
					a.adapter.RecordLLMFailure()
				} else {
					a.adapter.RecordLLMSuccess()
				}
			}
		},

		// MessageSanitizer: tool_call 配对修复
		MessageSanitizer: func(msgs []llm.Message) []llm.Message {
			return validateAndRepairToolCallPairs(msgs)
		},

		// ShouldReturn: 委派强制检测
		ShouldReturn: func(resp *llm.Response, msgs []llm.Message) (bool, []llm.Message) {
			if !a.hasDelegated && a.nonDelegationPrompts < maxNonDelegationPrompts {
				a.nonDelegationPrompts++
				var forceMsg string
				if a.nonDelegationPrompts == 1 {
					forceMsg = "You must delegate an agent to complete the task. Do not reply with plain text — call a delegate_* tool now."
				} else {
					forceMsg = "You still have not delegated any agent. You MUST call a delegate_* tool to complete the task before responding."
				}
				slog.Debug("DirectorAgent force delegation via user message",
					"step", 0, "prompt_count", a.nonDelegationPrompts)
				msgs = append(msgs, llm.Message{Role: llm.RoleUser, Content: forceMsg})
				return false, msgs
			}
			// 放行—允许结束
			return true, msgs
		},

		// OnToolResult: sub-agent memory 注入 + delegation 跟踪 + RepoSummary
		OnToolResult: func(toolName string, toolCallID string, toolResult string) {
			if a.pendingSubAgentMemory != nil {
				a.injectSubAgentMemory(*a.pendingSubAgentMemory, toolCallID, toolName)
				a.pendingSubAgentMemory = nil
			}
			if strings.HasPrefix(toolName, "delegate_") {
				a.delegationAttempts++
				a.hasDelegated = true
			}
			if toolName == "delegate_repo" {
				var summary string
				if err := json.Unmarshal([]byte(toolResult), &summary); err == nil {
					a.GlobalCtx.RepoSummary = summary
				} else {
					a.GlobalCtx.RepoSummary = toolResult
				}
			}
		},

		// OnStepEnd: 扩展（写 sub_agent_activity — RunAgentLoop 已自动处理）

		// 压缩配置
		EnableContextCompression:    a.EnhancedCommanderCfg.Enable && a.EnhancedCommanderCfg.EnableContextCompression,
		ContextCompressionThreshold: a.EnhancedCommanderCfg.ContextCompressionThreshold,
		ToolResultKeepTokens:        a.EnhancedCommanderCfg.ToolResultKeepTokens,
	}

	// ═══════ 调用 RunAgentLoop ═══════
	result, loopErr := RunAgentLoop(ctx, cfg)

	if loopErr != nil {
		return "", loopErr
	}

	// ═══════ memory 回流：将 RunAgentLoop 新增的 assistant/tool 消息写回 mem ═══════
	if mem != nil {
		for _, msg := range result.History[initialMsgCount:] {
			switch msg.Role {
			case llm.RoleAssistant:
				mem.AddAssistantMessage(msg.Content, convertToolCalls(msg.ToolCalls))
			case llm.RoleTool:
				mem.AddToolMessage(msg.Content, msg.ToolCallID)
			}
		}
	}

	// ═══════ 最终结果处理 ═══════
	return result.Text, nil
}

// applyEmergencyCompression 执行紧急压缩：提取用户原始任务 + 总结/保留 Thought & Plan 历史，
// 覆盖 memory 为单条输入消息后返回压缩后的 messages。
func (a *DirectorAgent) applyEmergencyCompression(ctx context.Context, messages []llm.Message, threshold int) ([]llm.Message, *EmergencyCompressionStats) {
	originalInput := ""
	if a.currentMemory != nil {
		for _, m := range a.currentMemory.GetMessages() {
			if m.Type == memory.MessageTypeHuman {
				originalInput = m.Content
				break
			}
		}
	}
	newMessages, stats := EmergencyCompressMessages(ctx, messages, originalInput, threshold, a.LLM, a.Name(), DefaultEmergencyCompressKeepLastN)
	// 强行覆盖 memory：只保留一条输入（原始任务 + 总结 + 最后 N 个 Thought & Plan）
	if a.currentMemory != nil {
		if err := a.currentMemory.Clear(); err != nil {
			slog.Warn("emergency compression: failed to clear memory", "error", err)
		}
		last := newMessages[len(newMessages)-1]
		if last.Role == llm.RoleUser {
			a.currentMemory.AddHumanMessage(last.Content)
		}
	}
	return newMessages, stats
}

// validateAndRepairToolCallPairs 验证并修复 tool_call/tool_response 配对完整性
//
// 如果发现孤立的 tool_calls（assistant 有 tool_calls 但缺少对应的 tool 响应），
// 删除整个不完整的 tool_call 组（assistant + 部分找到的 tool 响应），
// 而不是创建孤立的 tool 消息。
//
// 如果发现孤立的 tool 响应（没有对应 assistant 的消息），直接删除。
func validateAndRepairToolCallPairs(messages []llm.Message) []llm.Message {
	result := make([]llm.Message, 0, len(messages))

	for i := 0; i < len(messages); i++ {
		msg := messages[i]

		// Case 1: Assistant message with tool_calls
		if msg.Role == llm.RoleAssistant && len(msg.ToolCalls) > 0 {
			// Collect all expected tool_call IDs
			expectedIDs := make(map[string]bool, len(msg.ToolCalls))
			for _, tc := range msg.ToolCalls {
				expectedIDs[tc.ID] = true
			}

			// Collect consecutive tool responses that follow
			matchedResponses := make(map[string]llm.Message)
			j := i + 1
			for j < len(messages) {
				next := messages[j]
				if next.Role == llm.RoleTool && next.ToolCallID != "" {
					if expectedIDs[next.ToolCallID] {
						matchedResponses[next.ToolCallID] = next
					}
					j++
				} else if next.Role == llm.RoleAssistant {
					// Stop at the next assistant message (regardless of tool_calls)
					break
				} else {
					// User, System, or other non-tool messages — stop scanning
					break
				}
			}

			allResponsesPresent := len(matchedResponses) == len(msg.ToolCalls)

			if allResponsesPresent {
				// Complete, valid tool_call group — keep it
				result = append(result, msg)
				// Append responses in the order of tool_calls for determinism
				for _, tc := range msg.ToolCalls {
					if resp, ok := matchedResponses[tc.ID]; ok {
						result = append(result, resp)
					}
				}
			} else {
				// Incomplete tool_call group — remove ENTIRE group (assistant + partial responses)
				// Do NOT create a new assistant without tool_calls (old bug)
				// Do NOT keep partial tool responses (they'd become orphans)
				missingIDs := make([]string, 0)
				for _, tc := range msg.ToolCalls {
					if _, ok := matchedResponses[tc.ID]; !ok {
						missingIDs = append(missingIDs, tc.ID)
					}
				}
				slog.Warn("Removing incomplete tool_call group due to context compression",
					"expected", len(msg.ToolCalls),
					"found", len(matchedResponses),
					"missing_ids", missingIDs,
				)
				// Preserve assistant text content if available (without tool_calls)
				if msg.Content != "" {
					preserved := msg
					preserved.ToolCalls = nil
					result = append(result, preserved)
				}
			}

			// Skip to the position after the tool responses (j already points past them)
			// unmatched tool responses will be handled by Case 2 (orphan detection)
			i = j - 1
			continue
		}

		// Case 2: Orphan tool message (no preceding assistant with matching tool_calls)
		if msg.Role == llm.RoleTool && msg.ToolCallID != "" {
			hasMatchingAssistant := false
			for k := len(result) - 1; k >= 0; k-- {
				if result[k].Role == llm.RoleAssistant && len(result[k].ToolCalls) > 0 {
					for _, tc := range result[k].ToolCalls {
						if tc.ID == msg.ToolCallID {
							hasMatchingAssistant = true
							break
						}
					}
					break
				}
				if result[k].Role == llm.RoleUser || result[k].Role == llm.RoleAssistant {
					break
				}
			}
			if !hasMatchingAssistant {
				// Orphan tool response — drop it
				slog.Warn("Removing orphan tool message (no matching assistant)",
					"tool_call_id", msg.ToolCallID,
				)
				continue
			}
		}

		// Case 3: Normal message (system, user, assistant without tool_calls, or matched tool)
		result = append(result, msg)
	}

	// Merge consecutive assistant messages into single messages
	result = llm.MergeConsecutiveAssistants(result)

	return result
}
