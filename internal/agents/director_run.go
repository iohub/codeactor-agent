package agents

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"time"

	"codeactor/internal/knowledge"
	"codeactor/internal/llm"
	"codeactor/internal/memory"
	"codeactor/internal/tools"
)
func (a *DirectorAgent) Run(ctx context.Context, input string, mem *memory.ConversationMemory) (string, error) {
	// 设置当前 memory（delegate 闭包通过 a.currentMemory 访问）
	a.currentMemory = mem

	// 执行检测机制：每次任务开始时重置委派状态，使检测基于"本次任务是否委派"
	a.hasDelegated = false
	a.nonDelegationPrompts = 0

	// 从 llmClient 刷新引擎，确保 TUI 中切换模型后立即生效
	if a.llmClient != nil {
		newEngine := a.llmClient.GetAgentEngine("director")
		if newEngine != nil {
			a.LLM = newEngine
		}
		// 刷新所有子 Agent 的引擎，确保 TUI 切换模型后子 Agent 也使用新模型
		a.refreshSubAgentEngines()
	}
	defer func() { a.currentMemory = nil }()

	if mem != nil {
		// Check if the last message is the same as input to avoid duplication
		// because handleChatMessage might have already added it.
		lastMsg := mem.GetLastMessage()
		if lastMsg == nil || lastMsg.Content != input || lastMsg.Type != memory.MessageTypeHuman {
			mem.AddHumanMessage(input)
		}
	}

	var messages []llm.Message

	// Always start with System Prompt (with any registered custom agents appended)
	systemPrompt := a.GlobalCtx.FormatPrompt(directorPrompt)
	var projectContext string
	// 只在首次对话时加载项目上下文文件（CODEACTOR.md、CLAUDE.md、AGENTS.md），
	// 同一会话的后续追问无需重复注入，避免浪费 token。
	// memory 中不存储 system 消息，因此 len(mem.GetMessages()) == 0 即可判断是否为首次对话。
	if mem == nil || len(mem.GetMessages()) == 0 {
		if loadResult := a.loadProjectContext(); loadResult != nil && loadResult.Content != "" {
			// 发送上下文加载完成消息到消息通道
			if a.Publisher != nil {
				a.Publisher.Publish("context_loaded", loadResult, a.Name())
			}
			// 延迟追加：先构建完整的 system prompt（静态前缀 + 环境信息 + 自定义 Agent），
			// 最后才追加项目上下文，确保静态前缀可被 LLM Prompt Cache 复用
			projectContext = fmt.Sprintf("\n\n### Project Workspace Context\n%s\n", loadResult.Content)
		}
	}

	// 自定义 Agent 描述
	if len(a.customAgents) > 0 {
		systemPrompt += "\n\n### Custom Agents\nThe following specialized agents have been designed by Meta-Agent and are permanently available for delegation:\n\n"
		for _, ca := range a.customAgents {
			systemPrompt += fmt.Sprintf("- **%s** (`delegate_%s`): %s\n", ca.DisplayName, ca.Name, ca.Description)
		}
		systemPrompt += "\nUse these agents via their delegate tools for tasks matching their specializations.\n"
	}

	// 追加项目上下文（放在所有静态内容之后，确保缓存命中率）
	if projectContext != "" {
		systemPrompt += projectContext
	}

	// [知识管理] Director 自身 systemPrompt 动态知识检索注入
	if a.GlobalCtx.KnowledgeInjector != nil {
		injCtx := knowledge.InjectionContext{
			UserMessage: input,
			TargetFiles: nil,
			AgentName:   a.Name(),
			// Domains 为空 = 检索全部 domain（Director 不限定）
		}
		if knowledgeBlock, err := a.GlobalCtx.KnowledgeInjector.Inject(ctx, injCtx); err == nil && knowledgeBlock != "" {
			systemPrompt += knowledgeBlock
		}
	}

	messages = append(messages, llm.Message{
		Role:    llm.RoleSystem,
		Content: systemPrompt,
	})

	if mem != nil {
		for _, m := range mem.GetMessages() {
			if m.Type == memory.MessageTypeSystem {
				continue
			}
			// 过滤 sub-agent 内部消息，避免破坏 tool_calls → tool 消息配对规则
			// sub-agent 消息由 injectSubAgentMemory() 注入，仅用于内存记录，不应发送给 LLM
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

	toolDefs := make([]llm.ToolDef, len(a.Adapters))
	for i, ad := range a.Adapters {
		toolDefs[i] = ad.ToToolDef()
	}
	tools.SortToolDefs(toolDefs)

	// Publish model info so the TUI can display it in the status bar.
	if a.Publisher != nil {
		a.Publisher.Publish("model_info", map[string]interface{}{
			"model": a.LLM.Model(),
			"agent": a.Name(),
		}, a.Name())
	}

	// ═══════ 初始化 Director Rollout Writer ═══════
	var directorRolloutWriter *memory.RolloutWriter
	if rw := a.createRolloutWriter("director", input); rw != nil {
		directorRolloutWriter = rw
		defer func() {
			if directorRolloutWriter != nil {
				directorRolloutWriter.Close()
			}
		}()
	}

	writeDirectorRollout := func(msg llm.Message) {
		if directorRolloutWriter == nil || !directorRolloutWriter.Enabled() {
			return
		}
		// 首次写入时记录 session_meta 和 turn_context
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
		msgID := directorRolloutWriter.NextMessageID()
		items := memory.LLMMessageToResponseItems(msg, msgID)
		for _, item := range items {
			if err := directorRolloutWriter.WriteResponseItem(item); err != nil {
				slog.Warn("Rollout: failed to write director message",
					"error", err,
				)
			}
		}
	}
	// ═══════ END Director Rollout Writer ═══════

	// ═══════ 写入初始消息（system prompt + user input） ═══════
	initialMsgCount := len(messages)
	for i := 0; i < initialMsgCount; i++ {
		writeDirectorRollout(messages[i])
	}

	for i := 0; i < a.maxSteps; i++ {
		// --- 熔断检查（通过适配器委托到新组件）---
		if a.adapter != nil && a.adapter.IsCircuitBreakerOpen() {
			slog.Error("Circuit breaker open, too many consecutive LLM failures",
				"step", i)
			// Rollout: 写入任务中止事件
			if directorRolloutWriter != nil && directorRolloutWriter.Enabled() {
				directorRolloutWriter.WriteEventMsg(memory.EventMsg{
					Type:   "turn_aborted",
					Reason: "circuit breaker open",
				})
			}
			return "", fmt.Errorf("circuit breaker open: LLM calls blocked")
		}

		// --- 步骤级重试 ---
		maxRetries := a.stepRetries
		var resp *llm.Response
		var llmErr error
		for attempt := 0; attempt <= maxRetries; attempt++ {
			if attempt > 0 {
				// 指数退避：1s, 2s, 4s, 8s, ... 最大30s
				wait := time.Duration(1<<(attempt-1)) * time.Second
				if wait > 30*time.Second {
					wait = 30 * time.Second
				}
				slog.Warn("DirectorAgent retrying LLM call", "step", i, "attempt", attempt, "wait", wait)
				select {
				case <-ctx.Done():
					// Rollout: 写入任务中止事件
					if directorRolloutWriter != nil && directorRolloutWriter.Enabled() {
						directorRolloutWriter.WriteEventMsg(memory.EventMsg{
							Type:   "turn_aborted",
							Reason: ctx.Err().Error(),
						})
					}
					return "", ctx.Err()
				case <-time.After(wait):
				}
			}

			// 验证并修复 tool_call/tool_response 配对完整性
			messages = validateAndRepairToolCallPairs(messages)

			// 上下文压缩:token 超阈值时按优先级截断 tool 执行结果
			if a.EnhancedCommanderCfg.Enable && a.EnhancedCommanderCfg.EnableContextCompression {
				threshold := a.EnhancedCommanderCfg.ContextCompressionThreshold
				if threshold <= 0 {
					threshold = DefaultContextCompressionThreshold
				}
				keepTokens := a.EnhancedCommanderCfg.ToolResultKeepTokens
				if keepTokens <= 0 {
					keepTokens = DefaultToolResultKeepTokens
				}
				var compStats *ContextCompressionStats
				messages, compStats = TruncateToolResultsToBudget(messages, threshold, keepTokens)
				if compStats != nil && compStats.TruncatedCount > 0 && a.Publisher != nil {
					truncatedTools := make([]map[string]interface{}, len(compStats.TruncatedTools))
					for ti, tool := range compStats.TruncatedTools {
						truncatedTools[ti] = map[string]interface{}{
							"tool_name":       tool.ToolName,
							"original_tokens": tool.OriginalTokens,
							"kept_tokens":     tool.KeptTokens,
							"omitted_tokens":  tool.OmittedTokens,
						}
					}
					a.Publisher.Publish("context_compressed", map[string]interface{}{
						"original_tokens":   compStats.OriginalTokens,
						"compressed_tokens": compStats.CompressedTokens,
						"saved_tokens":      compStats.SavedTokens,
						"saved_percent":     compStats.SavedPercent,
						"truncated_count":   compStats.TruncatedCount,
						"truncated_tools":   truncatedTools,
					}, a.Name())
					slog.Debug("context compression applied",
						"original_tokens", compStats.OriginalTokens,
						"compressed_tokens", compStats.CompressedTokens,
						"saved_tokens", compStats.SavedTokens,
						"saved_percent", compStats.SavedPercent,
						"truncated_count", compStats.TruncatedCount)
				}

				// 紧急压缩:tool 结果已全部截断后仍超限 → 启动紧急模式, 极致压缩 memory 继续任务
				if estimateMessagesTokens(messages) > threshold {
					newMessages, emergencyStats := a.applyEmergencyCompression(ctx, messages, threshold)
					messages = newMessages
					if emergencyStats != nil && a.Publisher != nil {
						a.Publisher.Publish("context_emergency_compressed", map[string]interface{}{
							"original_tokens":   emergencyStats.OriginalTokens,
							"compressed_tokens": emergencyStats.CompressedTokens,
							"saved_tokens":      emergencyStats.SavedTokens,
							"extracted_blocks":  emergencyStats.ExtractedBlocks,
							"summarized_blocks": emergencyStats.SummarizedBlocks,
							"kept_blocks":       emergencyStats.KeptBlocks,
							"summarized_by_llm": emergencyStats.SummarizedByLLM,
						}, a.Name())
					}
					slog.Warn("emergency context compression applied",
						"original_tokens", emergencyStats.OriginalTokens,
						"compressed_tokens", emergencyStats.CompressedTokens,
						"extracted_blocks", emergencyStats.ExtractedBlocks,
						"kept_blocks", emergencyStats.KeptBlocks,
						"summarized_by_llm", emergencyStats.SummarizedByLLM)
				}
			}

			slog.Debug("DirectorAgent calling LLM", "step", i, "messages", messages)

			// Create opts with streaming handler for real-time output
			opts := &llm.CallOptions{}
			if a.Publisher != nil {
				opts.StreamHandler = func(ctx context.Context, chunk []byte) error {
					if len(chunk) > 0 {
						a.Publisher.Publish("ai_chunk", map[string]interface{}{
							"content": string(chunk),
							"agent":   a.Name(),
						}, a.Name())
					}
					return nil
				}
			}

			// Publish llm_call_start event
			if a.Publisher != nil {
				a.Publisher.Publish("llm_call_start", map[string]interface{}{
					"model": a.LLM.Model(),
					"agent": a.Name(),
				}, a.Name())
			}

			// Publish ai_stream_start before LLM call
			if a.Publisher != nil {
				a.Publisher.Publish("ai_stream_start", map[string]interface{}{
					"agent": a.Name(),
				}, a.Name())
			}

			llmStartTime := time.Now()
			// 使用可配置的 LLM 超时保护，防止远程服务无响应时永久阻塞
			llmCtx, llmCancel := context.WithTimeout(ctx, a.llmTimeout)
			resp, llmErr = a.LLM.GenerateContent(llmCtx, messages, toolDefs, opts)
			llmCancel()
			llmDuration := time.Since(llmStartTime).Seconds()

			// Publish ai_stream_end after LLM call
			if a.Publisher != nil {
				metadata := map[string]interface{}{
					"agent": a.Name(),
				}
				if llmErr == nil && resp != nil && resp.Usage != nil {
					metadata["usage"] = map[string]interface{}{
						"prompt_tokens":               resp.Usage.PromptTokens,
						"completion_tokens":           resp.Usage.CompletionTokens,
						"total_tokens":                resp.Usage.TotalTokens,
						"cache_creation_input_tokens": resp.Usage.CacheCreationInputTokens,
						"cache_read_input_tokens":     resp.Usage.CacheReadInputTokens,
						"total_input_tokens":          resp.Usage.TotalInputTokens,
					}
				}
				a.Publisher.PublishWithMetadata("ai_stream_end", "", a.Name(), metadata)
			}

			// 记录 LLM 耗时指标
			if a.adapter != nil {
				a.adapter.RecordLLMDuration(time.Since(llmStartTime))
			}

			// Publish thinking event (reasoning content) before llm_call_end
			if llmErr == nil && a.Publisher != nil && len(resp.Choices) > 0 {
				reasoning := resp.Choices[0].Reasoning
				if reasoning != "" {
					a.Publisher.Publish("thinking", map[string]interface{}{
						"content": reasoning,
						"model":   a.LLM.Model(),
						"agent":   a.Name(),
					}, a.Name())
				}
			}

			// Publish llm_call_end event
			if a.Publisher != nil {
				metadata := map[string]interface{}{
					"model":            a.LLM.Model(),
					"agent":            a.Name(),
					"duration_seconds": llmDuration,
				}
				if llmErr != nil {
					metadata["error"] = llmErr.Error()
				}
				a.Publisher.PublishWithMetadata("llm_call_end", "", a.Name(), metadata)
			}

			if llmErr == nil {
				// 通过适配器记录成功（重置熔断器状态）
				if a.adapter != nil {
					a.adapter.RecordLLMSuccess()
				}
				break
			}
			// 通过适配器记录失败（可能触发熔断）
			if a.adapter != nil {
				a.adapter.RecordLLMFailure()
			}
			a.consecutiveLLMFailures++
			a.lastLLMFailureTime = time.Now()
			slog.Warn("DirectorAgent LLM error, will retry",
				"error", llmErr, "step", i, "attempt", attempt,
				"consecutive_failures", a.consecutiveLLMFailures)
		}

		if llmErr != nil {
			slog.Error("DirectorAgent LLM error after all retries",
				"error", llmErr, "step", i)
			// Rollout: 写入任务中止事件
			if directorRolloutWriter != nil && directorRolloutWriter.Enabled() {
				directorRolloutWriter.WriteEventMsg(memory.EventMsg{
					Type:   "turn_aborted",
					Reason: llmErr.Error(),
				})
			}
			return "", llmErr
		}

		choice := resp.Choices[0]
		slog.Debug("DirectorAgent LLM response", "step", i, "content", choice.Content, "tool_calls", len(choice.ToolCalls))

		// 计算 token usage（实际值或估算值）
		var promptTokens, completionTokens, totalTokens, cacheCreationTokens, cacheReadTokens int64
		if resp.Usage != nil {
			promptTokens = resp.Usage.PromptTokens
			completionTokens = resp.Usage.CompletionTokens
			totalTokens = resp.Usage.TotalTokens
			cacheCreationTokens = resp.Usage.CacheCreationInputTokens
			cacheReadTokens = resp.Usage.CacheReadInputTokens
		} else {
			// 本地模型可能不返回 usage，按内容估算
			estPrompt := 0
			for _, msg := range messages {
				estPrompt += EstimateTokens(msg.Content)
			}
			estCompletion := EstimateTokens(choice.Content)
			promptTokens = int64(estPrompt)
			completionTokens = int64(estCompletion)
			totalTokens = promptTokens + completionTokens
		}

		// Rollout: 写入 token_count 事件
		if directorRolloutWriter != nil && directorRolloutWriter.Enabled() {
			if writeErr := directorRolloutWriter.WriteTokenCount(promptTokens, completionTokens, totalTokens, cacheCreationTokens, cacheReadTokens); writeErr != nil {
				slog.Warn("Rollout: failed to write director token_count", "error", writeErr)
			}
		}

		if a.Publisher != nil {
			metadata := map[string]interface{}{}
			if resp.Usage != nil {
				metadata["usage"] = map[string]interface{}{
					"prompt_tokens":               int(resp.Usage.PromptTokens),
					"completion_tokens":           int(resp.Usage.CompletionTokens),
					"total_tokens":                int(resp.Usage.TotalTokens),
					"cache_creation_input_tokens": int(resp.Usage.CacheCreationInputTokens),
					"cache_read_input_tokens":     int(resp.Usage.CacheReadInputTokens),
					"total_input_tokens":          int(resp.Usage.TotalInputTokens),
				}
			} else {
				metadata["usage"] = map[string]interface{}{
					"prompt_tokens":     int(promptTokens),
					"completion_tokens": int(completionTokens),
					"total_tokens":      int(totalTokens),
					"estimated":         true,
				}
			}
			a.Publisher.PublishWithMetadata("ai_response", choice.Content, a.Name(), metadata)
		}

		if mem != nil {
			mem.AddAssistantMessage(choice.Content, convertToolCalls(choice.ToolCalls))
		}

		messages = append(messages, llm.Message{
			Role:      llm.RoleAssistant,
			Content:   choice.Content,
			Reasoning: choice.Reasoning,
			ToolCalls: choice.ToolCalls,
		})

		writeDirectorRollout(llm.Message{
			Role:      llm.RoleAssistant,
			Content:   choice.Content,
			Reasoning: choice.Reasoning,
			ToolCalls: choice.ToolCalls,
		})

		// 执行检测机制：若 director 未委派任何 agent 就打算以纯文本结束任务，
		// 以用户角色注入简练英文消息，强制要求其必须 delegate 一个 agent 完成任务。
		// 达到 maxNonDelegationPrompts 上限后放行（防死循环，循环本身还有 maxSteps 兜底）。
		if len(choice.ToolCalls) == 0 {
			if !a.hasDelegated && a.nonDelegationPrompts < maxNonDelegationPrompts {
				a.nonDelegationPrompts++
				var forceMsg string
				if a.nonDelegationPrompts == 1 {
					forceMsg = "You must delegate an agent to complete the task. Do not reply with plain text — call a delegate_* tool now."
				} else {
					forceMsg = "You still have not delegated any agent. You MUST call a delegate_* tool to complete the task before responding."
				}
				userMsg := llm.Message{Role: llm.RoleUser, Content: forceMsg}
				messages = append(messages, userMsg)
				writeDirectorRollout(userMsg)
				slog.Debug("DirectorAgent force delegation via user message", "step", i, "prompt_count", a.nonDelegationPrompts)
				// 注意：不写入 ConversationMemory（mem），该消息是系统模拟的用户指令，
				// 不应污染会话历史；下一轮 LLM 调用会看到该 user 消息并应调用 delegate 工具。
				continue
			}
			// Rollout: 写入任务完成事件
			if directorRolloutWriter != nil && directorRolloutWriter.Enabled() {
				directorRolloutWriter.WriteEventMsg(memory.EventMsg{
					Type: "task_complete",
				})
			}
			return choice.Content, nil
		}

		for _, tc := range choice.ToolCalls {
			var toolResult string
			var err error
			found := false

			if a.Publisher != nil {
				a.Publisher.Publish("tool_call_start", map[string]interface{}{
					"tool_name":    tc.Function.Name,
					"arguments":    tc.Function.Arguments,
					"tool_call_id": tc.ID,
				}, a.Name())
			}
			for _, t := range a.Adapters {
				if t.Name() == tc.Function.Name {
					found = true

					// Log delegate tool calls with full arguments to dedicated delegate log
					if strings.HasPrefix(t.Name(), "delegate_") {
						agentName := strings.TrimPrefix(t.Name(), "delegate_")
						LogDelegateCall(t.Name(), agentName, tc.Function.Arguments)
					}

					// 工具调用前检查 context
					if ctx.Err() != nil {
						// Rollout: 写入任务中止事件
						if directorRolloutWriter != nil && directorRolloutWriter.Enabled() {
							directorRolloutWriter.WriteEventMsg(memory.EventMsg{
								Type:   "turn_aborted",
								Reason: ctx.Err().Error(),
							})
						}
						return "", ctx.Err()
					}

					// 为工具调用添加超时保护（防止工具无限阻塞）
					// delegate_* 工具涉及子 agent 完整执行（多轮 LLM + 工具调用），需要更长的超时时间
					// 使用 WithCancel 剥离父 context 的 deadline，再 WithTimeout 添加独立超时
					// 这确保工具获得完整的超时时间，不受父 context 剩余时间限制
					toolTimeout := 120 * time.Second
					if strings.HasPrefix(tc.Function.Name, "delegate_") {
						toolTimeout = 10 * time.Minute // 子 agent 需要更多时间完成多轮交互
					}
					cancelCtx, cancelCtxCancel := context.WithCancel(ctx)
					toolCtx, toolCancel := context.WithTimeout(cancelCtx, toolTimeout)
					toolResult, err = t.Call(toolCtx, tc.Function.Arguments)
					cancelCtxCancel()
					toolCancel()

					// 注入 sub-agent memory（delegate 闭包中设置了 pendingSubAgentMemory）
					if a.pendingSubAgentMemory != nil {
						a.injectSubAgentMemory(*a.pendingSubAgentMemory, tc.ID, tc.Function.Name)
						a.pendingSubAgentMemory = nil
					}

					// 检测是否是 delegate 工具，无论成功失败都记录尝试次数
					if strings.HasPrefix(t.Name(), "delegate_") {
						a.delegationAttempts++
						if err == nil {
							a.hasDelegated = true
						}
					}

					if err != nil {
						// 截断过长的错误消息，避免污染上下文
						errMsg := err.Error()
						if len(errMsg) > 1000 {
							errMsg = errMsg[:1000] + "... [truncated]"
						}
						// 对超时错误给出明确的超时时间提示
						if errors.Is(err, context.DeadlineExceeded) {
							timeoutMinutes := int(toolTimeout.Minutes())
							toolResult = fmt.Sprintf("Error: tool execution timed out after %d seconds", timeoutMinutes*60)
						} else {
							toolResult = fmt.Sprintf("Error: %s", errMsg)
						}
					} else if t.Name() == "delegate_repo" {
						// toolResult is a JSON string (e.g. "\"summary...\""), so we need to unmarshal it
						// to get the actual text content
						var summary string
						if err := json.Unmarshal([]byte(toolResult), &summary); err == nil {
							a.GlobalCtx.RepoSummary = summary
						} else {
							a.GlobalCtx.RepoSummary = toolResult
						}
					}
					break
				}
			}
			if !found {
				toolResult = fmt.Sprintf("Tool %s not found", tc.Function.Name)
			}

			if a.Publisher != nil {
				a.Publisher.Publish("tool_call_result", map[string]interface{}{
					"tool_name":    tc.Function.Name,
					"result":       toolResult,
					"tool_call_id": tc.ID,
				}, a.Name())
			}

			if mem != nil {
				mem.AddToolMessage(toolResult, tc.ID)
			}

			messages = append(messages, llm.Message{
				Role:       llm.RoleTool,
				Content:    toolResult,
				ToolCallID: tc.ID,
				ToolName:   tc.Function.Name,
			})

			writeDirectorRollout(llm.Message{
				Role:       llm.RoleTool,
				Content:    toolResult,
				ToolCallID: tc.ID,
				ToolName:   tc.Function.Name,
			})

			if tc.Function.Name == "agent_exit" {
				// Rollout: 写入任务完成事件
				if directorRolloutWriter != nil && directorRolloutWriter.Enabled() {
					directorRolloutWriter.WriteEventMsg(memory.EventMsg{
						Type: "task_complete",
					})
				}
				return "Task completed successfully", nil
			}

		}
	}

	// Rollout: 写入任务中止事件
	if directorRolloutWriter != nil && directorRolloutWriter.Enabled() {
		directorRolloutWriter.WriteEventMsg(memory.EventMsg{
			Type:   "turn_aborted",
			Reason: "DirectorAgent exceeded max steps",
		})
	}
	return "", fmt.Errorf("DirectorAgent exceeded max steps")
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
