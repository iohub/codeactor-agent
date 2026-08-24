package agents

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"codeactor/internal/llm"
	"codeactor/internal/memory"
)
// loadProjectContext 读取工作区目录下的项目上下文文件（CODEACTOR.md、CLAUDE.md、AGENTS.md），
// 将成功读取的文件内容格式化后组合返回。文件按顺序尝试，不存在或读取失败时忽略。
// 返回加载的文件列表和组合后的内容。
func (a *DirectorAgent) loadProjectContext() *ProjectContextLoadResult {
	// 如果已经加载过，直接返回缓存（同一 Agent 实例会话内只加载一次）
	if a.cachedProjectContext != nil {
		return a.cachedProjectContext
	}
	result := &ProjectContextLoadResult{
		LoadedFiles: []ProjectContextFile{},
	}
	var sb strings.Builder
	contextFiles := []string{"CODEACTOR.md", "CLAUDE.md", "AGENTS.md"}

	for _, fname := range contextFiles {
		fullPath := filepath.Join(a.GlobalCtx.ProjectPath, fname)
		data, err := os.ReadFile(fullPath)
		if err != nil {
			// 文件不存在或读取失败，忽略并继续尝试下一个
			continue
		}
		if len(data) > 0 {
			result.LoadedFiles = append(result.LoadedFiles, ProjectContextFile{
				FileName: fname,
				Content:  string(data),
			})
			sb.WriteString(fmt.Sprintf("\n### %s\n```\n%s\n```\n", fname, string(data)))
		}
	}
	result.Content = sb.String()
	// 缓存加载结果，避免后续调用重复读取文件
	a.cachedProjectContext = result
	return result
}

// injectSubAgentMemory 将 sub-agent 的执行结果摘要注入到 Director memory 中
// Phase 1: 只注入摘要，不再注入 sub-agent 的完整对话历史
// Phase 3+ : Sub-agent 的关键发现通过 SharedMemory 发布/订阅机制共享
func (a *DirectorAgent) injectSubAgentMemory(result AgentResult, toolCallID string, toolName string) {
	if a.currentMemory == nil {
		return
	}

	// 只注入摘要消息（result.Text），不注入完整历史
	// sub-agent 的完整对话历史保留在其自身的 LocalMemory 中（Phase 3）
	if result.Text != "" {
		summaryMsg := memory.ChatMessage{
			Type:       memory.MessageTypeAssistant,
			Content:    fmt.Sprintf("[Sub-Agent Result: %s]\n%s", toolName, result.Text),
			Timestamp:  time.Now(),
			GroupID:    fmt.Sprintf("%s_summary_%d", toolName, time.Now().UnixNano()),
			ParentID:   toolCallID,
			IsSubAgent: true,
			Metadata: map[string]interface{}{
				"type":      "sub_agent_summary",
				"tool":      toolName,
				"msg_count": len(result.Memory),
			},
		}
		a.currentMemory.Messages = append(a.currentMemory.Messages, summaryMsg)
	}

	// 重要：result.Memory（sub-agent 的完整对话历史）不再注入到 Director 的 memory 中
	// 这避免了 Director 上下文快速膨胀和 Compact Engine 频繁压缩造成的信息丢失
	// sub-agent 内部消息保留在 sub-agent 本地，通过 SharedMemory 的 publish/subscribe 机制共享关键信息（Phase 3）
}

// SetTaskID 设置当前任务的 taskID
func (a *DirectorAgent) SetTaskID(taskID string) {
	a.taskID = taskID
}

func convertToolCalls(tcs []llm.ToolCall) []memory.ToolCallData {
	var res []memory.ToolCallData
	for _, tc := range tcs {
		res = append(res, memory.ToolCallData{
			ID:   tc.ID,
			Type: tc.Type,
			Function: memory.ToolCallFunction{
				Name:      tc.Function.Name,
				Arguments: json.RawMessage(tc.Function.Arguments),
			},
		})
	}
	return res
}
