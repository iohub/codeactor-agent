package agents

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"log/slog"
	"path/filepath"
	"strings"

	"codeactor/internal/memory"
	"codeactor/internal/tools"
)

// executeCustomAgent runs a custom agent with its designed system prompt and selected tools.
// Uses the unified AgentExecutor.
func (a *DirectorAgent) executeCustomAgent(ctx context.Context, ca *CustomAgent, adapters []*tools.Adapter, task string) (string, error) {
	systemPrompt := a.GlobalCtx.FormatPrompt(ca.SystemPrompt)

	cfg := DefaultExecutorConfig()
	cfg.SystemPrompt = systemPrompt
	cfg.UserInput = task
	cfg.Adapters = adapters
	cfg.LLM = a.LLM
	cfg.MaxSteps = a.customAgentMaxSteps
	cfg.Publisher = a.Publisher
	cfg.AgentName = ca.DisplayName
	cfg.StopOnFinish = true
	cfg.LLMTimeout = a.llmTimeout
	// EnableCollaboration 已默认 true

	// 创建 Rollout Writer 并注入到 context
	if rolloutWriter := a.createRolloutWriter(ca.DisplayName, task); rolloutWriter != nil {
		defer rolloutWriter.Close()
		ctx = memory.WithRolloutWriter(ctx, rolloutWriter)
	}

	result, err := RunAgentLoop(ctx, cfg)
	if err != nil {
		return "", err
	}
	// 存储自定义 agent memory 供 Run 方法注入
	agentResult := AgentResult{
		Text:   result.Text,
		Memory: ConvertLLMHistoryToMemory(result.History),
	}
	a.pendingSubAgentMemory = &agentResult
	return result.Text, nil
}

// createRolloutWriter 为 delegate 创建 Rollout 写入器
// 返回 nil 表示创建失败（失败时仅警告，不阻断执行）
func (a *DirectorAgent) createRolloutWriter(agentName, task string) *memory.RolloutWriter {
	// 计算 projectID
	projectID := a.computeProjectID()

	writer, err := memory.NewRolloutWriter(agentName, a.taskID, projectID)
	if err != nil {
		slog.Warn("Rollout: failed to create writer, continuing without rollout logging",
			"agent", agentName,
			"error", err,
		)
		return nil
	}

	slog.Debug("Rollout: writer created for delegate agent",
		"agent", agentName,
		"file", writer.FilePath(),
	)

	return writer
}

// computeProjectID 从项目路径计算文件系统安全的 projectID
func (a *DirectorAgent) computeProjectID() string {
	projectPath := a.GlobalCtx.ProjectPath
	if projectPath == "" {
		return "default"
	}
	base := filepath.Base(projectPath)
	if base == "." || base == "/" {
		base = "root"
	}
	// 保留字母数字，其余替换为下划线
	sanitized := strings.Map(func(r rune) rune {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') {
			return r
		}
		return '_'
	}, base)
	// 限制长度
	if len(sanitized) > 100 {
		sanitized = sanitized[:100]
	}
	// 添加短哈希
	h := sha256.Sum256([]byte(projectPath))
	shortHash := hex.EncodeToString(h[:])[:8]
	return sanitized + "_" + shortHash
}

// applyEnhancedCommander 处理子 Agent 执行结果。
// 当前仅存储 sub-agent memory，并原样返回结果文本（不做压缩/截断）。
// agentType: 子 Agent 类型（如 "repo", "coding"）
// task: 委派的任务描述
// result: Agent 执行结果
// err: Agent 执行错误
// 返回: 处理后的结果文本和错误
func (a *DirectorAgent) applyEnhancedCommander(
	agentType string,
	task string,
	result AgentResult,
	err error,
) (string, error) {
	// 始终存储 sub-agent memory（保持现有行为）
	a.pendingSubAgentMemory = &result

	if err != nil {
		return "", err
	}

	return result.Text, nil
}
