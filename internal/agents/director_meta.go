package agents

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"

	"codeactor/internal/tools"
)
// getToolFunc returns the ToolFunc implementation for a given tool name.
// This is used when constructing tool adapters for dynamically created agents.
func (a *DirectorAgent) getToolFunc(name string) tools.ToolFunc {
	switch name {
	case "read_file":
		return a.GlobalCtx.FileOps.ExecuteReadFile
	case "search_replace_in_file":
		return a.GlobalCtx.ReplaceTool.ExecuteReplaceBlock
	case "create_file":
		return a.GlobalCtx.FileOps.ExecuteCreateFile
	case "run_bash":
		return a.GlobalCtx.SysOps.ExecuteRunBash
	case "search_by_regex":
		return a.GlobalCtx.SearchOps.ExecuteGrepSearch
	case "list_dir":
		return a.GlobalCtx.FileOps.ExecuteListDir
	case "print_dir_tree":
		return a.GlobalCtx.FileOps.ExecutePrintDirTree
	case "thinking":
		return func(ctx context.Context, params map[string]interface{}) (interface{}, error) {
			inputBytes, _ := json.Marshal(params)
			return a.GlobalCtx.ThinkingTool.Call(ctx, string(inputBytes))
		}
	case "micro_agent":
		return a.GlobalCtx.MicroAgentTool.Execute
	case "deepthinking":
		return a.GlobalCtx.DeepThinkingTool.Execute
	case "agent_exit":
		return a.GlobalCtx.FlowOps.ExecuteAgentExit
	case "ask_user_for_help":
		if a.GlobalCtx.FullYoloMode {
			return nil
		}
		return a.GlobalCtx.FlowOps.ExecuteAskUserForHelp
	default:
		return nil
	}
}

// parseMetaAgentOutput extracts and validates the JSON object from Meta-Agent's raw output.
// It strips markdown code fences and surrounding text to find the JSON.
func parseMetaAgentOutput(output string) (systemPrompt string, execResult *metaAgentResult, err error) {
	jsonStr := extractJSONObject(output)
	if jsonStr == "" {
		return "", nil, fmt.Errorf("no JSON object found in Meta-Agent output")
	}

	execResult = &metaAgentResult{}
	if err := json.Unmarshal([]byte(jsonStr), execResult); err != nil {
		return "", nil, fmt.Errorf("failed to parse Meta-Agent JSON: %w", err)
	}

	// Validate required fields
	if execResult.AgentName == "" {
		return "", nil, fmt.Errorf("agent_name is empty in Meta-Agent JSON")
	}
	if execResult.AgentDesign == "" {
		return "", nil, fmt.Errorf("agent_design is empty in Meta-Agent JSON")
	}

	return execResult.AgentDesign, execResult, nil
}

// extractJSONObject finds the outermost JSON object in a string.
// It strips markdown code fences and handles surrounding text.
func extractJSONObject(s string) string {
	raw := s

	// Strip markdown code fences: ```json ... ``` or ``` ... ```
	if idx := strings.Index(raw, "```"); idx != -1 {
		endFence := strings.Index(raw[idx+3:], "```")
		if endFence != -1 {
			inner := raw[idx+3 : idx+3+endFence]
			// Skip optional language tag after opening ```
			if newline := strings.Index(inner, "\n"); newline != -1 {
				inner = inner[newline+1:]
			}
			raw = inner
		}
	}

	// Find the outermost { ... }
	start := strings.Index(raw, "{")
	if start == -1 {
		return ""
	}
	// Walk braces to find the matching close brace
	depth := 0
	for i := start; i < len(raw); i++ {
		switch raw[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return raw[start : i+1]
			}
		}
	}
	return ""
}

// toSnakeCase converts a display name like "Security Auditor" to "security_auditor".
func toSnakeCase(name string) string {
	// Lowercase and replace non-alphanumeric characters with underscores
	var result strings.Builder
	for i, r := range strings.ToLower(name) {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			result.WriteRune(r)
		} else if r == ' ' || r == '-' || r == '_' {
			if i > 0 {
				result.WriteRune('_')
			}
		} else {
			result.WriteRune('_')
		}
	}
	// Trim leading/trailing underscores and collapse consecutive underscores
	raw := result.String()
	// Collapse consecutive underscores
	for strings.Contains(raw, "__") {
		raw = strings.ReplaceAll(raw, "__", "_")
	}
	raw = strings.Trim(raw, "_")
	if raw == "" {
		raw = "custom_agent"
	}
	return raw
}

// registerCustomAgent creates a new delegate_<name> tool for a custom agent designed by Meta-Agent
// and adds it to the Director's Adapters list. The agent becomes permanently available.
func (a *DirectorAgent) registerCustomAgent(ca *CustomAgent) {
	delegateName := "delegate_" + ca.Name

	// Check if already registered
	if _, exists := a.customAgents[delegateName]; exists {
		slog.Info("Custom agent already registered", "name", delegateName)
		return
	}

	// Build tool adapters for the custom agent's selected tools
	customAdapters := make([]*tools.Adapter, 0, len(ca.ToolsUsed))
	for _, toolName := range ca.ToolsUsed {
		fn := a.getToolFunc(toolName)
		if fn == nil {
			slog.Warn("Custom agent references unknown tool", "agent", ca.Name, "tool", toolName)
			continue
		}
		def, ok := a.toolDefMap[toolName]
		if !ok {
			slog.Warn("Tool definition not found in toolDefMap", "tool", toolName)
			continue
		}
		adapter := tools.NewAdapter(def.Name, def.Description, fn).WithSchema(def.Parameters)
		customAdapters = append(customAdapters, adapter)
	}

	// Add agent_exit tool so the custom agent can signal exit
	finishDef, ok := a.toolDefMap["agent_exit"]
	if ok {
		fn := a.getToolFunc("agent_exit")
		adapter := tools.NewAdapter("agent_exit", finishDef.Description, fn).WithSchema(finishDef.Parameters)
		customAdapters = append(customAdapters, adapter)
	}

	// Set workspace guard on the custom agent's adapters
	tools.SetGuardOnAdapters(customAdapters, a.GlobalCtx.Guard)

	// Create the delegate tool that executes the custom agent
	// Capture ca and customAdapters in closure
	agentRef := ca
	adaptersRef := customAdapters

	description := fmt.Sprintf("Delegate to %s — a custom specialized agent designed by Meta-Agent. %s",
		ca.DisplayName, ca.Description)

	delegateAdapter := tools.NewAdapter(delegateName, description,
		func(ctx context.Context, params map[string]interface{}) (interface{}, error) {
			task, ok := params["task"].(string)
			if !ok {
				return nil, fmt.Errorf("task parameter required")
			}
			return a.executeCustomAgent(ctx, agentRef, adaptersRef, task)
		}).WithSchema(map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"task": map[string]interface{}{"type": "string", "description": "The task description for " + ca.DisplayName},
		},
		"required": []string{"task"},
	})

	a.Adapters = append(a.Adapters, delegateAdapter)
	a.customAgents[delegateName] = ca

	// 同步更新工具分发索引
	if a.adapterIndex != nil {
		a.adapterIndex[delegateAdapter.Name()] = delegateAdapter
	}

	slog.Info("Custom agent registered", "delegate_name", delegateName, "display_name", ca.DisplayName, "tools", ca.ToolsUsed)
}
