package agents

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"time"

	director "codeactor/internal/agents/director"
	"codeactor/internal/config"
	"codeactor/internal/globalctx"
	"codeactor/internal/llm"
	"codeactor/internal/memory"
	"codeactor/internal/tools"
)

//go:embed director.prompt.md
var directorPrompt string

// maxNonDelegationPrompts 限制"未委派强制提醒"的最大次数。
// 当 director 未委派任何 agent 就打算以纯文本结束时，以用户角色注入强制消息；
// 若 LLM 持续拒绝委派，达到此上限后放行原内容，防止无限循环。
const maxNonDelegationPrompts = 3

// CustomAgent stores a dynamically designed agent created by Meta-Agent.
// Once registered, it becomes available as a permanent delegate tool.
type CustomAgent struct {
	Name         string   // snake_case identifier used for the delegate tool name
	DisplayName  string   // human-readable agent name
	SystemPrompt string   // the full system prompt designed by Meta-Agent
	ToolsUsed    []string // tool names this agent was designed to use
	Description  string   // short description for the LLM
}

// metaAgentResult parses the JSON output from Meta-Agent.
type metaAgentResult struct {
	Thinking     string   `json:"thinking"`
	AgentName    string   `json:"agent_name"`
	AgentDesign  string   `json:"agent_design"`
	ToolsUsed    []string `json:"tools_used"`
	TaskForAgent string   `json:"task_for_agent"`
}

// ProjectContextFile represents a successfully loaded project context file
type ProjectContextFile struct {
	FileName string `json:"file_name"`
	Content  string `json:"content"`
}

// ProjectContextLoadResult represents the result of loading project context files
type ProjectContextLoadResult struct {
	LoadedFiles []ProjectContextFile `json:"loaded_files"`
	Content     string               `json:"content"`
}

type DirectorAgent struct {
	BaseAgent
	RepoAgent      *RepoAgent
	CodingAgent    *CodingAgent
	ChatAgent      *ChatAgent
	MetaAgent      *MetaAgent
	DevOpsAgent    *DevOpsAgent
	BrowserAgent   *BrowserAgent
	GlobalCtx      *globalctx.GlobalCtx
	Adapters       []*tools.Adapter
	maxSteps       int
	metaRetryCount int                             // max retries for Meta-Agent JSON parse failures
	toolDefMap     map[string]tools.ToolDefinition // tool name → definition from tools.json
	customAgents   map[string]*CustomAgent         // delegate_<name> → agent design
	adapter        *DirectorAdapter                // 新旧整合适配器
	llmClient      *llm.Client                     // LLM客户端引用，用于运行时动态重新解析引擎

	cachedProjectContext *ProjectContextLoadResult // 缓存项目上下文文件（同一会话只加载一次）
	hasDelegated         bool                      // 标记是否已委派过 agent
	nonDelegationPrompts int                       // 本次任务中"未委派强制提醒"已注入次数（执行检测机制）
	delegationAttempts   int                       // 委派尝试次数统计

	currentMemory         *memory.ConversationMemory // 当前正在使用的 memory（Run 期间设置）
	pendingSubAgentMemory *AgentResult               // 最近一次 delegate 调用的完整结果（用于 memory 注入）

	// LLM 兜底机制字段
	stepRetries                int           // 步骤重试次数（从 config.LLM.StepRetries 读取）
	llmTimeout                 time.Duration // LLM调用超时，从配置读取，默认3分钟
	circuitBreakerThreshold    int           // 熔断阈值（连续失败次数），0=不启用
	circuitBreakerResetTimeout time.Duration // 熔断恢复时间
	consecutiveLLMFailures     int           // 连续 LLM 调用失败计数
	lastLLMFailureTime         time.Time     // 最近一次 LLM 失败时间

	// EnhancedCommander 增强型配置
	EnhancedCommanderCfg config.EnhancedCommanderConfig
	// taskID 当前任务的 taskID
	taskID string
}

func NewDirectorAgent(globalCtx *globalctx.GlobalCtx, engine llm.Engine, repo *RepoAgent, coding *CodingAgent, chat *ChatAgent, meta *MetaAgent, devops *DevOpsAgent, browser *BrowserAgent, maxSteps int, disabledAgents map[string]bool, metaRetryCount int, cfg config.Config, llmClient *llm.Client) *DirectorAgent {
	// self-reference for closures that need the DirectorAgent after construction
	var self *DirectorAgent

	delegateRepo := tools.NewAdapter("delegate_repo", "Delegate analysis task to Repo-Agent", func(ctx context.Context, params map[string]interface{}) (interface{}, error) {
		task, ok := params["task"].(string)
		if !ok {
			return nil, fmt.Errorf("task parameter required")
		}
		// 创建 Rollout Writer 并注入到 context
		if rolloutWriter := self.createRolloutWriter("repo", task); rolloutWriter != nil {
			defer rolloutWriter.Close()
			ctx = memory.WithRolloutWriter(ctx, rolloutWriter)
		}
		result, err := repo.Run(ctx, task)
		// 使用增强型 Commander 处理结果（压缩 + 注册）
		return self.applyEnhancedCommander("repo", task, result, err)
	}).WithSchema(map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"task": map[string]interface{}{"type": "string", "description": "The task description for Repo-Agent"},
		},
		"required": []string{"task"},
	})

	delegateCoding := tools.NewAdapter("delegate_coding", "Delegate coding task to Coding-Agent", func(ctx context.Context, params map[string]interface{}) (interface{}, error) {
		task, ok := params["task"].(string)
		if !ok {
			return nil, fmt.Errorf("task parameter required")
		}
		// 创建 Rollout Writer 并注入到 context
		if rolloutWriter := self.createRolloutWriter("coding", task); rolloutWriter != nil {
			defer rolloutWriter.Close()
			ctx = memory.WithRolloutWriter(ctx, rolloutWriter)
		}
		// RepoSummary is no longer injected into the task here — it is now passed
		// via ExecutorConfig.RepoContext and appended to the sub-agent's system prompt,
		// keeping the user message (task) variable and the system prompt cacheable.
		result, err := coding.Run(ctx, task)
		// 使用增强型 Commander 处理结果（压缩 + 注册）
		return self.applyEnhancedCommander("coding", task, result, err)
	}).WithSchema(map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"task": map[string]interface{}{"type": "string", "description": "The task description for Coding-Agent"},
		},
		"required": []string{"task"},
	})

	delegateChat := tools.NewAdapter("delegate_chat", "Delegate general conversation, explanation, or non-coding tasks to Chat-Agent", func(ctx context.Context, params map[string]interface{}) (interface{}, error) {
		task, ok := params["task"].(string)
		if !ok {
			return nil, fmt.Errorf("task parameter required")
		}
		// 创建 Rollout Writer 并注入到 context
		if rolloutWriter := self.createRolloutWriter("chat", task); rolloutWriter != nil {
			defer rolloutWriter.Close()
			ctx = memory.WithRolloutWriter(ctx, rolloutWriter)
		}
		result, err := chat.Run(ctx, task)
		// 使用增强型 Commander 处理结果（压缩 + 注册）
		return self.applyEnhancedCommander("chat", task, result, err)
	}).WithSchema(map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"task": map[string]interface{}{"type": "string", "description": "The message or question for Chat-Agent"},
		},
		"required": []string{"task"},
	})

	delegateDevOps := tools.NewAdapter("delegate_devops", "Delegate operational and system administration tasks to DevOps-Agent. DevOps-Agent can run shell commands, inspect files, check logs, manage processes, and perform any non-coding infrastructure work. Use this for tasks like checking disk usage, finding files, running diagnostics, inspecting configurations, or executing ad-hoc shell commands.", func(ctx context.Context, params map[string]interface{}) (interface{}, error) {
		task, ok := params["task"].(string)
		if !ok {
			return nil, fmt.Errorf("task parameter required")
		}
		// 创建 Rollout Writer 并注入到 context
		if rolloutWriter := self.createRolloutWriter("devops", task); rolloutWriter != nil {
			defer rolloutWriter.Close()
			ctx = memory.WithRolloutWriter(ctx, rolloutWriter)
		}
		result, err := devops.Run(ctx, task)
		// 使用增强型 Commander 处理结果（压缩 + 注册）
		return self.applyEnhancedCommander("devops", task, result, err)
	}).WithSchema(map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"task": map[string]interface{}{"type": "string", "description": "The operational task for DevOps-Agent, e.g., 'check disk usage', 'find all log files modified today', 'check if port 8080 is in use'."},
		},
		"required": []string{"task"},
	})

	delegateBrowser := tools.NewAdapter("delegate_browser",
		"Delegate browser automation tasks to Browser-Agent. Browser-Agent controls a headless Chrome browser using go-rod to navigate websites, click elements, fill forms, extract data, take screenshots, generate PDFs, execute JavaScript (with user confirmation), and manage cookies. Use this for tasks like: 'screenshot https://example.com', 'extract text from https://example.com/article', 'fill and submit the login form at https://example.com/login', 'check if website is reachable', 'get the current URL after navigation'. The agent handles all browser lifecycle and page management internally.",
		func(ctx context.Context, params map[string]interface{}) (interface{}, error) {
			task, ok := params["task"].(string)
			if !ok {
				return nil, fmt.Errorf("task parameter required")
			}
			// 创建 Rollout Writer 并注入到 context
			if rolloutWriter := self.createRolloutWriter("browser", task); rolloutWriter != nil {
				defer rolloutWriter.Close()
				ctx = memory.WithRolloutWriter(ctx, rolloutWriter)
			}
			// RepoSummary is no longer injected into the task here — it is now passed
			// via ExecutorConfig.RepoContext and appended to the sub-agent's system prompt.
			result, err := browser.Run(ctx, task)
			// 使用增强型 Commander 处理结果（压缩 + 注册）
			return self.applyEnhancedCommander("browser", task, result, err)
		}).WithSchema(map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"task": map[string]interface{}{
				"type":        "string",
				"description": "The browser automation task for Browser-Agent, e.g., 'screenshot https://example.com homepage', 'extract article text from https://example.com/blog/post-1', 'fill the login form and submit', 'navigate to https://example.com and return the page title'.",
			},
		},
		"required": []string{"task"},
	})

	delegateMeta := tools.NewAdapter("delegate_meta", "Delegate to Meta-Agent to DESIGN a custom specialized agent. Meta-Agent will craft a tailored system prompt using prompt engineering best practices and select appropriate tools. The designed agent is automatically registered and immediately executed to complete the task. After this, the new agent becomes a permanent delegate tool for future use.", func(ctx context.Context, params map[string]interface{}) (interface{}, error) {
		task, ok := params["task"].(string)
		if !ok {
			return nil, fmt.Errorf("task parameter required")
		}
		// 创建 Rollout Writer 并注入到 context
		if rolloutWriter := self.createRolloutWriter("meta", task); rolloutWriter != nil {
			defer rolloutWriter.Close()
			ctx = memory.WithRolloutWriter(ctx, rolloutWriter)
		}
		slog.Info("Director delegating to Meta-Agent (design)", "task", task)

		maxRetries := self.metaRetryCount
		var lastRawOutput string

		for attempt := 0; attempt < maxRetries; attempt++ {
			retryTask := task
			if attempt > 0 {
				retryTask = fmt.Sprintf(
					"%s\n\n[FORMAT CORRECTION — Attempt %d/%d]\nYour previous output was NOT valid JSON or missing required fields. You MUST output ONLY a valid JSON object with these exact top-level keys:\n{\n  \"thinking\": \"...\",\n  \"agent_name\": \"...\",\n  \"agent_design\": \"...\",\n  \"tools_used\": [...],\n  \"task_for_agent\": \"...\"\n}\n\nDo NOT wrap in markdown code fences (```). Do NOT include any text outside the JSON object.",
					task, attempt, maxRetries-1,
				)
			}

			metaResult, err := meta.Run(ctx, retryTask)
			if err != nil {
				return nil, fmt.Errorf("Meta-Agent design failed: %w", err)
			}
			// 存储 meta-agent memory（如果后续执行了自定义 agent，会被覆盖为自定义 agent 的 memory）
			self.pendingSubAgentMemory = &metaResult
			lastRawOutput = metaResult.Text

			systemPrompt, execResult, parseErr := parseMetaAgentOutput(metaResult.Text)
			if parseErr != nil {
				slog.Warn("Meta-Agent JSON parse failed, retrying", "attempt", attempt+1, "maxRetries", maxRetries, "error", parseErr)
				continue
			}

			// ── Parse succeeded ──
			// Register the newly designed agent if it has a valid name and prompt
			if execResult.AgentName != "" && systemPrompt != "" {
				snakeName := toSnakeCase(execResult.AgentName)
				customAgent := &CustomAgent{
					Name:         snakeName,
					DisplayName:  execResult.AgentName,
					SystemPrompt: systemPrompt,
					ToolsUsed:    execResult.ToolsUsed,
					Description:  fmt.Sprintf("Custom agent designed for: %s. Uses tools: %s.", execResult.AgentName, strings.Join(execResult.ToolsUsed, ", ")),
				}
				self.registerCustomAgent(customAgent)

				// Use task_for_agent (clean task without meta-design instructions) if available;
				// otherwise fall back to the original task.
				agentTask := execResult.TaskForAgent
				if agentTask == "" {
					agentTask = task
				}

				// ── Immediately execute the newly registered agent ──
				// Find the just-created delegate tool and call it
				delegateName := "delegate_" + snakeName
				for _, ad := range self.Adapters {
					if ad.Name() == delegateName {
						slog.Info("Director executing newly designed agent", "delegate", delegateName, "display_name", execResult.AgentName)
						callResult, callErr := ad.Call(ctx, fmt.Sprintf(`{"task": %q}`, agentTask))
						if callErr != nil {
							return nil, fmt.Errorf("new agent %s execution failed: %w", execResult.AgentName, callErr)
						}
						// ad.Call returns JSON-encoded string, unmarshal to get the raw result
						var rawResult string
						if err := json.Unmarshal([]byte(callResult), &rawResult); err != nil {
							rawResult = callResult
						}
						formattedResult := fmt.Sprintf(
							"[Meta-Agent: Agent Designed and Executed]\nDesigned Agent: %s\nTools: %s\n\n[Execution Result]\n%s\n\n[New Agent Registered]\nA new specialized agent \"%s\" is now available via the `%s` tool for future tasks of this type.",
							execResult.AgentName,
							strings.Join(execResult.ToolsUsed, ", "),
							rawResult,
							execResult.AgentName,
							delegateName,
						)
						return formattedResult, nil
					}
				}
				return nil, fmt.Errorf("newly registered agent %s not found in adapters", delegateName)
			}

			// Parse succeeded but no agent to register
			return fmt.Sprintf("[Meta-Agent Design Result]\nAgent could not be registered (missing name or design). Raw output: %s", metaResult.Text), nil
		}

		// All retries exhausted
		slog.Warn("Meta-Agent JSON parse failed after all retries, returning raw output")
		return lastRawOutput, nil
	}).WithSchema(map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"task": map[string]interface{}{"type": "string", "description": "Detailed task description for Meta-Agent. Include: what needs to be accomplished, why existing agents are insufficient, and what the expected output format should be."},
		},
		"required": []string{"task"},
	})

	adapters := []*tools.Adapter{
		tools.NewAdapter("agent_exit", "Exit the agent with a reason. Use this when you are done — whether the task completed successfully, failed, needs clarification, or must be terminated. The reason must explain WHY the agent is exiting.", globalCtx.FlowOps.ExecuteAgentExit).WithSchema(map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"reason": map[string]interface{}{"type": "string", "description": "The reason the agent is exiting, e.g., task completed, cannot proceed, blocked by missing information, or must terminate."},
			},
			"required": []string{"reason"},
		}),
	}

	var toolDefs []tools.ToolDefinition
	if err := json.Unmarshal(ToolsJSON, &toolDefs); err != nil {
		slog.Error("Failed to unmarshal tools", "error", err)
	}

	// Build a map from tool name to definition for later use by custom agents
	toolDefMap := make(map[string]tools.ToolDefinition, len(toolDefs))
	for _, def := range toolDefs {
		toolDefMap[def.Name] = def
	}

	for _, def := range toolDefs {
		var fn tools.ToolFunc
		switch def.Name {
		case "search_by_regex":
			fn = globalCtx.SearchOps.ExecuteGrepSearch
		case "list_dir":
			fn = globalCtx.FileOps.ExecuteListDir
		case "read_file":
			fn = globalCtx.FileOps.ExecuteReadFile
		case "print_dir_tree":
			fn = globalCtx.FileOps.ExecutePrintDirTree
		case "deepthinking":
			fn = globalCtx.DeepThinkingTool.Execute
		case "ask_user_for_help":
			if globalCtx.FullYoloMode {
				continue
			}
			fn = globalCtx.FlowOps.ExecuteAskUserForHelp
		default:
			continue
		}

		adapter := tools.NewAdapter(def.Name, def.Description, fn).WithSchema(def.Parameters)
		adapters = append(adapters, adapter)
	}

	// Conditionally register delegate tools based on disabledAgents
	var delegateAdapters []*tools.Adapter
	if !disabledAgents["repo"] {
		delegateAdapters = append(delegateAdapters, delegateRepo)
	}
	if !disabledAgents["coding"] {
		delegateAdapters = append(delegateAdapters, delegateCoding)
	}
	if !disabledAgents["chat"] {
		delegateAdapters = append(delegateAdapters, delegateChat)
	}
	if !disabledAgents["meta"] {
		delegateAdapters = append(delegateAdapters, delegateMeta)
	}
	if !disabledAgents["devops"] {
		delegateAdapters = append(delegateAdapters, delegateDevOps)
	}
	if !disabledAgents["browser"] {
		delegateAdapters = append(delegateAdapters, delegateBrowser)
	}

	// Set workspace guard on all adapters (delegate adapters are not dangerous tools)
	tools.SetGuardOnAdapters(adapters, globalCtx.Guard)
	tools.SetGuardOnAdapters(delegateAdapters, globalCtx.Guard)

	// 注册知识整理/维护工具（需要 llm engine + CodeSeekMCP）
	knowledgeAdapters := createKnowledgeToolAdapters(globalCtx, engine, "", "")
	var allAdapters []*tools.Adapter
	if len(knowledgeAdapters) > 0 {
		tools.SetGuardOnAdapters(knowledgeAdapters, globalCtx.Guard)
		allAdapters = append(adapters, delegateAdapters...)
		allAdapters = append(allAdapters, knowledgeAdapters...)
	} else {
		allAdapters = append(adapters, delegateAdapters...)
	}

	// Strangler Fig: 创建适配器桥接层，开始使用新组件（Metrics + CircuitBreaker）
	adapterCfg := director.DefaultRecoveryConfig()
	adapterCfg.MaxRetries = maxSteps // 使用 maxSteps 作为重试次数
	adapterCfg.CircuitBreakerThreshold = cfg.LLM.CircuitBreakerThreshold
	adapterCfg.CircuitBreakerResetTimeout = cfg.LLM.CircuitBreakerResetTimeout
	directorAdapter := NewDirectorAdapter(true, adapterCfg) // enabled=true 启动 Metrics

	self = &DirectorAgent{
		BaseAgent:          BaseAgent{LLM: engine, Publisher: globalCtx.Publisher},
		RepoAgent:          repo,
		CodingAgent:        coding,
		ChatAgent:          chat,
		MetaAgent:          meta,
		DevOpsAgent:        devops,
		BrowserAgent:       browser,
		GlobalCtx:          globalCtx,
		Adapters:           allAdapters,
		maxSteps:           maxSteps,
		metaRetryCount:     metaRetryCount,
		toolDefMap:         toolDefMap,
		customAgents:       make(map[string]*CustomAgent),
		adapter:            directorAdapter,
		llmClient:          llmClient,
		hasDelegated:       false, // 初始未委派过
		delegationAttempts: 0,     // 委派尝试次数初始为0

		// LLM 兜底机制配置
		stepRetries: cfg.LLM.StepRetries,
		llmTimeout: func() time.Duration {
			if cfg.LLM.Timeout > 0 {
				return cfg.LLM.Timeout
			}
			return 5 * time.Minute
		}(),
		circuitBreakerThreshold:    cfg.LLM.CircuitBreakerThreshold,
		circuitBreakerResetTimeout: cfg.LLM.CircuitBreakerResetTimeout,
		// consecutiveLLMFailures 和 lastLLMFailureTime 保持零值即可

		// EnhancedCommander 配置
		EnhancedCommanderCfg: cfg.EnhancedCommander,
	}

	// 计算并记录 Tool Definitions 哈希，用于验证 Prompt Cache 一致性
	toolDefsForHash := make([]llm.ToolDef, len(allAdapters))
	for i, ad := range allAdapters {
		toolDefsForHash[i] = ad.ToToolDef()
	}
	toolHash := tools.ComputeToolDefsHash(toolDefsForHash)
	names := make([]string, len(toolDefsForHash))
	for i, td := range toolDefsForHash {
		names[i] = td.Function.Name
	}
	slog.Info("Director tool definitions initialized",
		"hash", toolHash,
		"tool_count", len(toolDefsForHash),
		"tool_names", names)

	return self
}

func (a *DirectorAgent) Name() string {
	return "Director"
}

// refreshSubAgentEngines 从 llmClient 刷新所有子 Agent 的引擎，
// 确保 TUI 中切换模型（全局或针对特定 agent）后立即生效。
func (a *DirectorAgent) refreshSubAgentEngines() {
	if a.llmClient == nil {
		return
	}
	if a.RepoAgent != nil {
		if e := a.llmClient.GetAgentEngine("repo"); e != nil {
			a.RepoAgent.LLM = e
		}
	}
	if a.CodingAgent != nil {
		if e := a.llmClient.GetAgentEngine("coding"); e != nil {
			a.CodingAgent.LLM = e
		}
	}
	if a.ChatAgent != nil {
		if e := a.llmClient.GetAgentEngine("chat"); e != nil {
			a.ChatAgent.LLM = e
		}
	}
	if a.MetaAgent != nil {
		if e := a.llmClient.GetAgentEngine("meta"); e != nil {
			a.MetaAgent.LLM = e
		}
	}
	if a.DevOpsAgent != nil {
		if e := a.llmClient.GetAgentEngine("devops"); e != nil {
			a.DevOpsAgent.LLM = e
		}
	}
	if a.BrowserAgent != nil {
		if e := a.llmClient.GetAgentEngine("browser"); e != nil {
			a.BrowserAgent.LLM = e
		}
	}
}
