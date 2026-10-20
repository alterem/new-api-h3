package openai

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/constant"
	relaycommon "github.com/QuantumNous/new-api/relay/common"
	relayconstant "github.com/QuantumNous/new-api/relay/constant"
	"github.com/QuantumNous/new-api/relaykit/dto"
	"github.com/QuantumNous/new-api/relaykit/types"
	"github.com/QuantumNous/new-api/service"
	"github.com/QuantumNous/new-api/setting/operation_setting"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOaiResponsesHandlerCountsOutputCallsNotDeclarations(t *testing.T) {
	gin.SetMode(gin.TestMode)
	operation_setting.SetToolPriceForTest("priced_fn", 5.0)
	t.Cleanup(func() {
		operation_setting.DeleteToolPriceForTest("priced_fn")
	})

	body, err := common.Marshal(dto.OpenAIResponsesResponse{
		Tools: []map[string]any{
			{"type": "web_search_preview"},
			{"type": "file_search"},
		},
		Output: []dto.ResponsesOutput{
			{Type: dto.BuildInCallWebSearchCall},
			{Type: dto.BuildInCallWebSearchCall},
			{Type: dto.BuildInCallFunctionCall, Name: "priced_fn"},
			{Type: dto.BuildInCallFunctionCall, Name: "unpriced_fn"},
		},
		Usage: &dto.Usage{InputTokens: 1, OutputTokens: 1, TotalTokens: 2},
	})
	require.NoError(t, err)

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)

	info := &relaycommon.RelayInfo{
		OriginModelName: "gpt-5.1",
		ResponsesUsageInfo: &relaycommon.ResponsesUsageInfo{
			BuiltInTools: map[string]*relaycommon.BuildInToolInfo{
				dto.BuildInToolWebSearchPreview: {ToolName: dto.BuildInToolWebSearchPreview, CallCount: 0},
				dto.BuildInToolFileSearch:       {ToolName: dto.BuildInToolFileSearch, CallCount: 0},
			},
		},
	}
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(bytes.NewReader(body)),
		Header:     http.Header{"Content-Type": []string{"application/json"}},
	}

	usage, apiErr := OaiResponsesHandler(c, info, resp)
	require.Nil(t, apiErr)
	require.NotNil(t, usage)
	assert.Equal(t, 2, info.ResponsesUsageInfo.BuiltInTools[dto.BuildInToolWebSearchPreview].CallCount)
	assert.Equal(t, 0, info.ResponsesUsageInfo.BuiltInTools[dto.BuildInToolFileSearch].CallCount)
	require.Contains(t, info.ResponsesUsageInfo.BuiltInTools, "priced_fn")
	assert.Equal(t, 1, info.ResponsesUsageInfo.BuiltInTools["priced_fn"].CallCount)
	assert.NotContains(t, info.ResponsesUsageInfo.BuiltInTools, "unpriced_fn")
}

func TestOaiResponsesHandlerDeclaredToolsWithoutOutputCountZero(t *testing.T) {
	gin.SetMode(gin.TestMode)

	body, err := common.Marshal(dto.OpenAIResponsesResponse{
		Tools: []map[string]any{
			{"type": "web_search_preview"},
			{"type": "file_search"},
		},
		Output: []dto.ResponsesOutput{
			{Type: "message", Role: "assistant"},
		},
		Usage: &dto.Usage{InputTokens: 1, OutputTokens: 1, TotalTokens: 2},
	})
	require.NoError(t, err)

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)

	info := &relaycommon.RelayInfo{
		OriginModelName: "gpt-5.1",
		ResponsesUsageInfo: &relaycommon.ResponsesUsageInfo{
			BuiltInTools: map[string]*relaycommon.BuildInToolInfo{
				dto.BuildInToolWebSearchPreview: {ToolName: dto.BuildInToolWebSearchPreview, CallCount: 0},
				dto.BuildInToolFileSearch:       {ToolName: dto.BuildInToolFileSearch, CallCount: 0},
			},
		},
	}
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(bytes.NewReader(body)),
		Header:     http.Header{"Content-Type": []string{"application/json"}},
	}

	_, apiErr := OaiResponsesHandler(c, info, resp)
	require.Nil(t, apiErr)
	assert.Equal(t, 0, info.ResponsesUsageInfo.BuiltInTools[dto.BuildInToolWebSearchPreview].CallCount)
	assert.Equal(t, 0, info.ResponsesUsageInfo.BuiltInTools[dto.BuildInToolFileSearch].CallCount)
}

func TestOaiResponsesHandlerCountsCompletedImageGenerationOutputs(t *testing.T) {
	gin.SetMode(gin.TestMode)

	body, err := common.Marshal(dto.OpenAIResponsesResponse{
		Status: []byte(`"completed"`),
		Output: []dto.ResponsesOutput{
			{
				Type:   dto.ResponsesOutputTypeImageGenerationCall,
				ID:     "img_1",
				Status: "completed",
				Result: "base64-a",
			},
			{
				Type:   dto.ResponsesOutputTypeImageGenerationCall,
				ID:     "img_2",
				Status: "completed",
				Result: "base64-b",
			},
			{
				Type:   dto.ResponsesOutputTypeImageGenerationCall,
				ID:     "img_empty",
				Status: "completed",
				Result: "",
			},
		},
		Usage: &dto.Usage{InputTokens: 1, OutputTokens: 1, TotalTokens: 2},
	})
	require.NoError(t, err)

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	info := &relaycommon.RelayInfo{OriginModelName: "gpt-5.1"}
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(bytes.NewReader(body)),
		Header:     http.Header{"Content-Type": []string{"application/json"}},
	}

	_, apiErr := OaiResponsesHandler(c, info, resp)
	require.Nil(t, apiErr)
	require.Contains(t, info.ResponsesUsageInfo.BuiltInTools, dto.BuildInToolImageGeneration)
	assert.Equal(t, 2, info.ResponsesUsageInfo.BuiltInTools[dto.BuildInToolImageGeneration].CallCount)
	assert.False(t, c.GetBool("image_generation_call"))
}

func TestOaiResponsesHandlerCountsImageGenerationOnIncompleteStatus(t *testing.T) {
	gin.SetMode(gin.TestMode)

	body, err := common.Marshal(dto.OpenAIResponsesResponse{
		Status: []byte(`"incomplete"`),
		Output: []dto.ResponsesOutput{
			{
				Type:   dto.ResponsesOutputTypeImageGenerationCall,
				ID:     "img_1",
				Status: "completed",
				Result: "base64-a",
			},
		},
		Usage: &dto.Usage{InputTokens: 1, OutputTokens: 1, TotalTokens: 2},
	})
	require.NoError(t, err)

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	info := &relaycommon.RelayInfo{
		OriginModelName: "gpt-5.1",
		ResponsesUsageInfo: &relaycommon.ResponsesUsageInfo{
			BuiltInTools: map[string]*relaycommon.BuildInToolInfo{
				dto.BuildInToolImageGeneration: {ToolName: dto.BuildInToolImageGeneration, CallCount: 0},
			},
		},
	}
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(bytes.NewReader(body)),
		Header:     http.Header{"Content-Type": []string{"application/json"}},
	}

	_, apiErr := OaiResponsesHandler(c, info, resp)
	require.Nil(t, apiErr)
	assert.Equal(t, 1, info.ResponsesUsageInfo.BuiltInTools[dto.BuildInToolImageGeneration].CallCount)
}

func runResponsesImageBillingStream(t *testing.T, events ...string) *relaycommon.RelayInfo {
	t.Helper()
	gin.SetMode(gin.TestMode)
	oldTimeout := constant.StreamingTimeout
	constant.StreamingTimeout = 30
	t.Cleanup(func() {
		constant.StreamingTimeout = oldTimeout
	})

	var body strings.Builder
	for _, event := range events {
		body.WriteString("data: ")
		body.WriteString(event)
		body.WriteString("\n\n")
	}
	body.WriteString("data: [DONE]\n\n")

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	c.Set(common.RequestIdKey, "responses-image-billing-test")
	info := &relaycommon.RelayInfo{
		OriginModelName: "gpt-5.1",
		DisablePing:     true,
		ChannelMeta: &relaycommon.ChannelMeta{
			UpstreamModelName: "gpt-5.1",
		},
	}
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(body.String())),
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
	}

	_, apiErr := OaiResponsesStreamHandler(c, info, resp)
	require.Nil(t, apiErr)
	require.NotNil(t, info.ResponsesUsageInfo)
	require.Contains(t, info.ResponsesUsageInfo.BuiltInTools, dto.BuildInToolImageGeneration)
	return info
}

func TestOaiResponsesStreamHandlerDeduplicatesCompletedImageOutput(t *testing.T) {
	item := `{"type":"image_generation_call","id":"img_1","call_id":"call_1","status":"completed","result":"base64-a"}`
	info := runResponsesImageBillingStream(
		t,
		`{"type":"response.output_item.done","output_index":0,"item":`+item+`}`,
		`{"type":"response.completed","response":{"status":"completed","output":[`+item+`],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}`,
	)

	assert.Equal(t, 1, info.ResponsesUsageInfo.BuiltInTools[dto.BuildInToolImageGeneration].CallCount)
}

func TestOaiResponsesStreamHandlerCountsImageOutputOnIncomplete(t *testing.T) {
	info := runResponsesImageBillingStream(
		t,
		`{"type":"response.output_item.done","output_index":0,"item":{"type":"image_generation_call","id":"img_1","status":"completed","result":"base64-a"}}`,
		`{"type":"response.incomplete","response":{"status":"incomplete"}}`,
	)

	assert.Equal(t, 1, info.ResponsesUsageInfo.BuiltInTools[dto.BuildInToolImageGeneration].CallCount)
}

func TestOaiResponsesStreamHandlerDoesNotCountPartialImageEvent(t *testing.T) {
	info := runResponsesImageBillingStream(
		t,
		`{"type":"response.image_generation_call.partial_image","output_index":0,"partial_image_b64":"partial-bytes"}`,
		`{"type":"response.completed","response":{"status":"completed","output":[]}}`,
	)

	assert.Equal(t, 0, info.ResponsesUsageInfo.BuiltInTools[dto.BuildInToolImageGeneration].CallCount)
}

func TestOaiResponsesHandlerRewritesSGLangCreatedAtToInt(t *testing.T) {
	gin.SetMode(gin.TestMode)

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)

	info := &relaycommon.RelayInfo{
		ChannelMeta: &relaycommon.ChannelMeta{ChannelType: constant.ChannelTypeSGLang},
	}
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(`{"id":"resp_1","object":"response","created_at":1786588600.0,"status":"completed","output":[]}`)),
		Header:     http.Header{"Content-Type": []string{"application/json"}},
	}

	_, apiErr := OaiResponsesHandler(c, info, resp)
	require.Nil(t, apiErr)
	assert.Contains(t, w.Body.String(), `"created_at":1786588600`)
	assert.NotContains(t, w.Body.String(), `1786588600.0`)
}

func TestOaiResponsesStreamHandlerRewritesSGLangCreatedAtToInt(t *testing.T) {
	gin.SetMode(gin.TestMode)
	oldTimeout := constant.StreamingTimeout
	constant.StreamingTimeout = 30
	t.Cleanup(func() {
		constant.StreamingTimeout = oldTimeout
	})

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	c.Set(common.RequestIdKey, "sglang-created-at-test")

	info := &relaycommon.RelayInfo{
		DisablePing: true,
		ChannelMeta: &relaycommon.ChannelMeta{
			ChannelType:       constant.ChannelTypeSGLang,
			UpstreamModelName: "served-model",
		},
	}
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Body: io.NopCloser(strings.NewReader(
			"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"created_at\":1.7865886E9,\"status\":\"completed\",\"output\":[]}}\n\n" +
				"data: [DONE]\n\n",
		)),
		Header: http.Header{"Content-Type": []string{"text/event-stream"}},
	}

	_, apiErr := OaiResponsesStreamHandler(c, info, resp)
	require.Nil(t, apiErr)
	assert.Contains(t, w.Body.String(), `"created_at":1786588600`)
	assert.NotContains(t, w.Body.String(), `1.7865886E9`)
	assert.NotContains(t, w.Body.String(), `1786588600.0`)
}

func TestOaiResponsesStreamHandlerKeepsNonSGLangCreatedAt(t *testing.T) {
	gin.SetMode(gin.TestMode)
	oldTimeout := constant.StreamingTimeout
	constant.StreamingTimeout = 30
	t.Cleanup(func() {
		constant.StreamingTimeout = oldTimeout
	})

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	c.Set(common.RequestIdKey, "openai-created-at-test")

	info := &relaycommon.RelayInfo{
		DisablePing: true,
		ChannelMeta: &relaycommon.ChannelMeta{ChannelType: constant.ChannelTypeOpenAI, UpstreamModelName: "gpt-test"},
	}
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Body: io.NopCloser(strings.NewReader(
			"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"created_at\":1786588600.0,\"status\":\"completed\",\"output\":[]}}\n\n" +
				"data: [DONE]\n\n",
		)),
		Header: http.Header{"Content-Type": []string{"text/event-stream"}},
	}

	_, apiErr := OaiResponsesStreamHandler(c, info, resp)
	require.Nil(t, apiErr)
	assert.Contains(t, w.Body.String(), `1786588600.0`)
}

// A converted hosted web search reaches Azure and Xinference Chat Completions
// as web_search_options, like OpenAI; the upstream decides whether it applies.
// Azure Responses bills web search by tool_usage.web_search.num_requests (Bing
// transactions) under bing_web_search, and the explicit zeros silence the
// web_search_call items, which leaves one surcharge row.
func TestAzureHostedWebSearch(t *testing.T) {
	gin.SetMode(gin.TestMode)
	oldTimeout := constant.StreamingTimeout
	constant.StreamingTimeout = 30
	t.Cleanup(func() {
		constant.StreamingTimeout = oldTimeout
	})

	t.Run("Claude to Chat", func(t *testing.T) {
		const body = `{"model":"gpt-5.1","max_tokens":64,"messages":[{"role":"user","content":"Weather?"}],"tools":[{"type":"web_search_20250305","name":"web_search"}]}`
		for _, channelType := range []int{constant.ChannelTypeAzure, constant.ChannelTypeXinference, constant.ChannelTypeOpenAI} {
			c, _ := gin.CreateTestContext(httptest.NewRecorder())
			info := &relaycommon.RelayInfo{
				RelayFormat: types.RelayFormatClaude,
				ChannelMeta: &relaycommon.ChannelMeta{ChannelType: channelType, UpstreamModelName: "gpt-5.1"},
			}
			adaptor := &Adaptor{}
			adaptor.Init(info)
			var request dto.ClaudeRequest
			require.NoError(t, common.UnmarshalJsonStr(body, &request))
			converted, err := adaptor.ConvertClaudeRequest(c, info, &request)
			require.NoError(t, err)
			wire, err := common.Marshal(converted)
			require.NoError(t, err)
			assert.True(t, strings.Contains(string(wire), `"web_search_options"`), channelType)
			var codes []string
			for _, diagnostic := range info.ConversionDiagnostics() {
				codes = append(codes, diagnostic.Code)
			}
			assert.False(t, slices.Contains(codes, "hosted_web_search_omitted"), channelType)
		}
	})

	const searchCall = `{"type":"web_search_call","status":"completed","action":{"type":"search","query":"go"}}`
	const usage = `"usage":{"input_tokens":10,"output_tokens":3,"total_tokens":13},"tool_usage":{"web_search":{"num_requests":5}}`
	output := strings.Repeat(searchCall+",", 3) + `{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Go 1.25"}]}`
	tests := []struct {
		name        string
		contentType string
		body        string
	}{
		{
			name:        "json",
			contentType: "application/json",
			body:        `{"id":"resp_1","object":"response","model":"gpt-5.1","status":"completed","output":[` + output + `],` + usage + `}`,
		},
		{
			name:        "stream",
			contentType: "text/event-stream",
			body: strings.Repeat(`data: {"type":"response.output_item.done","output_index":0,"item":`+searchCall+"}\n\n", 3) +
				`data: {"type":"response.completed","response":{"id":"resp_1","object":"response","model":"gpt-5.1","status":"completed","output":[` + output + `],` + usage + "}}\n\ndata: [DONE]\n\n",
		},
	}
	for _, tt := range tests {
		t.Run("Responses "+tt.name, func(t *testing.T) {
			c, _ := gin.CreateTestContext(httptest.NewRecorder())
			c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
			info := &relaycommon.RelayInfo{
				RelayMode:       relayconstant.RelayModeResponses,
				RelayFormat:     types.RelayFormatOpenAIResponses,
				OriginModelName: "gpt-5.1",
				IsStream:        tt.name == "stream",
				DisablePing:     true,
				ChannelMeta:     &relaycommon.ChannelMeta{ChannelType: constant.ChannelTypeAzure, UpstreamModelName: "gpt-5.1"},
			}
			adaptor := &Adaptor{}
			adaptor.Init(info)
			resp := &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(tt.body)),
				Header:     http.Header{"Content-Type": []string{tt.contentType}},
			}
			var apiErr *types.NewAPIError
			if info.IsStream {
				_, apiErr = OaiResponsesStreamHandler(c, info, resp)
			} else {
				_, apiErr = OaiResponsesHandler(c, info, resp)
			}
			require.Nil(t, apiErr)
			billed := map[string]int{}
			for name, tool := range info.ResponsesUsageInfo.BuiltInTools {
				if tool.CallCount > 0 {
					billed[name] = tool.CallCount
				}
			}
			assert.Equal(t, map[string]int{"bing_web_search": 5}, billed)
		})
	}
}

func TestClassifyResponsesStartupError(t *testing.T) {
	tests := []struct {
		name      string
		err       types.OpenAIError
		status    int
		skipRetry bool
	}{
		{name: "rate limit", err: types.OpenAIError{Code: "rate_limit_exceeded"}, status: http.StatusTooManyRequests},
		{name: "capacity", err: types.OpenAIError{Type: "overloaded_error"}, status: http.StatusServiceUnavailable},
		{name: "authentication", err: types.OpenAIError{Code: "invalid_api_key"}, status: http.StatusUnauthorized},
		{name: "permission", err: types.OpenAIError{Code: "permission_denied"}, status: http.StatusForbidden},
		{name: "invalid request", err: types.OpenAIError{Type: "invalid_request_error"}, status: http.StatusBadRequest, skipRetry: true},
		{name: "server", err: types.OpenAIError{Code: "server_error"}, status: http.StatusInternalServerError},
		{name: "numeric rate limit", err: types.OpenAIError{Code: 429}, status: http.StatusTooManyRequests},
		{name: "numeric capacity", err: types.OpenAIError{Code: 503}, status: http.StatusServiceUnavailable},
		{name: "numeric authentication", err: types.OpenAIError{Code: 401}, status: http.StatusUnauthorized},
		{name: "azure rate limit", err: types.OpenAIError{Code: "RateLimitReached"}, status: http.StatusTooManyRequests},
		{name: "azure too many requests", err: types.OpenAIError{Code: "TooManyRequests"}, status: http.StatusTooManyRequests},
		{name: "numeric invalid request", err: types.OpenAIError{Code: 400}, status: http.StatusBadRequest, skipRetry: true},
		{name: "unknown", err: types.OpenAIError{Code: "provider_specific_error"}, status: http.StatusBadGateway},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			status, skipRetry := classifyResponsesStartupError(&tt.err)
			assert.Equal(t, tt.status, status)
			assert.Equal(t, tt.skipRetry, skipRetry)
		})
	}
}
func TestOaiResponsesStartupGuardBuffersBeforeCommit(t *testing.T) {
	gin.SetMode(gin.TestMode)
	oldEnabled := constant.ResponsesStreamStartupGuardEnabled
	oldTimeout := constant.StreamingTimeout
	constant.ResponsesStreamStartupGuardEnabled = true
	constant.StreamingTimeout = 5
	t.Cleanup(func() {
		constant.ResponsesStreamStartupGuardEnabled = oldEnabled
		constant.StreamingTimeout = oldTimeout
	})

	sse := func(events ...string) string {
		return strings.Join(events, "\n\n") + "\n\n"
	}
	created := `event: response.created
data: {"type":"response.created","response":{"id":"r1","status":"in_progress"}}`
	inProgress := `event: response.in_progress
data: {"type":"response.in_progress","response":{"id":"r1","status":"in_progress"}}`
	failed := `event: response.failed
data: {"type":"response.failed","response":{"id":"r1","status":"failed","error":{"code":"rate_limit_exceeded","message":"limited"}}}`
	completed := `event: response.completed
data: {"type":"response.completed","response":{"id":"r1","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}`

	t.Run("startup failure returns retryable error without committing SSE", func(t *testing.T) {
		recorder := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(recorder)
		c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
		info := &relaycommon.RelayInfo{
			ChannelMeta: &relaycommon.ChannelMeta{ChannelType: constant.ChannelTypeAzure, UpstreamModelName: "gpt-test"},
			RelayMode:   relayconstant.RelayModeResponses,
			IsStream:    true,
			DisablePing: true,
		}
		resp := &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(sse(created, inProgress, failed))), Header: http.Header{"Content-Type": []string{"text/event-stream"}}}

		_, apiErr := OaiResponsesStreamHandler(c, info, resp)
		require.NotNil(t, apiErr)
		assert.Equal(t, http.StatusTooManyRequests, apiErr.StatusCode)
		assert.Empty(t, recorder.Body.String())
		assert.Empty(t, recorder.Header().Get("Content-Type"))
	})

	t.Run("successful startup replays buffered events", func(t *testing.T) {
		recorder := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(recorder)
		c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
		info := &relaycommon.RelayInfo{
			ChannelMeta: &relaycommon.ChannelMeta{ChannelType: constant.ChannelTypeOpenAI, UpstreamModelName: "gpt-test"},
			RelayMode:   relayconstant.RelayModeResponses,
			RelayFormat: types.RelayFormatOpenAIResponses,
			IsStream:    true,
			DisablePing: true,
		}
		resp := &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(sse(created, inProgress, completed) + "data: [DONE]\n\n")), Header: http.Header{"Content-Type": []string{"text/event-stream"}}}

		_, apiErr := OaiResponsesStreamHandler(c, info, resp)
		require.Nil(t, apiErr)
		assert.Equal(t, "text/event-stream", recorder.Header().Get("Content-Type"))
		assert.Contains(t, recorder.Body.String(), `"type":"response.created"`)
		assert.Contains(t, recorder.Body.String(), `"type":"response.completed"`)
	})

	t.Run("startup limits are bounded", func(t *testing.T) {
		events := make([]string, responsesStartupMaxEvents+1)
		for i := range events {
			events[i] = created
		}
		recorder := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(recorder)
		c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
		info := &relaycommon.RelayInfo{
			ChannelMeta: &relaycommon.ChannelMeta{ChannelType: constant.ChannelTypeOpenAI, UpstreamModelName: "gpt-test"},
			RelayMode:   relayconstant.RelayModeResponses,
			IsStream:    true,
			DisablePing: true,
		}
		resp := &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(sse(events...))), Header: http.Header{"Content-Type": []string{"text/event-stream"}}}

		_, apiErr := OaiResponsesStreamHandler(c, info, resp)
		require.NotNil(t, apiErr)
		assert.Equal(t, http.StatusBadGateway, apiErr.StatusCode)
		assert.Empty(t, recorder.Body.String())
	})
}

func TestResponsesStartupGuardScenarios(t *testing.T) {
	gin.SetMode(gin.TestMode)
	settings := operation_setting.GetGeneralSetting()
	oldGuard, oldTimeout := constant.ResponsesStreamStartupGuardEnabled, constant.StreamingTimeout
	oldPing, oldInterval := settings.PingIntervalEnabled, settings.PingIntervalSeconds
	constant.ResponsesStreamStartupGuardEnabled, constant.StreamingTimeout = true, 5
	t.Cleanup(func() {
		constant.ResponsesStreamStartupGuardEnabled, constant.StreamingTimeout = oldGuard, oldTimeout
		settings.PingIntervalEnabled, settings.PingIntervalSeconds = oldPing, oldInterval
	})
	newInfo := func(channelType int) *relaycommon.RelayInfo {
		return &relaycommon.RelayInfo{
			ChannelMeta: &relaycommon.ChannelMeta{ChannelType: channelType, UpstreamModelName: "gpt-test"},
			RelayMode:   relayconstant.RelayModeResponses,
			RelayFormat: types.RelayFormatOpenAIResponses,
			IsStream:    true,
			DisablePing: true,
		}
	}
	sseResponse := func(body string, header http.Header) *http.Response {
		return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(body)), Header: header}
	}
	created := "data: {\"type\":\"response.created\",\"response\":{\"id\":\"failed-attempt\",\"status\":\"in_progress\"}}\n\n"

	// Error shape reported in production: created, in_progress, top-level TPM error, then response.failed.
	reportedTPM := created +
		"data: {\"type\":\"response.in_progress\",\"response\":{\"id\":\"failed-attempt\",\"status\":\"in_progress\"}}\n\n" +
		`data: {"type":"error","error":{"type":"tokens","code":"rate_limit_exceeded","message":"Rate limit reached for gpt-test on tokens per min (TPM)","param":null}}` + "\n\n" +
		`data: {"type":"response.failed","response":{"id":"failed-attempt","status":"failed","output":[],"usage":null}}` + "\n\n"

	t.Run("reported TPM error leaks through when the guard is off", func(t *testing.T) {
		constant.ResponsesStreamStartupGuardEnabled = false
		t.Cleanup(func() { constant.ResponsesStreamStartupGuardEnabled = true })
		recorder := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(recorder)
		c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
		info := newInfo(constant.ChannelTypeOpenAI)

		_, apiErr := OaiResponsesStreamHandler(c, info, sseResponse(reportedTPM, make(http.Header)))
		require.Nil(t, apiErr)
		assert.Contains(t, recorder.Body.String(), "rate_limit_exceeded")
		assert.True(t, info.StreamStatus.ResponseFailed())
	})

	errorCases := []struct {
		name      string
		body      string
		status    int
		code      string
		skipRetry bool
	}{
		{"reported TPM error", reportedTPM, 429, "rate_limit_exceeded", false},
		{"flat error", `data: {"type":"error","code":"rate_limit_exceeded","message":"TPM exceeded"}` + "\n\n", 429, "rate_limit_exceeded", false},
		{"Azure numeric code", `data: {"type":"error","error":{"code":429,"message":"Rate limit reached"}}` + "\n\n", 429, "429", false},
		{"Azure named code", `data: {"type":"error","error":{"code":"RateLimitReached","message":"Rate limit reached"}}` + "\n\n", 429, "RateLimitReached", false},
		{"capacity", created + `data: {"type":"response.failed","response":{"status":"failed","error":{"code":"model_at_capacity","message":"Busy"}}}` + "\n\n", 503, "model_at_capacity", false},
		{"invalid key may fail over", created + `data: {"type":"response.failed","response":{"status":"failed","error":{"code":"invalid_api_key","message":"Bad key"}}}` + "\n\n", 401, "invalid_api_key", false},
		{"invalid input", created + `data: {"type":"response.failed","response":{"status":"failed","error":{"code":"context_length_exceeded","message":"Too long"}}}` + "\n\n", 400, "context_length_exceeded", true},
		{"premature EOF", created, 502, "startup_stream_ended", false},
		{"invalid JSON", "data: {invalid\n\n", 502, "startup_invalid_json", false},
	}
	for _, channelType := range []int{constant.ChannelTypeOpenAI, constant.ChannelTypeAzure} {
		for _, tt := range errorCases {
			t.Run(fmt.Sprintf("channel_%d/%s", channelType, tt.name), func(t *testing.T) {
				recorder := httptest.NewRecorder()
				c, _ := gin.CreateTestContext(recorder)
				c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
				info := newInfo(channelType)
				header := http.Header{"X-Codex-Turn-State": []string{"failed-state"}}

				usage, apiErr := OaiResponsesStreamHandler(c, info, sseResponse(tt.body, header))
				require.NotNil(t, apiErr)
				assert.Nil(t, usage)
				assert.Equal(t, tt.status, apiErr.StatusCode)
				assert.Equal(t, types.ErrorCode(tt.code), apiErr.GetErrorCode())
				assert.Equal(t, tt.skipRetry, types.IsSkipRetryError(apiErr))
				assert.Equal(t, !tt.skipRetry, service.ShouldRetryRelayError(c, apiErr, 1))
				assert.False(t, c.Writer.Written())
				assert.Empty(t, recorder.Body.String())
				assert.Empty(t, recorder.Header().Get("Content-Type"))
				assert.Empty(t, recorder.Header().Get("X-Codex-Turn-State"))
				assert.True(t, info.StreamStatus.ResponseFailed())
			})
		}
	}

	t.Run("retry exposes only the successful attempt", func(t *testing.T) {
		recorder := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(recorder)
		c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
		info := newInfo(constant.ChannelTypeOpenAI)
		limited := created + `data: {"type":"response.failed","response":{"status":"failed","error":{"code":"rate_limit_exceeded","message":"TPM exceeded"}}}` + "\n\n"
		succeeded := "data: {\"type\":\"response.created\",\"response\":{\"id\":\"successful-attempt\",\"status\":\"in_progress\"}}\n\n" +
			"data: {\"type\":\"response.output_text.delta\",\"delta\":\"OK\"}\n\n" +
			"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"successful-attempt\",\"status\":\"completed\",\"usage\":{\"input_tokens\":10,\"output_tokens\":2,\"total_tokens\":12}}}\n\n"

		_, apiErr := OaiResponsesStreamHandler(c, info, sseResponse(limited, http.Header{"X-Reasoning-Included": []string{"failed-state"}}))
		require.NotNil(t, apiErr)
		require.True(t, service.ShouldRetryRelayError(c, apiErr, 1))
		require.False(t, c.Writer.Written())

		usage, apiErr := OaiResponsesStreamHandler(c, info, sseResponse(succeeded, http.Header{"X-Reasoning-Included": []string{"success-state"}}))
		require.Nil(t, apiErr)
		require.NotNil(t, usage)
		assert.Equal(t, 10, usage.PromptTokens)
		assert.Equal(t, 2, usage.CompletionTokens)
		assert.NotContains(t, recorder.Body.String(), "failed-attempt")
		assert.NotContains(t, recorder.Body.String(), "TPM exceeded")
		assert.Contains(t, recorder.Body.String(), "successful-attempt")
		assert.Contains(t, recorder.Body.String(), `"delta":"OK"`)
		assert.Equal(t, []string{"success-state"}, recorder.Header().Values("X-Reasoning-Included"))
		assert.Equal(t, "text/event-stream", recorder.Header().Get("Content-Type"))
	})

	t.Run("failure after output is not replayed", func(t *testing.T) {
		recorder := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(recorder)
		c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
		info := newInfo(constant.ChannelTypeOpenAI)
		body := "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Partial\"}\n\n" +
			"data: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",\"error\":{\"code\":\"rate_limit_exceeded\",\"message\":\"Late failure\"},\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\n"

		_, apiErr := OaiResponsesStreamHandler(c, info, sseResponse(body, make(http.Header)))
		assert.Nil(t, apiErr)
		assert.Contains(t, recorder.Body.String(), "Partial")
		assert.Contains(t, recorder.Body.String(), "Late failure")
		assert.True(t, info.StreamStatus.ResponseFailed())
	})

	t.Run("terminal event ends the stream while upstream stays open", func(t *testing.T) {
		for _, guard := range []bool{false, true} {
			constant.ResponsesStreamStartupGuardEnabled = guard
			recorder := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(recorder)
			c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
			info := newInfo(constant.ChannelTypeAzure)
			reader, writer := io.Pipe()
			go func() {
				_, _ = io.WriteString(writer, "data: {\"type\":\"response.created\",\"response\":{\"id\":\"r1\",\"status\":\"in_progress\"}}\n\n")
				_, _ = io.WriteString(writer, "data: {\"type\":\"response.output_text.delta\",\"delta\":\"OK\"}\n\n")
				_, _ = io.WriteString(writer, "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"r1\",\"status\":\"completed\",\"usage\":{\"input_tokens\":10,\"output_tokens\":2,\"total_tokens\":12}}}\n\n")
			}()

			usage, apiErr := OaiResponsesStreamHandler(c, info, &http.Response{StatusCode: http.StatusOK, Body: reader, Header: make(http.Header)})
			_ = writer.Close()
			require.Nil(t, apiErr, "guard=%v", guard)
			require.NotNil(t, usage)
			assert.Equal(t, relaycommon.StreamEndReasonDone, info.StreamStatus.EndReason, "guard=%v", guard)
			assert.False(t, info.StreamStatus.HasErrors(), "guard=%v", guard)
			assert.Equal(t, 10, usage.PromptTokens)
			assert.Equal(t, 2, usage.CompletionTokens)
			assert.Contains(t, recorder.Body.String(), `"type":"response.completed"`)
		}
		constant.ResponsesStreamStartupGuardEnabled = true
	})

	t.Run("client cancellation during startup sends no ping", func(t *testing.T) {
		settings.PingIntervalEnabled, settings.PingIntervalSeconds = true, 1
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		recorder := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(recorder)
		c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil).WithContext(ctx)
		info := newInfo(constant.ChannelTypeOpenAI)
		info.DisablePing = false
		reader, writer := io.Pipe()
		defer writer.Close()
		go func() {
			_, _ = io.WriteString(writer, created)
		}()

		_, apiErr := OaiResponsesStreamHandler(c, info, &http.Response{StatusCode: http.StatusOK, Body: reader, Header: make(http.Header)})
		require.NotNil(t, apiErr)
		assert.Equal(t, http.StatusRequestTimeout, apiErr.StatusCode)
		assert.True(t, types.IsSkipRetryError(apiErr))
		assert.False(t, c.Writer.Written())
		assert.Empty(t, recorder.Body.String())
	})
}
