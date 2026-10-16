package openai

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/constant"
	relaycommon "github.com/QuantumNous/new-api/relay/common"
	relayconstant "github.com/QuantumNous/new-api/relay/constant"
	"github.com/QuantumNous/new-api/relaykit/types"
	"github.com/QuantumNous/new-api/service"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// An upstream that answers HTTP 200 with a rate limit error must fail the
// attempt with 429 before anything reaches the client, so the controller
// retries another channel instead of logging a zero-cost success.
func TestDoResponseTreatsRateLimitInSuccessBodyAsRetryable429(t *testing.T) {
	gin.SetMode(gin.TestMode)
	oldTimeout := constant.StreamingTimeout
	constant.StreamingTimeout = 30
	t.Cleanup(func() { constant.StreamingTimeout = oldTimeout })

	const tpm = `rate limit exceeded: Rate limit reached for gpt-6.1-sol in organization org-test-upstream-limited on tokens per min (TPM): Limit 40000000, Used 39920237, Requested 203475. Please try again in 185ms. Visit https://platform.openai.com/account/rate-limits to learn more.`
	chatError := `{"error":{"message":"` + tpm + `","type":"tokens","param":null,"code":"rate_limit_exceeded"}}`
	messageOnly := `{"error":{"message":"` + tpm + `"}}`
	sse := func(frames ...string) string {
		var b strings.Builder
		for _, frame := range frames {
			b.WriteString("data: " + frame + "\n\n")
		}
		b.WriteString("data: [DONE]\n\n")
		return b.String()
	}
	chatChunk := `{"id":"c1","object":"chat.completion.chunk","model":"gpt-6-astra","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}`

	tests := []struct {
		name        string
		mode        int
		format      types.RelayFormat
		stream      bool
		contentType string
		body        string
		wantStatus  int // 0 means the attempt succeeds
	}{
		{name: "chat json", mode: relayconstant.RelayModeChatCompletions, format: types.RelayFormatOpenAI, contentType: "application/json", body: chatError, wantStatus: 429},
		{name: "chat json without type", mode: relayconstant.RelayModeChatCompletions, format: types.RelayFormatOpenAI, contentType: "application/json", body: messageOnly, wantStatus: 429},
		{name: "chat json other error keeps status", mode: relayconstant.RelayModeChatCompletions, format: types.RelayFormatOpenAI, contentType: "application/json", body: `{"error":{"message":"bad","type":"invalid_request_error"}}`, wantStatus: 200},
		{name: "chat stream first frame", mode: relayconstant.RelayModeChatCompletions, format: types.RelayFormatOpenAI, stream: true, contentType: "text/event-stream", body: sse(chatError), wantStatus: 429},
		{name: "chat stream plain text first frame", mode: relayconstant.RelayModeChatCompletions, format: types.RelayFormatOpenAI, stream: true, contentType: "text/event-stream", body: sse(tpm), wantStatus: 429},
		{name: "chat stream string error first frame", mode: relayconstant.RelayModeChatCompletions, format: types.RelayFormatOpenAI, stream: true, contentType: "text/event-stream", body: sse(`{"error":"` + tpm + `"}`), wantStatus: 429},
		{name: "chat stream top-level message first frame", mode: relayconstant.RelayModeChatCompletions, format: types.RelayFormatOpenAI, stream: true, contentType: "text/event-stream", body: sse(`{"message":"` + tpm + `"}`), wantStatus: 429},
		{name: "chat stream json body", mode: relayconstant.RelayModeChatCompletions, format: types.RelayFormatOpenAI, stream: true, contentType: "application/json", body: chatError, wantStatus: 429},
		{name: "chat stream json top-level message", mode: relayconstant.RelayModeChatCompletions, format: types.RelayFormatOpenAI, stream: true, contentType: "application/json", body: `{"message":"` + tpm + `"}`, wantStatus: 429},
		{name: "chat stream json plain error", mode: relayconstant.RelayModeChatCompletions, format: types.RelayFormatOpenAI, stream: true, contentType: "application/json", body: tpm, wantStatus: 429},
		{name: "chat stream plain text body", mode: relayconstant.RelayModeChatCompletions, format: types.RelayFormatOpenAI, stream: true, contentType: "text/plain", body: tpm, wantStatus: 429},
		{name: "chat stream normal", mode: relayconstant.RelayModeChatCompletions, format: types.RelayFormatOpenAI, stream: true, contentType: "text/event-stream", body: sse(chatChunk)},
		{name: "chat stream rate limit text after output", mode: relayconstant.RelayModeChatCompletions, format: types.RelayFormatOpenAI, stream: true, contentType: "text/event-stream", body: sse(chatChunk, chatError)},
		{name: "responses json", mode: relayconstant.RelayModeResponses, format: types.RelayFormatOpenAIResponses, contentType: "application/json", body: chatError, wantStatus: 429},
		{name: "responses stream error event", mode: relayconstant.RelayModeResponses, format: types.RelayFormatOpenAIResponses, stream: true, contentType: "text/event-stream", body: sse(`{"type":"error","code":"rate_limit_exceeded","message":"` + tpm + `"}`), wantStatus: 429},
		{name: "responses stream plain text first frame", mode: relayconstant.RelayModeResponses, format: types.RelayFormatOpenAIResponses, stream: true, contentType: "text/event-stream", body: sse(tpm), wantStatus: 429},
		{name: "responses stream failed event", mode: relayconstant.RelayModeResponses, format: types.RelayFormatOpenAIResponses, stream: true, contentType: "text/event-stream", body: sse(`{"type":"response.failed","response":{"id":"r1","status":"failed","error":{"code":"rate_limit_exceeded","message":"` + tpm + `"}}}`), wantStatus: 429},
		{name: "responses stream json body", mode: relayconstant.RelayModeResponses, format: types.RelayFormatOpenAIResponses, stream: true, contentType: "application/json", body: chatError, wantStatus: 429},
		{name: "responses stream normal", mode: relayconstant.RelayModeResponses, format: types.RelayFormatOpenAIResponses, stream: true, contentType: "text/event-stream", body: sse(`{"type":"response.completed","response":{"id":"r1","status":"completed","output":[]}}`)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(recorder)
			c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
			c.Set(common.RequestIdKey, "rate-limit-test")
			info := &relaycommon.RelayInfo{
				RelayMode:       tt.mode,
				RelayFormat:     tt.format,
				OriginModelName: "gpt-6-astra",
				IsStream:        tt.stream,
				DisablePing:     true,
				ChannelMeta:     &relaycommon.ChannelMeta{ChannelType: constant.ChannelTypeOpenAI, UpstreamModelName: "gpt-6-astra"},
			}
			adaptor := &Adaptor{}
			adaptor.Init(info)
			resp := &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(tt.body)),
				Header:     http.Header{"Content-Type": []string{tt.contentType}},
			}

			_, apiErr := adaptor.DoResponse(c, resp, info)

			if tt.wantStatus == 0 {
				require.Nil(t, apiErr)
				return
			}
			require.NotNil(t, apiErr)
			assert.Equal(t, tt.wantStatus, apiErr.StatusCode)
			if tt.wantStatus == 429 {
				assert.Contains(t, apiErr.Error(), "Rate limit reached", "upstream message is kept")
				assert.Contains(t, apiErr.Error(), "org-test-upstream-limited")
			}
			assert.Empty(t, recorder.Body.String(), "nothing reaches the client before the retry decision")
			assert.Equal(t, tt.wantStatus == 429, service.ShouldRetryRelayError(c, apiErr, 1))
		})
	}
}

// Conversion paths forward the first upstream event to the client in another
// format; a rate limit on it must surface as a retryable 429 instead of a
// converted failure event.
func TestConvertedStreamsTreatFirstFrameRateLimitAsRetryable429(t *testing.T) {
	gin.SetMode(gin.TestMode)
	oldTimeout := constant.StreamingTimeout
	constant.StreamingTimeout = 30
	t.Cleanup(func() { constant.StreamingTimeout = oldTimeout })

	rateLimit := `{"type":"error","code":"rate_limit_exceeded","message":"Rate limit reached for gpt-6-astra on tokens per min"}`
	chatError := `{"error":{"message":"Rate limit reached for gpt-6-astra on tokens per min","type":"tokens","code":"rate_limit_exceeded"}}`
	handlers := map[string]struct {
		format  types.RelayFormat
		body    string
		handler func(*gin.Context, *relaycommon.RelayInfo, *http.Response) (any, *types.NewAPIError)
	}{
		"responses to chat stream": {types.RelayFormatOpenAI, "data: " + rateLimit + "\n\n", func(c *gin.Context, i *relaycommon.RelayInfo, r *http.Response) (any, *types.NewAPIError) {
			return OaiResponsesToChatStreamHandler(c, i, r)
		}},
		"responses to chat buffered": {types.RelayFormatOpenAI, "data: " + rateLimit + "\n\n", func(c *gin.Context, i *relaycommon.RelayInfo, r *http.Response) (any, *types.NewAPIError) {
			return OaiResponsesToChatBufferedStreamHandler(c, i, r)
		}},
		"chat to responses stream": {types.RelayFormatOpenAIResponses, "data: " + chatError + "\n\n", func(c *gin.Context, i *relaycommon.RelayInfo, r *http.Response) (any, *types.NewAPIError) {
			return OaiChatToResponsesStreamHandler(c, i, r)
		}},
	}
	for name, tt := range handlers {
		t.Run(name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(recorder)
			c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
			info := &relaycommon.RelayInfo{
				RelayFormat: tt.format,
				IsStream:    true,
				DisablePing: true,
				ChannelMeta: &relaycommon.ChannelMeta{ChannelType: constant.ChannelTypeOpenAI, UpstreamModelName: "gpt-6-astra"},
			}
			resp := &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(tt.body)),
				Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
			}

			_, apiErr := tt.handler(c, info, resp)

			require.NotNil(t, apiErr)
			assert.Equal(t, http.StatusTooManyRequests, apiErr.StatusCode)
			assert.Empty(t, recorder.Body.String())
			assert.True(t, service.ShouldRetryRelayError(c, apiErr, 1))
		})
	}
}
