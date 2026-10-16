package openai

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/relaykit/dto"
	"github.com/QuantumNous/new-api/relaykit/types"
)

// maxJSONStreamErrorBodyBytes bounds how much of a JSON body answering a
// stream request is inspected for an error envelope; error bodies are small.
const maxJSONStreamErrorBodyBytes = 64 << 10

// upstreamBodyError converts an error envelope found in a response body into a
// relay error. Some OpenAI-compatible upstreams answer HTTP 200 with such an
// envelope; a rate limit is then reported as 429 so the request is retried on
// another channel instead of ending as a failure with a success status.
func upstreamBodyError(oaiError *types.OpenAIError, statusCode int) *types.NewAPIError {
	if oaiError == nil {
		return nil
	}
	code := ""
	if oaiError.Code != nil {
		code = strings.TrimSpace(fmt.Sprint(oaiError.Code))
	}
	if oaiError.Type == "" && strings.TrimSpace(oaiError.Message) == "" && code == "" {
		return nil
	}
	if statusCode/100 == 2 && isRateLimitError(oaiError) {
		statusCode = http.StatusTooManyRequests
	}
	return types.WithOpenAIError(*oaiError, statusCode)
}

func isRateLimitError(oaiError *types.OpenAIError) bool {
	if oaiError == nil {
		return false
	}
	code := ""
	if oaiError.Code != nil {
		code = strings.TrimSpace(fmt.Sprint(oaiError.Code))
	}
	if code == "429" {
		return true
	}
	text := strings.ToLower(oaiError.Type + " " + code + " " + oaiError.Message)
	return strings.Contains(text, "rate_limit") ||
		strings.Contains(text, "rate limit") ||
		strings.Contains(text, "too many requests")
}

// streamFrameRateLimitError reports a rate limit carried by the first SSE
// data frame. Some compatible upstreams send plain text instead of JSON.
func streamFrameRateLimitError(data string) *types.NewAPIError {
	if isPlainRateLimit(data) {
		return types.WithOpenAIError(types.OpenAIError{Message: data, Code: "rate_limit_exceeded"}, http.StatusTooManyRequests)
	}
	var frame struct {
		Type     string `json:"type"`
		Code     any    `json:"code"`
		Message  string `json:"message"`
		Error    any    `json:"error"`
		Response *struct {
			Error any `json:"error"`
		} `json:"response"`
	}
	if common.UnmarshalJsonStr(data, &frame) != nil {
		return nil
	}
	var oaiError *types.OpenAIError
	switch {
	case frame.Error != nil:
		oaiError = dto.GetOpenAIError(frame.Error)
	case frame.Type == "error":
		oaiError = &types.OpenAIError{Message: frame.Message, Code: frame.Code}
	case frame.Response != nil && frame.Response.Error != nil:
		oaiError = dto.GetOpenAIError(frame.Response.Error)
	case frame.Message != "" && isPlainRateLimit(frame.Message):
		oaiError = &types.OpenAIError{Message: frame.Message, Code: "rate_limit_exceeded"}
	}
	if !isRateLimitError(oaiError) {
		return nil
	}
	return types.WithOpenAIError(*oaiError, http.StatusTooManyRequests)
}

func isPlainRateLimit(text string) bool {
	text = strings.ToLower(strings.TrimSpace(text))
	return strings.HasPrefix(text, "rate limit exceeded:") ||
		strings.HasPrefix(text, "rate limit reached for ")
}

// upstreamJSONStreamBodyError detects a stream request answered with a JSON
// error body instead of SSE events. The stream scanner reads only "data:"
// lines, so such a body would otherwise end as an empty successful stream.
// Any other body is restored unchanged for the stream handler.
func upstreamJSONStreamBodyError(resp *http.Response) *types.NewAPIError {
	if resp == nil || resp.Body == nil {
		return nil
	}
	contentType := strings.ToLower(resp.Header.Get("Content-Type"))
	if !strings.Contains(contentType, "json") && !strings.HasPrefix(contentType, "text/plain") {
		return nil
	}
	original := resp.Body
	head, err := io.ReadAll(io.LimitReader(original, maxJSONStreamErrorBodyBytes))
	resp.Body = struct {
		io.Reader
		io.Closer
	}{io.MultiReader(bytes.NewReader(head), original), original}
	if err != nil {
		return nil
	}
	if strings.HasPrefix(contentType, "text/plain") {
		if isPlainRateLimit(string(head)) {
			return types.WithOpenAIError(types.OpenAIError{Message: string(head), Code: "rate_limit_exceeded"}, http.StatusTooManyRequests)
		}
		return nil
	}
	var envelope struct {
		Error   any    `json:"error"`
		Message string `json:"message"`
	}
	if common.Unmarshal(head, &envelope) != nil {
		if isPlainRateLimit(string(head)) {
			return types.WithOpenAIError(types.OpenAIError{Message: string(head), Code: "rate_limit_exceeded"}, http.StatusTooManyRequests)
		}
		return nil
	}
	if apiErr := upstreamBodyError(dto.GetOpenAIError(envelope.Error), resp.StatusCode); apiErr != nil {
		return apiErr
	}
	if isPlainRateLimit(envelope.Message) {
		return types.WithOpenAIError(types.OpenAIError{Message: envelope.Message, Code: "rate_limit_exceeded"}, http.StatusTooManyRequests)
	}
	return nil
}
