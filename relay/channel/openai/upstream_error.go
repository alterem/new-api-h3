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

// streamFrameRateLimitError reports a rate limit carried by one SSE data
// frame: a chat error envelope, a Responses "error" event, or a failed
// Responses event. Stream handlers check the first frame, before anything has
// been forwarded to the client, so the request can still be retried.
func streamFrameRateLimitError(data string) *types.NewAPIError {
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
	}
	if !isRateLimitError(oaiError) {
		return nil
	}
	return types.WithOpenAIError(*oaiError, http.StatusTooManyRequests)
}

// upstreamJSONStreamBodyError detects a stream request answered with a JSON
// error body instead of SSE events. The stream scanner reads only "data:"
// lines, so such a body would otherwise end as an empty successful stream.
// Any other body is restored unchanged for the stream handler.
func upstreamJSONStreamBodyError(resp *http.Response) *types.NewAPIError {
	if resp == nil || resp.Body == nil || !strings.Contains(strings.ToLower(resp.Header.Get("Content-Type")), "json") {
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
	var envelope struct {
		Error any `json:"error"`
	}
	if common.Unmarshal(head, &envelope) != nil {
		return nil
	}
	return upstreamBodyError(dto.GetOpenAIError(envelope.Error), resp.StatusCode)
}
