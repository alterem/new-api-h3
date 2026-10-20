package openai

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/constant"
	relaycommon "github.com/QuantumNous/new-api/relay/common"
	"github.com/QuantumNous/new-api/relay/helper"
	"github.com/QuantumNous/new-api/relaykit/dto"
	"github.com/QuantumNous/new-api/relaykit/types"

	"github.com/gin-gonic/gin"
)

const (
	responsesStartupMaxBytes  = 256 << 10
	responsesStartupMaxEvents = 128
)

type responsesStartupFrame struct {
	Type     string                       `json:"type"`
	Code     any                          `json:"code"`
	Message  string                       `json:"message"`
	Error    any                          `json:"error"`
	Response *dto.OpenAIResponsesResponse `json:"response"`
}

type responsesStartupReadResult struct {
	prefix []byte
	reader *bufio.Reader
	err    *types.NewAPIError
}

type replayBody struct {
	io.Reader
	io.Closer
}

// guardResponsesStartupStream consumes only the bounded Responses startup
// phase. It leaves consumed bytes in front of the original reader so the
// existing scanner, usage accumulator, and billing path still see the stream
// exactly once after the guard approves it.
func guardResponsesStartupStream(c *gin.Context, info *relaycommon.RelayInfo, resp *http.Response) *types.NewAPIError {
	if !helper.ResponsesStartupGuardActive(info) || resp == nil || resp.Body == nil {
		return nil
	}

	ctx := context.Background()
	if c != nil && c.Request != nil {
		ctx = c.Request.Context()
	}
	originalBody := resp.Body
	resultCh := make(chan responsesStartupReadResult, 1)
	go func() {
		resultCh <- readResponsesStartupStream(originalBody)
	}()

	timeout := time.Duration(constant.StreamingTimeout) * time.Second
	if timeout <= 0 {
		timeout = 120 * time.Second
	}
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	select {
	case result := <-resultCh:
		if result.err != nil {
			return result.err
		}
		resp.Body = replayBody{
			Reader: io.MultiReader(bytes.NewReader(result.prefix), result.reader),
			Closer: originalBody,
		}
		return nil
	case <-timer.C:
		_ = originalBody.Close()
		<-resultCh
		return responsesStartupError("Responses stream startup timed out", "upstream_timeout", http.StatusGatewayTimeout, false)
	case <-ctx.Done():
		_ = originalBody.Close()
		<-resultCh
		return responsesStartupError("Responses stream canceled by client", "client_canceled", http.StatusRequestTimeout, true)
	}
}

func readResponsesStartupStream(body io.Reader) responsesStartupReadResult {
	reader := bufio.NewReader(body)
	var prefix bytes.Buffer
	var event bytes.Buffer
	var dataLines []string
	var eventName string
	eventCount := 0

	for {
		line, readErr := reader.ReadString('\n')
		if len(line) > 0 {
			if prefix.Len()+event.Len()+len(line) > responsesStartupMaxBytes {
				return responsesStartupReadResult{err: responsesStartupError(
					"Responses stream startup buffer exceeded 256 KiB",
					"startup_buffer_limit_exceeded",
					http.StatusBadGateway,
					false,
				)}
			}
			event.WriteString(line)

			trimmed := strings.TrimSpace(line)
			switch {
			case strings.HasPrefix(trimmed, "event:"):
				eventName = strings.TrimSpace(strings.TrimPrefix(trimmed, "event:"))
			case strings.HasPrefix(trimmed, "data:"):
				dataLines = append(dataLines, strings.TrimSpace(strings.TrimPrefix(trimmed, "data:")))
			}
		}

		if strings.TrimSpace(line) == "" || readErr == io.EOF {
			if event.Len() > 0 {
				eventCount++
				if eventCount > responsesStartupMaxEvents {
					return responsesStartupReadResult{err: responsesStartupError(
						"Responses stream startup event count exceeded 128",
						"startup_event_limit_exceeded",
						http.StatusBadGateway,
						false,
					)}
				}

				data := strings.Join(dataLines, "\n")
				release, apiErr := inspectResponsesStartupEvent(eventName, data)
				if apiErr != nil {
					return responsesStartupReadResult{err: apiErr}
				}
				if release {
					prefix.WriteString(event.String())
					return responsesStartupReadResult{prefix: prefix.Bytes(), reader: reader}
				}
				prefix.WriteString(event.String())
			}
			event.Reset()
			dataLines = nil
			eventName = ""
		}

		if readErr != nil {
			if readErr == io.EOF {
				return responsesStartupReadResult{err: responsesStartupError(
					"Responses stream ended during startup",
					"startup_stream_ended",
					http.StatusBadGateway,
					false,
				)}
			}
			return responsesStartupReadResult{err: responsesStartupError(
				fmt.Sprintf("Responses stream startup read failed: %v", readErr),
				"startup_stream_read_failed",
				http.StatusBadGateway,
				false,
			)}
		}
	}
}

func inspectResponsesStartupEvent(eventName, data string) (bool, *types.NewAPIError) {
	if data == "" {
		return false, nil
	}

	var frame responsesStartupFrame
	if err := common.UnmarshalJsonStr(data, &frame); err != nil {
		return false, responsesStartupError(
			fmt.Sprintf("invalid Responses startup event JSON: %v", err),
			"startup_invalid_json",
			http.StatusBadGateway,
			false,
		)
	}

	eventType := strings.TrimSpace(frame.Type)
	if eventType == "" {
		eventType = strings.TrimSpace(eventName)
	}
	if apiErr := responsesStartupFrameError(eventType, &frame); apiErr != nil {
		return false, apiErr
	}
	return !isResponsesStartupEvent(eventType), nil
}

func isResponsesStartupEvent(eventType string) bool {
	switch strings.ToLower(strings.TrimSpace(eventType)) {
	case "response.created", "response.in_progress", "response.queued":
		return true
	default:
		return false
	}
}

func responsesStartupFrameError(eventType string, frame *responsesStartupFrame) *types.NewAPIError {
	if frame == nil {
		return responsesStartupError("missing Responses startup event", "startup_invalid_event", http.StatusBadGateway, false)
	}

	var upstreamErr *types.OpenAIError
	switch {
	case frame.Error != nil:
		upstreamErr = dto.GetOpenAIError(frame.Error)
	case frame.Response != nil && frame.Response.Error != nil:
		upstreamErr = frame.Response.GetOpenAIError()
	case strings.EqualFold(eventType, "error"):
		upstreamErr = &types.OpenAIError{Code: frame.Code, Message: frame.Message}
	}

	statusFailed := false
	if frame.Response != nil && len(frame.Response.Status) > 0 {
		var status string
		if err := common.Unmarshal(frame.Response.Status, &status); err == nil {
			statusFailed = strings.EqualFold(status, "failed")
		}
	}
	if strings.EqualFold(eventType, "response.failed") || strings.EqualFold(eventType, "response.error") || statusFailed {
		if upstreamErr == nil {
			upstreamErr = &types.OpenAIError{Code: frame.Code, Message: frame.Message}
		}
		if strings.TrimSpace(upstreamErr.Message) == "" {
			upstreamErr.Message = "Responses startup event reported failure"
		}
	}
	if upstreamErr == nil {
		return nil
	}

	status, skipRetry := classifyResponsesStartupError(upstreamErr)
	return types.WithOpenAIError(*upstreamErr, status, func(err *types.NewAPIError) {
		if skipRetry {
			types.ErrOptionWithSkipRetry()(err)
		}
	})
}

func classifyResponsesStartupError(upstreamErr *types.OpenAIError) (int, bool) {
	code := ""
	typ := ""
	if upstreamErr != nil {
		if upstreamErr.Code != nil {
			code = strings.ToLower(strings.TrimSpace(fmt.Sprint(upstreamErr.Code)))
		}
		typ = strings.ToLower(strings.TrimSpace(upstreamErr.Type))
	}
	text := strings.NewReplacer("-", "_", " ", "_", ":", "_").Replace(code + " " + typ)

	switch {
	case code == "429" || containsAny(text, "rate_limit_exceeded", "rate_limit_error", "ratelimitreached", "too_many_requests", "toomanyrequests", "insufficient_quota"):
		return http.StatusTooManyRequests, false
	case code == "503" || containsAny(text, "model_at_capacity", "overloaded_error", "server_overloaded"):
		return http.StatusServiceUnavailable, false
	case code == "401" || containsAny(text, "invalid_api_key", "authentication_error", "unauthorized"):
		return http.StatusUnauthorized, false
	case code == "403" || containsAny(text, "permission_denied", "permission_error", "forbidden"):
		return http.StatusForbidden, false
	case code == "400" || containsAny(text, "context_length_exceeded", "invalid_request_error", "bad_request"):
		return http.StatusBadRequest, true
	case code == "500" || containsAny(text, "server_error", "internal_server_error"):
		return http.StatusInternalServerError, false
	default:
		return http.StatusBadGateway, false
	}
}

func containsAny(text string, values ...string) bool {
	for _, value := range values {
		if strings.Contains(text, value) {
			return true
		}
	}
	return false
}

func responsesStartupError(message, code string, status int, skipRetry bool) *types.NewAPIError {
	return types.WithOpenAIError(types.OpenAIError{
		Message: message,
		Type:    "upstream_error",
		Code:    code,
	}, status, func(err *types.NewAPIError) {
		if skipRetry {
			types.ErrOptionWithSkipRetry()(err)
		}
	})
}
