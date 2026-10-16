package controller

import (
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/constant"
	"github.com/QuantumNous/new-api/i18n"
	"github.com/QuantumNous/new-api/model"
	"github.com/QuantumNous/new-api/relaykit/types"
	"github.com/QuantumNous/new-api/service"

	"github.com/gin-gonic/gin"
	"github.com/glebarez/sqlite"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gorm.io/gorm"
)

func TestPublicRelayErrorHidesUpstream429AfterRetries(t *testing.T) {
	const detail = "rate limit exceeded: Rate limit reached for gpt-6.1-sol in organization org-test-upstream-limited on tokens per min (TPM): Limit 40000000, Used 39920237, Requested 203475. Please try again in 185ms. Visit https://platform.openai.com/account/rate-limits to learn more."
	gin.SetMode(gin.TestMode)
	require.NoError(t, i18n.Init())
	cases := []struct {
		name     string
		language string
		status   int
		upstream bool
		want     string
	}{
		{"upstream 429 English", "en", 429, true, "The upstream of the current group is at capacity. Please try again later"},
		{"upstream 429 Chinese", "zh-CN", 429, true, "当前分组"},
		{"local 429 unchanged", "en", 429, false, "rate limit exceeded: Rate limit reached for gpt-6.1-sol"},
		{"other upstream error unchanged", "en", 500, true, "rate limit exceeded: Rate limit reached for gpt-6.1-sol"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx, _ := gin.CreateTestContext(httptest.NewRecorder())
			ctx.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
			ctx.Request.Header.Set("Accept-Language", tc.language)
			original := types.WithOpenAIError(types.OpenAIError{Message: detail, Type: "tokens", Code: "rate_limit_exceeded"}, tc.status)
			public := publicRelayError(ctx, original, tc.upstream)
			assert.Equal(t, detail, original.Error())
			public.SetMessage(common.MessageWithRequestId(public.Error(), "request-1"))
			assert.Equal(t, tc.status, public.StatusCode)
			assert.Contains(t, public.ToOpenAIError().Message, tc.want)
			assert.Contains(t, public.ToClaudeError().Message, tc.want)
			if tc.upstream && tc.status == http.StatusTooManyRequests {
				assert.Equal(t, i18n.T(ctx, i18n.MsgRelayGroupUpstreamSaturated), strings.TrimSuffix(public.Error(), " (request id: request-1)"))
				assert.Equal(t, "upstream_error", public.ToOpenAIError().Type)
				assert.Equal(t, "rate_limit_exceeded", public.ToOpenAIError().Code)
				assert.NotContains(t, public.ToOpenAIError().Message, "org-test-upstream-limited")
				assert.NotContains(t, public.ToClaudeError().Message, "org-test-upstream-limited")
				assert.NotContains(t, public.ToOpenAIError().Message, "platform.openai.com")
			}
		})
	}
}

func TestUpstreamHTTP429BodyDoesNotReachClient(t *testing.T) {
	const detail = "rate limit exceeded: Rate limit reached for gpt-6.1-sol in organization org-test-upstream-limited on tokens per min (TPM): Limit 40000000, Used 39920237, Requested 203475. Please try again in 185ms. Visit https://platform.openai.com/account/rate-limits to learn more."
	require.NoError(t, i18n.Init())
	ctx, _ := gin.CreateTestContext(httptest.NewRecorder())
	ctx.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	ctx.Request.Header.Set("Accept-Language", "en")
	messageJSON, err := common.Marshal(detail)
	require.NoError(t, err)
	for _, body := range []string{
		`{"error":{"message":` + string(messageJSON) + `,"type":"tokens","code":"rate_limit_exceeded"}}`,
		`{"message":` + string(messageJSON) + `}`,
	} {
		resp := &http.Response{StatusCode: http.StatusTooManyRequests, Body: io.NopCloser(strings.NewReader(body))}
		upstream := service.RelayErrorHandler(ctx.Request.Context(), resp, false)
		require.NotNil(t, upstream)
		assert.Equal(t, detail, upstream.Error())
		public := publicRelayError(ctx, upstream, true)
		public.SetMessage(common.MessageWithRequestId(public.Error(), "request-1"))
		assert.Equal(t, http.StatusTooManyRequests, public.StatusCode)
		assert.NotContains(t, public.ToOpenAIError().Message, "org-test-upstream-limited")
		assert.Contains(t, public.ToOpenAIError().Message, "The upstream of the current group is at capacity")
	}
}

func TestProcessChannelErrorUsesSnapshotWithoutLeakingChannelMetadata(t *testing.T) {
	gin.SetMode(gin.TestMode)
	previousDB, previousLogDB := model.DB, model.LOG_DB
	previousRedisEnabled := common.RedisEnabled
	previousMainDatabaseType := common.MainDatabaseType()
	previousLogDatabaseType := common.LogDatabaseType()
	previousErrorLogEnabled := constant.ErrorLogEnabled

	database, err := gorm.Open(sqlite.Open(":memory:"), &gorm.Config{})
	require.NoError(t, err)
	sqlDB, err := database.DB()
	require.NoError(t, err)
	sqlDB.SetMaxOpenConns(1)
	require.NoError(t, database.AutoMigrate(&model.User{}, &model.Log{}))
	model.DB, model.LOG_DB = database, database
	common.RedisEnabled = false
	common.SetDatabaseTypes(common.DatabaseTypeSQLite, common.DatabaseTypeSQLite)
	constant.ErrorLogEnabled = true
	t.Cleanup(func() {
		model.DB, model.LOG_DB = previousDB, previousLogDB
		common.RedisEnabled = previousRedisEnabled
		common.SetDatabaseTypes(previousMainDatabaseType, previousLogDatabaseType)
		constant.ErrorLogEnabled = previousErrorLogEnabled
		require.NoError(t, sqlDB.Close())
	})

	require.NoError(t, database.Create(&model.User{Id: 7, Username: "log-owner", Group: "default"}).Error)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	ctx.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	ctx.Set("id", 7)
	ctx.Set("username", "log-owner")
	ctx.Set("token_name", "test-token")
	ctx.Set("token_id", 11)
	ctx.Set("original_model", "gpt-test")
	ctx.Set("group", "default")
	ctx.Set("channel_id", 202)
	ctx.Set("channel_name", "mutable-context-channel")
	ctx.Set("channel_type", 9)
	ctx.Set("use_channel", []string{"101"})
	common.SetContextKey(ctx, constant.ContextKeyRequestStartTime, time.Now().Add(-time.Second))

	channelSnapshot := types.ChannelError{
		ChannelId:   101,
		ChannelType: 1,
		ChannelName: "snapshot-channel",
		AutoBan:     false,
	}
	apiErr := types.NewOpenAIError(errors.New("upstream failed"), types.ErrorCodeBadResponseStatusCode, http.StatusBadGateway)

	processChannelError(ctx, channelSnapshot, apiErr, nil)

	var stored model.Log
	require.NoError(t, database.First(&stored).Error)
	assert.Equal(t, channelSnapshot.ChannelId, stored.ChannelId)
	storedOther, err := common.StrToMap(stored.Other)
	require.NoError(t, err)
	assert.Equal(t, float64(http.StatusBadGateway), storedOther["status_code"])
	for _, key := range []string{"channel_id", "channel_name", "channel_type"} {
		assert.NotContains(t, storedOther, key)
	}
	adminInfo, ok := storedOther["admin_info"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, []any{"101"}, adminInfo["use_channel"])

	logs, total, err := model.GetUserLogs(7, model.LogFilter{Type: model.LogTypeError}, 0, 10)
	require.NoError(t, err)
	require.Equal(t, int64(1), total)
	require.Len(t, logs, 1)
	assert.Equal(t, channelSnapshot.ChannelId, logs[0].ChannelId)
	assert.Empty(t, logs[0].ChannelName)
	userOther, err := common.StrToMap(logs[0].Other)
	require.NoError(t, err)
	assert.NotContains(t, userOther, "admin_info")
	for _, key := range []string{"channel_id", "channel_name", "channel_type"} {
		assert.NotContains(t, userOther, key)
	}
}
