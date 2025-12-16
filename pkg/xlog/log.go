package xlog

import (
	"context"
	"log/slog"
	"os"
	"runtime"
	"strings"
)

type LogLevel string

func (l LogLevel) ToSlogLevel() slog.Level {
	switch strings.ToLower(string(l)) {
	case "info", "i":
		return slog.LevelInfo
	case "warn", "warning":
		return slog.LevelWarn
	case "error", "err":
		return slog.LevelError
	case "debug", "d":
		return slog.LevelDebug
	}

	return slog.LevelDebug
}

var logger *slog.Logger

func init() {
	logger = NewLogger(LogLevel(os.Getenv("COGITO_LOG_LEVEL")), os.Getenv("LOG_FORMAT"))
}

func SetLogger(l *slog.Logger) {
	logger = l
}

func NewLogger(level LogLevel, format string) *slog.Logger {
	var handler slog.Handler
	switch format {
	case "json":
		handler = slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: level.ToSlogLevel()})
	default:
		handler = slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: level.ToSlogLevel()})
	}
	return slog.New(handler)
}

func _log(level slog.Level, msg string, args ...any) {
	_, f, l, _ := runtime.Caller(2)
	group := slog.Group(
		"source",
		slog.Attr{
			Key:   "file",
			Value: slog.AnyValue(f),
		},
		slog.Attr{
			Key:   "L",
			Value: slog.AnyValue(l),
		},
	)
	args = append(args, group)
	logger.Log(context.Background(), level, msg, args...)
}

func Info(msg string, args ...any) {
	_log(slog.LevelInfo, msg, args...)
}

func Debug(msg string, args ...any) {
	_log(slog.LevelDebug, msg, args...)
}

func Error(msg string, args ...any) {
	_log(slog.LevelError, msg, args...)
}

func Warn(msg string, args ...any) {
	_log(slog.LevelWarn, msg, args...)
}
