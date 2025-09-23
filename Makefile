GINKGO_ARGS?=--fail-fast -v -r --flake-attempts=5
LOG_LEVEL?=debug

test:
	LOG_LEVEL=$(LOG_LEVEL) go run github.com/onsi/ginkgo/v2/ginkgo $(GINKGO_ARGS)

example-chat:
	go run examples/chat/main.go