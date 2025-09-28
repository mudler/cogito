GINKGO_ARGS?=--fail-fast -v -r --flake-attempts=5
LOG_LEVEL?=debug

test:
	LOG_LEVEL=$(LOG_LEVEL) go run github.com/onsi/ginkgo/v2/ginkgo $(GINKGO_ARGS) --label-filter=!e2e

test-e2e:
	LOG_LEVEL=$(LOG_LEVEL) go run github.com/onsi/ginkgo/v2/ginkgo $(GINKGO_ARGS) --label-filter=e2e

example-chat:
	go run examples/chat/main.go