package cogito_test

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/docker/go-connections/nat"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/sashabaranov/go-openai"
	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/wait"
)

var (
	containerImage = os.Getenv("LOCALAI_IMAGE")
	modelsDir      = os.Getenv("LOCALAI_MODELS_DIR")
	backendDir     = os.Getenv("LOCALAI_BACKEND_DIR")

	apiEndpoint string
	container   testcontainers.Container
)

const (
	defaultApiPort = "8080"
	defaultModel   = "qwen3-0.6b"
	//defaultModel = "gemma-3-4b-it-qat" // Less thinking traces, faster tests
)

func Test(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "cogito test suite")
}

var _ = BeforeSuite(func() {

	var defaultConfig openai.ClientConfig
	startDockerImage()
	apiPort, err := container.MappedPort(context.Background(), nat.Port(defaultApiPort))
	Expect(err).To(Not(HaveOccurred()))

	defaultConfig = openai.DefaultConfig("")
	apiEndpoint = "http://localhost:" + apiPort.Port() + "/v1" // So that other tests can reference this value safely.
	defaultConfig.BaseURL = apiEndpoint

	// Wait for API to be ready
	client := openai.NewClientWithConfig(defaultConfig)

	Eventually(func() error {
		_, err := client.ListModels(context.TODO())
		return err
	}, "50m").ShouldNot(HaveOccurred())
})

var _ = AfterSuite(func() {
	if container != nil {
		Expect(container.Terminate(context.Background())).To(Succeed())
	}
})

func startDockerImage() {
	// get cwd
	cwd, err := os.Getwd()
	Expect(err).To(Not(HaveOccurred()))
	md := cwd + "/models"

	bd := cwd + "/backends"

	if backendDir != "" {
		bd = backendDir
	}

	if modelsDir != "" {
		md = modelsDir
	}

	_ = os.MkdirAll(md, 0755)
	_ = os.MkdirAll(bd, 0755)

	if containerImage == "" {
		containerImage = "localai/localai:latest"
	}

	proc := runtime.NumCPU()

	req := testcontainers.ContainerRequest{
		Cmd:          []string{defaultModel},
		Image:        containerImage,
		ExposedPorts: []string{defaultApiPort},
		LogConsumerCfg: &testcontainers.LogConsumerConfig{
			Consumers: []testcontainers.LogConsumer{
				&logConsumer{},
			},
		},
		Env: map[string]string{
			"MODELS_PATH":                   "/models",
			"BACKENDS_PATH":                 "/backends",
			"DEBUG":                         "true",
			"THREADS":                       fmt.Sprint(proc),
			"LOCALAI_SINGLE_ACTIVE_BACKEND": "true",
		},
		Mounts: testcontainers.ContainerMounts{
			{

				Source: testcontainers.DockerBindMountSource{ //nolint:all
					HostPath: md,
				},
				Target: "/models",
			},
			{
				Source: testcontainers.DockerBindMountSource{ //nolint:all
					HostPath: bd,
				},
				Target: "/backends",
			},
		},
		WaitingFor: wait.ForAll(
			wait.ForListeningPort(nat.Port(defaultApiPort)).WithStartupTimeout(10*time.Minute),
			wait.ForHTTP("/v1/models").WithPort(nat.Port(defaultApiPort)).WithStartupTimeout(10*time.Minute),
		),
	}

	GinkgoWriter.Printf("Launching Docker Container %s\n", containerImage)

	ctx := context.Background()
	c, err := testcontainers.GenericContainer(ctx, testcontainers.GenericContainerRequest{
		ContainerRequest: req,
		Started:          true,
	})
	Expect(err).To(Not(HaveOccurred()))

	container = c
}

type logConsumer struct {
}

func (l *logConsumer) Accept(log testcontainers.Log) {
	_, _ = GinkgoWriter.Write([]byte(log.Content))
}
