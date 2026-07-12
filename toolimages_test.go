package cogito

import (
	"strings"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	openai "github.com/sashabaranov/go-openai"
)

func imgStatus() ToolStatus {
	return ToolStatus{ResultData: &mcp.CallToolResult{Content: []mcp.Content{
		&mcp.TextContent{Text: "here is the screen"},
		&mcp.ImageContent{MIMEType: "image/png", Data: []byte("\x89PNGfake")},
	}}}
}

func TestAppendToolImagesForwardsImage(t *testing.T) {
	f := appendToolImages(Fragment{}, imgStatus(), true, "computer_use")
	if len(f.Messages) != 1 {
		t.Fatalf("expected 1 appended message, got %d", len(f.Messages))
	}
	m := f.Messages[0]
	if m.Role != string(UserMessageRole) {
		t.Fatalf("expected user role, got %q", m.Role)
	}
	var gotImage bool
	for _, p := range m.MultiContent {
		if p.Type == openai.ChatMessagePartTypeImageURL && p.ImageURL != nil &&
			strings.HasPrefix(p.ImageURL.URL, "data:image/png;base64,") {
			gotImage = true
		}
	}
	if !gotImage {
		t.Fatalf("expected an image_url data URI part, got %+v", m.MultiContent)
	}
}

func TestAppendToolImagesOffIsNoop(t *testing.T) {
	f := appendToolImages(Fragment{}, imgStatus(), false, "computer_use")
	if len(f.Messages) != 0 {
		t.Fatalf("expected no messages when forwarding off, got %d", len(f.Messages))
	}
}

func TestAppendToolImagesNoImageIsNoop(t *testing.T) {
	s := ToolStatus{ResultData: &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: "text only"}}}}
	f := appendToolImages(Fragment{}, s, true, "computer_use")
	if len(f.Messages) != 0 {
		t.Fatalf("expected no messages for text-only result, got %d", len(f.Messages))
	}
}
