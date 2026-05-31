package cogito

import (
	"strings"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// TestContentToStringImageDoesNotPanic pins the fix for the crash reported in
// mudler/LocalAI#10101: an MCP tool that returns a non-text content block (e.g.
// an image from osmmcp's get_map_image) used to make contentToString panic with
// "interface conversion: mcp.Content is *mcp.ImageContent, not *mcp.TextContent",
// taking the whole process down. Mixed text + image content must be handled
// gracefully instead.
func TestContentToStringImageDoesNotPanic(t *testing.T) {
	content := []mcp.Content{
		&mcp.TextContent{Text: "here is your map: "},
		&mcp.ImageContent{MIMEType: "image/png", Data: []byte("\x89PNGfakebytes")},
	}

	result := contentToString(content)

	// The text block must still be preserved verbatim.
	if !strings.Contains(result, "here is your map: ") {
		t.Fatalf("expected text content to be preserved, got %q", result)
	}
	// The image block must contribute a descriptive, non-empty marker rather
	// than panicking or being silently lost, so the model knows an image came back.
	if !strings.Contains(result, "image/png") {
		t.Fatalf("expected image mime type to surface in result, got %q", result)
	}
}

// TestContentToStringTextOnly pins the original behavior: plain text blocks are
// concatenated verbatim with no extra decoration.
func TestContentToStringTextOnly(t *testing.T) {
	content := []mcp.Content{
		&mcp.TextContent{Text: "hello "},
		&mcp.TextContent{Text: "world"},
	}

	if got := contentToString(content); got != "hello world" {
		t.Fatalf("expected %q, got %q", "hello world", got)
	}
}

// TestContentToStringEmbeddedResourceText surfaces the text of an embedded
// resource so it remains usable by the model.
func TestContentToStringEmbeddedResourceText(t *testing.T) {
	content := []mcp.Content{
		&mcp.EmbeddedResource{Resource: &mcp.ResourceContents{URI: "file://x", Text: "embedded body"}},
	}

	if got := contentToString(content); !strings.Contains(got, "embedded body") {
		t.Fatalf("expected embedded resource text in result, got %q", got)
	}
}

// TestContentToStringAudioAndResourceLink ensures the remaining non-text content
// variants are summarized rather than panicking.
func TestContentToStringAudioAndResourceLink(t *testing.T) {
	content := []mcp.Content{
		&mcp.AudioContent{MIMEType: "audio/wav", Data: []byte("RIFFfake")},
		&mcp.ResourceLink{URI: "https://example.com/r"},
	}

	got := contentToString(content)
	if !strings.Contains(got, "audio/wav") {
		t.Fatalf("expected audio mime type in result, got %q", got)
	}
	if !strings.Contains(got, "https://example.com/r") {
		t.Fatalf("expected resource link URI in result, got %q", got)
	}
}
