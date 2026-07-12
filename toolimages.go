package cogito

import (
	"encoding/base64"
	"fmt"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// imageDataURI is a minimal Multimedia carrying a data: URI image so it routes
// through AddMessage's image_url MultiContent path.
type imageDataURI struct{ uri string }

func (i imageDataURI) URL() string { return i.uri }

// imagesFromResultData extracts image blocks from a raw *mcp.CallToolResult
// (as stashed in ToolStatus.ResultData) as data-URI Multimedia values.
func imagesFromResultData(data any) []Multimedia {
	res, ok := data.(*mcp.CallToolResult)
	if !ok || res == nil {
		return nil
	}
	var out []Multimedia
	for _, c := range res.Content {
		if img, ok := c.(*mcp.ImageContent); ok && len(img.Data) > 0 {
			mime := img.MIMEType
			if mime == "" {
				mime = "image/png"
			}
			uri := fmt.Sprintf("data:%s;base64,%s", mime, base64.StdEncoding.EncodeToString(img.Data))
			out = append(out, imageDataURI{uri: uri})
		}
	}
	return out
}

// appendToolImages, when forwarding is enabled and the tool result carried
// images, appends a follow-up user message that carries those images so a
// vision model can see them. OpenAI tool-role messages cannot hold image parts,
// so the image must ride a separate user turn.
func appendToolImages(f Fragment, status ToolStatus, forwarding bool, toolName string) Fragment {
	if !forwarding {
		return f
	}
	imgs := imagesFromResultData(status.ResultData)
	if len(imgs) == 0 {
		return f
	}
	return f.AddMessage(UserMessageRole, fmt.Sprintf("[image output from tool %q]", toolName), imgs...)
}
