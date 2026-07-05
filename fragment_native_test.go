package cogito

import "testing"

type fakeMM struct {
	url          string
	kind         MediaKind
	data, format string
}

func (f fakeMM) URL() string          { return f.url }
func (f fakeMM) MediaKind() MediaKind { return f.kind }
func (f fakeMM) Data() string         { return f.data }
func (f fakeMM) Format() string       { return f.format }

// plainMM implements only Multimedia (URL) — the legacy shape.
type plainMM struct{ url string }

func (p plainMM) URL() string { return p.url }

func TestAddMessageBackwardCompatImage(t *testing.T) {
	f := Fragment{}.AddMessage("user", "hi", plainMM{url: "http://x/y.png"})
	last := f.Messages[len(f.Messages)-1]
	if len(last.MultiContent) != 2 || last.MultiContent[1].Type != "image_url" {
		t.Fatalf("plain Multimedia must produce an image_url part, got %+v", last.MultiContent)
	}
	if len(f.PendingNativeParts) != 0 {
		t.Fatalf("plain image must not populate PendingNativeParts")
	}
}

func TestAddMessageTypedAudioVideoGoPending(t *testing.T) {
	f := Fragment{}.AddMessage("user", "listen",
		fakeMM{kind: MediaAudio, data: "AAAA", format: "wav"},
		fakeMM{kind: MediaVideo, url: "data:video/mp4;base64,BBBB"},
	)
	if len(f.PendingNativeParts) != 2 {
		t.Fatalf("audio+video must land in PendingNativeParts, got %d", len(f.PendingNativeParts))
	}
	if f.PendingNativeParts[0].Kind != MediaAudio || f.PendingNativeParts[0].Format != "wav" || f.PendingNativeParts[0].Data != "AAAA" {
		t.Fatalf("audio part mismatch: %+v", f.PendingNativeParts[0])
	}
	if f.PendingNativeParts[1].Kind != MediaVideo || f.PendingNativeParts[1].URL != "data:video/mp4;base64,BBBB" {
		t.Fatalf("video part mismatch: %+v", f.PendingNativeParts[1])
	}
	last := f.Messages[len(f.Messages)-1]
	for _, p := range last.MultiContent {
		if p.Type == "image_url" {
			t.Fatalf("audio/video must NOT be baked into MultiContent")
		}
	}
}

func TestAddMessageTypedImageGoesMultiContent(t *testing.T) {
	f := Fragment{}.AddMessage("user", "see", fakeMM{kind: MediaImage, url: "data:image/png;base64,CCCC"})
	last := f.Messages[len(f.Messages)-1]
	if len(last.MultiContent) != 2 || last.MultiContent[1].Type != "image_url" || last.MultiContent[1].ImageURL.URL != "data:image/png;base64,CCCC" {
		t.Fatalf("typed image must be an image_url MultiContent part, got %+v", last.MultiContent)
	}
	if len(f.PendingNativeParts) != 0 {
		t.Fatalf("image must not populate PendingNativeParts")
	}
}
