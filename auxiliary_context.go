package cogito

import (
	"context"
	"fmt"
	"strings"
)

// auxiliaryContext renders a new public-text analysis, not a replay of source
// envelopes. Validate every included projection before rendering any text.
// Shallow analyses may include only the immediate parent; deep analyses walk
// all parents and reject cycles rather than recursively rendering forever.
func auxiliaryContext(ctx context.Context, f Fragment, includeParent, deep bool) (string, string, error) {
	if err := ctx.Err(); err != nil {
		return "", "", err
	}
	fragments := []*Fragment{&f}
	seen := map[*Fragment]bool{&f: true}
	if includeParent {
		for p := f.ParentFragment; p != nil; p = p.ParentFragment {
			if seen[p] {
				return "", "", fmt.Errorf("auxiliary context: cyclic fragment parents")
			}
			seen[p] = true
			fragments = append(fragments, p)
			if !deep {
				break
			}
		}
	}
	for _, fragment := range fragments {
		if _, err := fragment.Conversation(); err != nil {
			return "", "", err
		}
	}
	parents := make([]string, 0, len(fragments)-1)
	for _, p := range fragments[1:] {
		parents = append(parents, p.String())
	}
	return f.String(), strings.Join(parents, "\n\n"), nil
}
