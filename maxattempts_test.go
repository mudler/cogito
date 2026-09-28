package cogito

import "testing"

// TestWithMaxAttemptsClampsBelowOne locks the fix for a silent tool-execution
// no-op: the tool-run loops iterate `for range o.maxAttempts`, so a maxAttempts
// of 0 (a caller forwarding an unset config field into WithMaxAttempts) would
// run the tool zero times and hand the model an empty result with no error.
// WithMaxAttempts must clamp anything below 1 up to 1.
func TestWithMaxAttemptsClampsBelowOne(t *testing.T) {
	cases := []struct {
		in, want int
	}{
		{0, 1},
		{-5, 1},
		{1, 1},
		{3, 3},
	}
	for _, c := range cases {
		o := &Options{}
		WithMaxAttempts(c.in)(o)
		if o.maxAttempts != c.want {
			t.Errorf("WithMaxAttempts(%d): maxAttempts=%d, want %d", c.in, o.maxAttempts, c.want)
		}
	}
}

// TestWithMaxAdjustmentAttemptsIgnoresNonPositive mirrors the WithMaxAttempts
// clamp: now that ExecuteTools enforces maxAdjustmentAttempts, a 0 forwarded from an unset config
// field would silently turn every Adjustment into an approval. Values below 1
// leave the default in place instead.
func TestWithMaxAdjustmentAttemptsIgnoresNonPositive(t *testing.T) {
	cases := []struct {
		in, want int
	}{
		{0, 5},
		{-1, 5},
		{1, 1},
		{3, 3},
	}
	for _, c := range cases {
		o := defaultOptions()
		WithMaxAdjustmentAttempts(c.in)(o)
		if o.maxAdjustmentAttempts != c.want {
			t.Errorf("WithMaxAdjustmentAttempts(%d): maxAdjustmentAttempts=%d, want %d", c.in, o.maxAdjustmentAttempts, c.want)
		}
	}
}
