package cogito

import "testing"

func TestNormalizeToolName(t *testing.T) {
	cases := []struct{ in, want string }{
		{"foo", "foo"},
		{"Foo", "foo"},
		{"functions.foo", "foo"},
		{"tools/foo", "foo"},
		{"namespace::foo", "foo"},
		{"foo-bar", "foo_bar"},
		{"Foo-Bar", "foo_bar"},
		{"functions.Coach_Regeln", "coach_regeln"},
		{"  foo  ", "foo"},
		{"", ""},
		{"functions.", ""}, // nothing after the separator → empty key
	}
	for _, c := range cases {
		if got := normalizeToolName(c.in); got != c.want {
			t.Errorf("normalizeToolName(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestFindLenient(t *testing.T) {
	tools := Tools{newNamedTool("coach_regeln")}

	// exact + lenient variants local models routinely emit
	for _, n := range []string{
		"coach_regeln",           // exact
		"functions.coach_regeln", // namespaced
		"tools/coach_regeln",
		"Coach_Regeln", // case
		"coach-regeln", // separator
	} {
		if tools.Find(n) == nil {
			t.Errorf("Find(%q) = nil, want match", n)
		}
	}

	for _, n := range []string{"other_tool", "", "functions."} {
		if tools.Find(n) != nil {
			t.Errorf("Find(%q) matched, want nil", n)
		}
	}
}

// An exact match must always win, even when a lenient candidate appears earlier
// in the slice — the lenient pass is a fallback, never an override.
func TestFindExactWinsOverLenient(t *testing.T) {
	tools := Tools{newNamedTool("functions.foo"), newNamedTool("foo")}
	got := tools.Find("foo")
	if got == nil || got.Tool().Function.Name != "foo" {
		name := "<nil>"
		if got != nil {
			name = got.Tool().Function.Name
		}
		t.Errorf(`Find("foo") = %q, want exact "foo"`, name)
	}
}
