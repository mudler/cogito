package cogito

import (
	"errors"
	"testing"
)

func TestUserAnswerString(t *testing.T) {
	cases := []struct {
		name string
		in   UserAnswer
		want string
	}{
		{"selected only", UserAnswer{Selected: []string{"print", "validate"}}, "Selected: print, validate"},
		{"text only", UserAnswer{Text: "both, please"}, "Answer: both, please"},
		{"selected and text", UserAnswer{Selected: []string{"print"}, Text: "and log"}, "Selected: print\nAnswer: and log"},
		{"empty", UserAnswer{}, ""},
		{"blank text is empty", UserAnswer{Text: "   "}, ""},
	}
	for _, c := range cases {
		if got := c.in.String(); got != c.want {
			t.Errorf("%s: got %q, want %q", c.name, got, c.want)
		}
	}
}

func TestUserQuestionValidate(t *testing.T) {
	withOptions := UserQuestion{Question: "Print or validate?", Options: []string{"print", "validate"}}
	freeToo := UserQuestion{Question: "Print or validate?", Options: []string{"print", "validate"}, AllowFreeText: true}
	noOptions := UserQuestion{Question: "What is the flag name?", AllowFreeText: true}

	cases := []struct {
		name    string
		q       UserQuestion
		a       UserAnswer
		wantErr error // nil means valid
	}{
		{"one option", withOptions, UserAnswer{Selected: []string{"print"}}, nil},
		{"several options", withOptions, UserAnswer{Selected: []string{"print", "validate"}}, nil},
		{"empty answer", withOptions, UserAnswer{}, ErrInvalidAnswer},
		{"blank text only", withOptions, UserAnswer{Text: " "}, ErrInvalidAnswer},
		{"unknown label", withOptions, UserAnswer{Selected: []string{"delete"}}, ErrInvalidAnswer},
		{"free text not allowed", withOptions, UserAnswer{Text: "both"}, ErrInvalidAnswer},
		{"free text allowed", freeToo, UserAnswer{Text: "both"}, nil},
		{"option plus free text", freeToo, UserAnswer{Selected: []string{"print"}, Text: "and log"}, nil},
		{"text for a question without options", noOptions, UserAnswer{Text: "--dry-run"}, nil},
		{"selection for a question without options", noOptions, UserAnswer{Selected: []string{"x"}}, ErrInvalidAnswer},
	}
	for _, c := range cases {
		err := c.q.Validate(c.a)
		if c.wantErr == nil && err != nil {
			t.Errorf("%s: unexpected error %v", c.name, err)
		}
		if c.wantErr != nil && !errors.Is(err, c.wantErr) {
			t.Errorf("%s: got %v, want %v", c.name, err, c.wantErr)
		}
	}
}
