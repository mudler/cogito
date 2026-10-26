package cogito

import "errors"

// permanentError marks an error that no retry can fix.
type permanentError struct{ err error }

func (p permanentError) Error() string { return p.err.Error() }
func (p permanentError) Unwrap() error { return p.err }

// Permanent marks err as one that will fail the same way on every attempt, for
// example a request the model's host refuses for this user. The decision loops
// return it at once instead of retrying with backoff. A client calls it on the
// error it returns from a request. Permanent(nil) is nil, and the error stays
// findable with errors.Is and errors.As.
func Permanent(err error) error {
	if err == nil {
		return nil
	}
	return permanentError{err: err}
}

// IsPermanent reports whether err, or an error it wraps, was marked with
// Permanent. A caller that retries on its own, such as a session layer above
// the decision loops, uses it to stop too.
func IsPermanent(err error) bool { return isPermanentError(err) }

// isPermanentError reports whether err, or an error it wraps, was marked with Permanent.
func isPermanentError(err error) bool {
	var p permanentError
	return errors.As(err, &p)
}
