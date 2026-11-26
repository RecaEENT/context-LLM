package runner

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"context-LLM/internal/config"
	"context-LLM/internal/provider"
	"context-LLM/internal/scoring"
)

func TestCalibration(t *testing.T) {
	actual := 189294
	result := Calibrate(200000, 6.25, &actual)
	if result == nil || result.RecommendedCharsPerToken != 6.2073 {
		t.Fatalf("unexpected calibration: %#v", result)
	}
}

func TestCalibrateFromTrialsUsesMedian(t *testing.T) {
	trials := []Trial{
		{PromptTokens: intPointer(100)},
		{PromptTokens: intPointer(110)},
		{PromptTokens: intPointer(120)},
	}
	result := CalibrateFromTrials(200000, 6.25, trials)
	if result == nil || result.ActualPromptTokens != 110 {
		t.Fatalf("expected median 110, got %#v", result)
	}
}

func TestNextSeedIsUnique(t *testing.T) {
	runner := New(config.Config{}, t.TempDir(), io.Discard)
	seen := map[int64]bool{}
	for i := 0; i < 100; i++ {
		seed, err := runner.nextSeed()
		if err != nil {
			t.Fatal(err)
		}
		if seen[seed] {
			t.Fatalf("duplicate seed %d", seed)
		}
		seen[seed] = true
	}
}

func TestNextSeedIsSafeAcrossGoroutines(t *testing.T) {
	runner := New(config.Config{}, t.TempDir(), io.Discard)
	seeds := make([]int64, 100)
	var wait sync.WaitGroup
	for index := range seeds {
		wait.Add(1)
		go func(index int) {
			defer wait.Done()
			seed, err := runner.nextSeed()
			if err != nil {
				t.Errorf("nextSeed: %v", err)
				return
			}
			seeds[index] = seed
		}(index)
	}
	wait.Wait()
	seen := map[int64]bool{}
	for _, seed := range seeds {
		if seen[seed] {
			t.Fatalf("duplicate concurrent seed %d", seed)
		}
		seen[seed] = true
	}
}

func TestRequestThrottleHonorsInterval(t *testing.T) {
	runner := New(config.Config{RequestIntervalMS: 20}, t.TempDir(), io.Discard)
	started := time.Now()
	runner.throttle()
	runner.throttle()
	if elapsed := time.Since(started); elapsed < 15*time.Millisecond {
		t.Fatalf("throttle returned too early: %s", elapsed)
	}
}

func TestRetryDelayUsesBackoffAndRetryAfter(t *testing.T) {
	if got := retryDelay(1, provider.Result{}); got != 500*time.Millisecond {
		t.Fatalf("unexpected first delay: %s", got)
	}
	if got := retryDelay(3, provider.Result{}); got != 2*time.Second {
		t.Fatalf("unexpected third delay: %s", got)
	}
	if got := retryDelay(1, provider.Result{RetryAfter: 4 * time.Second}); got != 4*time.Second {
		t.Fatalf("Retry-After was not honored: %s", got)
	}
	if got := retryDelay(1, provider.Result{RetryAfter: time.Minute}); got != 30*time.Second {
		t.Fatalf("Retry-After was not capped: %s", got)
	}
	if got := retryDelay(1, provider.Result{RetryClass: "rate_limit"}); got != time.Second {
		t.Fatalf("rate-limit delay was not increased: %s", got)
	}
	if got := retryDelay(1, provider.Result{RetryClass: "model_cooldown"}); got != 15*time.Second {
		t.Fatalf("model-cooldown delay was not increased: %s", got)
	}
	if got := retryDelay(1, provider.Result{RetryClass: "model_cooldown", RetryAfter: time.Minute}); got != time.Minute {
		t.Fatalf("model-cooldown Retry-After was not honored: %s", got)
	}
}

func TestResumeTrialReusesExistingResult(t *testing.T) {
	runner := New(config.Config{}, t.TempDir(), io.Discard)
	existing := Trial{Tier: 200000, Label: "primary_1", Seed: 42}
	state := Checkpoint{Trials: []Trial{existing}}
	resumed, err := runner.resumeTrial(&state, 200000, "primary_1")
	if err != nil {
		t.Fatal(err)
	}
	if resumed.Seed != existing.Seed || len(state.Trials) != 1 {
		t.Fatalf("existing trial was not reused: %#v", resumed)
	}
}

func TestCheckpointRoundTripAndConfigGuard(t *testing.T) {
	cfg := config.Config{
		BaseURL: "https://example.test/v1",
		Model:   "model-a",
		Tiers:   []int{200000},
	}
	path := filepath.Join(t.TempDir(), "checkpoint.json")
	checkpoint := Checkpoint{Version: checkpointVersion, Phase: "configured", ConfigSignature: ConfigSignature(cfg)}
	if err := SaveCheckpoint(path, checkpoint); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadCheckpoint(path, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Phase != "configured" || loaded.Complete {
		t.Fatalf("unexpected checkpoint: %#v", loaded)
	}
	changed := cfg
	changed.Model = "model-b"
	if _, err := LoadCheckpoint(path, changed); err == nil {
		t.Fatal("expected changed config to be rejected")
	}
	changed = cfg
	changed.StructuredOutput = !cfg.StructuredOutput
	if _, err := LoadCheckpoint(path, changed); err == nil {
		t.Fatal("expected structured output change to be rejected")
	}
}

func TestConfigSignatureTracksAllNonSecretFields(t *testing.T) {
	base := config.Config{}
	baseline := ConfigSignature(base)
	configType := reflect.TypeOf(base)
	for i := 0; i < configType.NumField(); i++ {
		field := configType.Field(i)
		t.Run(field.Name, func(t *testing.T) {
			changed := base
			value := reflect.ValueOf(&changed).Elem().FieldByName(field.Name)
			switch value.Kind() {
			case reflect.String:
				value.SetString("changed")
			case reflect.Bool:
				value.SetBool(true)
			case reflect.Int:
				value.SetInt(1)
			case reflect.Int64:
				value.SetInt(int64(time.Second))
			case reflect.Float64:
				value.SetFloat(1)
			case reflect.Slice:
				element := reflect.New(value.Type().Elem()).Elem()
				switch element.Kind() {
				case reflect.String:
					element.SetString("changed")
				case reflect.Int:
					element.SetInt(1)
				default:
					t.Fatalf("unhandled slice field type: %s", value.Type())
				}
				value.Set(reflect.Append(reflect.MakeSlice(value.Type(), 0, 1), element))
			default:
				t.Fatalf("unhandled config field type: %s", value.Type())
			}
			equal := ConfigSignature(changed) == baseline
			if field.Name == "APIKey" {
				if !equal {
					t.Fatal("API key must not affect checkpoint signature")
				}
			} else if equal {
				t.Fatalf("%s is omitted from checkpoint signature", field.Name)
			}
		})
	}
}

func TestResumeRestoresCalibrationAndStopsOnRequestFailure(t *testing.T) {
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, `{"error":"invalid api key"}`)
	}))
	defer server.Close()
	cfg := config.Config{BaseURL: server.URL, Model: "test", APIKey: "key", APIMode: "chat-completions", Tiers: []int{13000, 14000}, CharsPerToken: 6.25, MaxOutputTokens: 2500, Timeout: time.Second, AutoCalibrate: true, EnhancedAutoTest: true, AnchorCount: 13, TestModes: []string{"needle"}}
	root := t.TempDir()
	r := New(cfg, root, io.Discard)
	r.SetCheckpointPath(filepath.Join(root, "checkpoint.json"))
	state := r.newCheckpoint("configured")
	state.TierIndex = 1
	state.LastPassed = 13000
	r.Config.CharsPerToken = 4.5
	if err := r.saveCheckpoint(state); err != nil {
		t.Fatal(err)
	}
	resumed := New(cfg, root, io.Discard)
	resumed.SetCheckpointPath(filepath.Join(root, "checkpoint.json"))
	trials, err := resumed.Run()
	if err != nil {
		t.Fatal(err)
	}
	if requests != 1 || len(trials) != 1 || trials[0].Score.State != "request_failed" {
		t.Fatalf("request failure should stop without binary search: requests=%d trials=%#v", requests, trials)
	}
	if trials[0].CharsPerTokenUsed != 4.5 {
		t.Fatalf("calibration was not restored: %v", trials[0].CharsPerTokenUsed)
	}
	loaded, err := LoadCheckpoint(filepath.Join(root, "checkpoint.json"), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !loaded.Complete || loaded.BinaryHigh != 0 {
		t.Fatalf("unexpected checkpoint: %#v", loaded)
	}
}

func TestEnhancedStopsOnFailedResponseWithHTTP200(t *testing.T) {
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		_, _ = io.WriteString(w, `{"status":"failed","error":{"code":"server_error","message":"unavailable"}}`)
	}))
	defer server.Close()
	cfg := config.Config{BaseURL: server.URL, Model: "test", APIKey: "key", APIMode: "responses", Tiers: []int{13000, 14000}, CharsPerToken: 1, MaxOutputTokens: 2500, Timeout: time.Second, EnhancedAutoTest: true, AnchorCount: 13, TestModes: []string{"needle"}}
	root := t.TempDir()
	r := New(cfg, root, io.Discard)
	r.SetCheckpointPath(filepath.Join(root, "checkpoint.json"))
	state := r.newCheckpoint("binary")
	state.BinaryLow, state.BinaryHigh = 13000, 40000
	if err := r.saveCheckpoint(state); err != nil {
		t.Fatal(err)
	}
	trials, err := r.Run()
	if err != nil {
		t.Fatal(err)
	}
	if requests != 1 || len(trials) != 1 || trials[0].Score.State != "request_failed" || trials[0].HTTPStatus != http.StatusOK {
		t.Fatalf("failed response misclassified: requests=%d trials=%#v", requests, trials)
	}
	loaded, err := LoadCheckpoint(filepath.Join(root, "checkpoint.json"), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.BinaryLow != 13000 || loaded.BinaryHigh != 40000 {
		t.Fatalf("failure changed bounds: %#v", loaded)
	}
}

func TestRunTierReusesTrialsInSerialAndParallel(t *testing.T) {
	for _, parallel := range []bool{false, true} {
		t.Run(fmt.Sprintf("parallel=%t", parallel), func(t *testing.T) {
			var requests atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests.Add(1)
				w.WriteHeader(http.StatusBadRequest)
				_, _ = io.WriteString(w, `{"error":"context_too_large"}`)
			}))
			defer server.Close()
			cfg := config.Config{BaseURL: server.URL, APIMode: "chat-completions", Timeout: time.Second, CharsPerToken: 1, MaxOutputTokens: 2500, AnchorCount: 13, TestModes: []string{"needle"}, ParallelTests: parallel}
			r := New(cfg, t.TempDir(), io.Discard)
			state := r.newCheckpoint("configured")
			first, err := r.runTier(13000, "primary", &state)
			if err != nil {
				t.Fatal(err)
			}
			if first.passed || !first.contextTooLarge || first.requestFailed || len(first.trials) != 3 || requests.Load() != 3 {
				t.Fatalf("unexpected outcome: %#v", first)
			}
			second, err := r.runTier(13000, "primary", &state)
			if err != nil {
				t.Fatal(err)
			}
			if len(second.trials) != 3 || requests.Load() != 3 {
				t.Fatalf("completed trials were not reused: %#v", second)
			}
		})
	}
}

func TestRunTrialExpandsTruncatedOutput(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if requests.Add(1) == 1 {
			_, _ = io.WriteString(w, `{"choices":[{"finish_reason":"length","message":{"content":"{}"}}],"usage":{"prompt_tokens":100,"completion_tokens":10}}`)
			return
		}
		_, _ = io.WriteString(w, `{"choices":[{"finish_reason":"stop","message":{"content":"{}"}}],"usage":{"prompt_tokens":100,"completion_tokens":20}}`)
	}))
	defer server.Close()
	cfg := config.Config{BaseURL: server.URL, APIMode: "chat-completions", Timeout: time.Second, CharsPerToken: 1, MaxOutputTokens: 2500, AnchorCount: 13, TestModes: []string{"needle"}}
	trial, err := New(cfg, t.TempDir(), io.Discard).runTrial(13000, "primary")
	if err != nil {
		t.Fatal(err)
	}
	if requests.Load() != 2 || trial.OutputRetries != 1 || trial.MaxOutputTokensUsed != 5000 || len(trial.ResponseAttemptMetadata) != 2 {
		t.Fatalf("unexpected trial: %#v", trial)
	}
}

func TestRunTrialRetriesServerFailure(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if requests.Add(1) == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = io.WriteString(w, `{"error":"unavailable"}`)
			return
		}
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"error":"context_too_large"}`)
	}))
	defer server.Close()
	cfg := config.Config{BaseURL: server.URL, APIMode: "chat-completions", Timeout: time.Second, TransportRetries: 1, CharsPerToken: 1, MaxOutputTokens: 2500, AnchorCount: 13, TestModes: []string{"needle"}}
	trial, err := New(cfg, t.TempDir(), io.Discard).runTrial(13000, "primary")
	if err != nil {
		t.Fatal(err)
	}
	if requests.Load() != 2 || trial.Attempts != 2 || len(trial.ResponseAttemptMetadata) != 2 || !trial.ContextTooLarge {
		t.Fatalf("unexpected trial: %#v", trial)
	}
}

func TestBinarySearchUsesCompletedTrialsForBounds(t *testing.T) {
	cfg := config.Config{BinaryPrecisionTokens: 5000}
	r := New(cfg, t.TempDir(), io.Discard)
	state := r.newCheckpoint("binary")
	state.BinaryLow, state.BinaryHigh = 13000, 33000
	for _, step := range []struct {
		tier          int
		label, result string
	}{
		{20000, "binary_13000_33000", scoring.StatePassed},
		{25000, "binary_20000_33000", scoring.StateScoredFailure},
	} {
		for repetition := 1; repetition <= 3; repetition++ {
			state.Trials = append(state.Trials, Trial{Tier: step.tier, Label: fmt.Sprintf("%s_%d", step.label, repetition), Score: scoring.Score{State: step.result}})
		}
	}
	trials, err := r.runBinarySearch(&state)
	if err != nil {
		t.Fatal(err)
	}
	if !state.Complete || state.BinaryLow != 20000 || state.BinaryHigh != 25000 || len(trials) != 6 {
		t.Fatalf("unexpected bounds: %#v", state)
	}
}

func TestFindLatestCheckpointSkipsCompleteRuns(t *testing.T) {
	runsDir := t.TempDir()
	completeDir := filepath.Join(runsDir, "run_a")
	pendingDir := filepath.Join(runsDir, "run_b")
	if err := os.MkdirAll(completeDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(pendingDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := SaveCheckpoint(filepath.Join(completeDir, "checkpoint.json"), Checkpoint{Version: checkpointVersion, Complete: true}); err != nil {
		t.Fatal(err)
	}
	if err := SaveCheckpoint(filepath.Join(pendingDir, "checkpoint.json"), Checkpoint{Version: checkpointVersion, Phase: "configured"}); err != nil {
		t.Fatal(err)
	}
	found, err := FindLatestCheckpoint(runsDir)
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Dir(found) != pendingDir {
		t.Fatalf("found %q, want %q", found, pendingDir)
	}
}

func intPointer(value int) *int {
	return &value
}

func TestCleanupOnlyRemovesTrialDirectory(t *testing.T) {
	root := t.TempDir()
	corpus := filepath.Join(root, "200k_primary")
	if err := os.Mkdir(corpus, 0o755); err != nil {
		t.Fatal(err)
	}
	outside := filepath.Join(filepath.Dir(root), "context-LLM-outside")
	if err := os.MkdirAll(outside, 0o755); err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(outside)
	trials := []Trial{{Corpus: corpus}, {Corpus: outside}}
	deleted := CleanGeneratedFiles(root, trials)
	if len(deleted) != 1 || deleted[0] != "200k_primary" || !trials[0].TestFilesCleaned {
		t.Fatalf("unexpected cleanup: %#v, %#v", deleted, trials)
	}
	if _, err := os.Stat(outside); err != nil {
		t.Fatalf("cleanup touched outside directory: %v", err)
	}
}

func TestOutputLimitExpandsWithAnchorCount(t *testing.T) {
	if got := OutputLimitFor(13, 2500); got != 2500 {
		t.Fatalf("small anchor set should retain configured minimum, got %d", got)
	}
	if got := OutputLimitFor(50, 2500); got != 512+50*64 {
		t.Fatalf("large anchor set should expand, got %d", got)
	}
	if got := OutputLimitFor(50, 4000); got != 4000 {
		t.Fatalf("configured minimum should win, got %d", got)
	}
}
