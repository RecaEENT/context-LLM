package runner

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"context-LLM/internal/corpus"
	"context-LLM/internal/provider"
	"context-LLM/internal/scoring"
)

func (runner *Runner) runTrial(tier int, label string) (Trial, error) {
	seed, err := runner.nextSeed()
	if err != nil {
		return Trial{}, err
	}
	charsUsed := runner.Config.CharsPerToken
	corpusPath := filepath.Join(runner.Root, fmt.Sprintf("%dk_%s", tier/1000, label))
	manifest, err := corpus.Generate(corpusPath, tier, seed, charsUsed, runner.Config.AnchorCount, runner.Config.TestModes)
	if err != nil {
		return Trial{}, err
	}
	material, err := os.ReadFile(filepath.Join(corpusPath, manifest.File))
	if err != nil {
		return Trial{}, err
	}
	questions, err := os.ReadFile(filepath.Join(corpusPath, "questions.json"))
	if err != nil {
		return Trial{}, err
	}

	requests, err := runner.executeRequests(corpusPath, manifest, material, questions)
	if err != nil {
		return Trial{}, err
	}
	result, attempts := requests.result, requests.attempts
	outputLimit, outputRetries := requests.outputLimit, requests.outputRetries
	responseFile, attemptMetadata := requests.responseFile, requests.attemptMetadata
	prompt := result.PromptTokens()
	completion := result.CompletionTokens()
	status := result.HTTPStatus
	if status == 0 {
		status = 200
	}
	trial := Trial{
		Tier:                    tier,
		Label:                   label,
		Seed:                    seed,
		Corpus:                  corpusPath,
		ResponseFile:            responseFile,
		Attempts:                attempts,
		MaxOutputTokensUsed:     outputLimit,
		OutputRetries:           outputRetries,
		CharsPerTokenUsed:       charsUsed,
		PromptTokens:            prompt,
		CompletionTokens:        completion,
		ElapsedSeconds:          requests.elapsedSeconds,
		ContextTooLarge:         result.ContextTooLarge(),
		HTTPStatus:              status,
		TransportError:          result.TransportError,
		ResponseMetadata:        provider.Metadata(result.Raw),
		ResponseAttemptMetadata: attemptMetadata,
		Score: scoring.Response(corpusPath, scoring.ResponseInput{
			Text: result.Text, HTTPStatus: result.HTTPStatus, TransportError: result.TransportError,
			ProviderError: result.ProviderError, ErrorDetail: result.ErrorDetail(),
		}, runner.Config.EvidenceTolerance),
	}
	trial.Calibration = Calibrate(tier, charsUsed, prompt)
	if err := writeJSON(filepath.Join(corpusPath, "trial_summary.json"), trial); err != nil {
		return Trial{}, err
	}
	return trial, nil
}

type trialRequests struct {
	result          provider.Result
	attempts        int
	outputLimit     int
	outputRetries   int
	responseFile    string
	attemptMetadata []map[string]any
	elapsedSeconds  float64
}

func (runner *Runner) executeRequests(corpusPath string, manifest corpus.Manifest, material, questions []byte) (trialRequests, error) {
	var result provider.Result
	attempts := 0
	outputLimit := OutputLimitFor(runner.Config.AnchorCount, runner.Config.MaxOutputTokens)
	outputCap := outputAutoCap(runner.Config.MaxOutputTokens)
	outputRetries := 0
	responseFile := ""
	attemptMetadata := []map[string]any{}
	elapsedSeconds := 0.0
	for attempts = 1; attempts <= runner.Config.TransportRetries+1; attempts++ {
		for {
			requestConfig := runner.Config
			requestConfig.MaxOutputTokens = outputLimit
			started := time.Now()
			firstRequest := true
			result = provider.Send(requestConfig, string(material), string(questions), func() {
				runner.throttle()
				if firstRequest {
					started = time.Now()
					firstRequest = false
				}
			})
			result.Raw["_benchmark"] = manifest
			result.Raw["_request"] = map[string]string{
				"base_url":          runner.Config.BaseURL,
				"api_mode":          runner.Config.APIMode,
				"model":             runner.Config.Model,
				"reasoning_effort":  runner.Config.ReasoningEffort,
				"reasoning_summary": runner.Config.ReasoningSummary,
			}
			result.Raw["_extracted_text"] = result.Text
			elapsedSeconds = math.Round(float64(time.Since(started).Milliseconds())/1000*1000) / 1000
			result.Raw["_elapsed_seconds"] = elapsedSeconds
			result.Raw["_attempt"] = attempts
			result.Raw["_max_output_tokens"] = outputLimit
			result.Raw["_retryable"] = result.Retryable
			result.Raw["_retry_after_ms"] = result.RetryAfter.Milliseconds()
			result.Raw["_retry_class"] = result.RetryClass
			result.Raw["_structured_output"] = result.StructuredOutput
			result.Raw["_structured_output_fallback"] = result.StructuredOutputFallback
			suffix := ""
			if outputRetries > 0 {
				suffix = fmt.Sprintf("_output_%d", outputRetries)
			}
			responseFile = filepath.Join(corpusPath, fmt.Sprintf("response_attempt_%d%s.json", attempts, suffix))
			result.Raw["_response_file"] = filepath.Base(responseFile)
			if err := writeJSON(responseFile, result.Raw); err != nil {
				return trialRequests{}, err
			}
			attemptMetadata = append(attemptMetadata, provider.Metadata(result.Raw))
			if result.FinishReason() == "length" && outputRetries < 1 && outputLimit < outputCap {
				nextLimit := outputLimit * 2
				if nextLimit > outputCap {
					nextLimit = outputCap
				}
				runner.logMu.Lock()
				fmt.Fprintf(runner.Out, "Output limit reached; expanding from %d to %d tokens …\n", outputLimit, nextLimit)
				runner.logMu.Unlock()
				outputLimit = nextLimit
				outputRetries++
				continue
			}
			break
		}
		if !result.Retryable || attempts >= runner.Config.TransportRetries+1 {
			break
		}
		delay := retryDelay(attempts, result)
		fmt.Fprintf(runner.Out, "Retrying %s after %s (attempt %d/%d) …\n", result.RetryClass, delay, attempts+1, runner.Config.TransportRetries+1)
		time.Sleep(delay)
	}

	return trialRequests{
		result: result, attempts: attempts, outputLimit: outputLimit, outputRetries: outputRetries,
		responseFile: responseFile, attemptMetadata: attemptMetadata, elapsedSeconds: elapsedSeconds,
	}, nil
}

func retryDelay(attempt int, result provider.Result) time.Duration {
	base := 500 * time.Millisecond
	maximum := 30 * time.Second
	switch result.RetryClass {
	case "model_cooldown":
		base = 15 * time.Second
		maximum = 5 * time.Minute
	case "rate_limit", "server_error", "timeout":
		base = time.Second
	}
	if result.RetryAfter > 0 {
		if result.RetryAfter > maximum {
			return maximum
		}
		return result.RetryAfter
	}
	delay := base
	for index := 1; index < attempt && delay < maximum; index++ {
		delay *= 2
	}
	if delay > maximum {
		return maximum
	}
	return delay
}

func OutputLimitFor(anchorCount, configured int) int {
	estimated := 512 + anchorCount*64
	if estimated < configured {
		return configured
	}
	if estimated > outputAutoCap(configured) {
		return outputAutoCap(configured)
	}
	return estimated
}

func outputAutoCap(configured int) int {
	const automaticMaximum = 16384
	if configured > automaticMaximum {
		return configured
	}
	return automaticMaximum
}
