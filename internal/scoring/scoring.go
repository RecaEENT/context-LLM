package scoring

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"context-LLM/internal/corpus"
)

type Score struct {
	State         string               `json:"state"`
	AnswerHits    int                  `json:"answer_hits"`
	EvidenceHits  int                  `json:"evidence_hits"`
	QuestionCount int                  `json:"question_count"`
	Missed        []string             `json:"missed,omitempty"`
	Detail        string               `json:"detail,omitempty"`
	Repaired      bool                 `json:"repaired,omitempty"`
	ByMode        map[string]ModeScore `json:"by_mode,omitempty"`
}

type ModeScore struct {
	AnswerHits    int      `json:"answer_hits"`
	EvidenceHits  int      `json:"evidence_hits"`
	QuestionCount int      `json:"question_count"`
	Missed        []string `json:"missed,omitempty"`
}

const (
	StatePassed            = "passed"
	StateRequestFailed     = "request_failed"
	StateInvalidAnswerJSON = "invalid_answer_json"
	StateScoredFailure     = "scored_failure"
)

type ResponseInput struct {
	Text           string
	HTTPStatus     int
	TransportError string
	ProviderError  bool
	ErrorDetail    string
}

var evidenceID = regexp.MustCompile(`[a-z0-9]+([-_][a-z0-9]+)*`)

func Response(corpusPath string, response ResponseInput, evidenceTolerance int) Score {
	expectedData, err := os.ReadFile(filepath.Join(corpusPath, "answers.private.json"))
	if err != nil {
		return Score{State: StateRequestFailed, Detail: err.Error()}
	}
	expected := map[string]corpus.Answer{}
	if err := json.Unmarshal(expectedData, &expected); err != nil {
		return Score{State: StateRequestFailed, Detail: err.Error()}
	}
	if response.HTTPStatus >= 400 || response.TransportError != "" || response.ProviderError {
		detail := response.ErrorDetail
		if detail == "" {
			detail = response.TransportError
		}
		return Score{State: StateRequestFailed, QuestionCount: len(expected), Detail: detail}
	}

	actual, repaired, err := decodeAnswers(response.Text)
	if err != nil {
		return Score{State: StateInvalidAnswerJSON, QuestionCount: len(expected), Detail: err.Error()}
	}

	answerHits, evidenceHits := 0, 0
	missed := []string{}
	byMode := map[string]ModeScore{}
	keys := make([]string, 0, len(expected))
	for key := range expected {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		truth, found := expected[key], actual[key]
		mode := truth.Mode
		if mode == "" {
			mode = "unknown"
		}
		modeScore := byMode[mode]
		modeScore.QuestionCount++
		answerOK := normalize(found.Answer) == normalize(truth.Answer)
		evidenceOK := true
		foundIDs := evidenceID.FindAllString(normalize(found.Evidence), -1)
		for _, required := range strings.Split(normalize(truth.Evidence), " and ") {
			matched := false
			for _, id := range foundIDs {
				if id == required {
					matched = true
					break
				}
			}
			if !matched {
				evidenceOK = false
			}
		}
		if answerOK {
			answerHits++
			modeScore.AnswerHits++
		}
		if evidenceOK {
			evidenceHits++
			modeScore.EvidenceHits++
		}
		if !answerOK || !evidenceOK {
			missed = append(missed, key)
			modeScore.Missed = append(modeScore.Missed, key)
		}
		byMode[mode] = modeScore
	}

	state := StatePassed
	for _, modeScore := range byMode {
		if modeScore.AnswerHits != modeScore.QuestionCount ||
			modeScore.QuestionCount-modeScore.EvidenceHits > evidenceTolerance {
			state = StateScoredFailure
			break
		}
	}
	return Score{
		State:         state,
		AnswerHits:    answerHits,
		EvidenceHits:  evidenceHits,
		QuestionCount: len(expected),
		Missed:        missed,
		Repaired:      repaired,
		ByMode:        byMode,
	}
}

func decodeAnswers(text string) (map[string]corpus.Answer, bool, error) {
	text = extractJSONText(strings.TrimSpace(text))
	direct := map[string]corpus.Answer{}
	if err := json.Unmarshal([]byte(text), &direct); err == nil {
		return direct, false, nil
	}

	candidate := extractJSONObject(text)
	if candidate == "" {
		candidate = text
	}
	repaired, err := repairJSON(candidate)
	if err != nil {
		return nil, false, err
	}
	actual := map[string]corpus.Answer{}
	if err := json.Unmarshal([]byte(repaired), &actual); err != nil {
		return nil, false, err
	}
	return actual, true, nil
}

func extractJSONText(text string) string {
	if !strings.HasPrefix(text, "```") {
		return text
	}
	if newline := strings.Index(text, "\n"); newline >= 0 {
		text = text[newline+1:]
	}
	if end := strings.Index(text, "```"); end >= 0 {
		text = text[:end]
	}
	return strings.TrimSpace(text)
}

func extractJSONObject(text string) string {
	start := strings.Index(text, "{")
	end := strings.LastIndex(text, "}")
	if start < 0 || end < start {
		return ""
	}
	return text[start : end+1]
}

func repairJSON(text string) (string, error) {
	repaired := strings.NewReplacer("“", "\"", "”", "\"").Replace(text)
	repaired = removeTrailingCommas(repaired)
	repaired = balanceDelimiters(repaired)
	if !json.Valid([]byte(repaired)) {
		return "", fmt.Errorf("JSON could not be repaired")
	}
	return repaired, nil
}

func removeTrailingCommas(text string) string {
	data := []byte(text)
	inString, escaped := false, false
	for index := 0; index < len(data); index++ {
		current := data[index]
		if inString {
			if escaped {
				escaped = false
			} else if current == '\\' {
				escaped = true
			} else if current == '"' {
				inString = false
			}
			continue
		}
		if current == '"' {
			inString = true
			continue
		}
		if current != ',' {
			continue
		}
		next := index + 1
		for next < len(data) && (data[next] == ' ' || data[next] == '\n' || data[next] == '\r' || data[next] == '\t') {
			next++
		}
		if next < len(data) && (data[next] == '}' || data[next] == ']') {
			data[index] = ' '
		}
	}
	return string(data)
}

func balanceDelimiters(text string) string {
	stack := []byte{}
	inString, escaped := false, false
	for index := 0; index < len(text); index++ {
		current := text[index]
		if inString {
			if escaped {
				escaped = false
			} else if current == '\\' {
				escaped = true
			} else if current == '"' {
				inString = false
			}
			continue
		}
		switch current {
		case '"':
			inString = true
		case '{', '[':
			stack = append(stack, current)
		case '}', ']':
			if len(stack) == 0 {
				continue
			}
			expected := byte('}')
			if stack[len(stack)-1] == '[' {
				expected = ']'
			}
			if current == expected {
				stack = stack[:len(stack)-1]
			}
		}
	}
	for index := len(stack) - 1; index >= 0; index-- {
		if stack[index] == '{' {
			text += "}"
		} else {
			text += "]"
		}
	}
	return text
}

func normalize(value string) string {
	return strings.ToLower(strings.Join(strings.Fields(value), " "))
}
