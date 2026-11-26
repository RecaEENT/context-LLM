package corpus

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	mathrand "math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"context-LLM/internal/config"
)

const GeneratorVersion = 2

type Answer struct {
	Answer   string `json:"answer"`
	Evidence string `json:"evidence"`
	Mode     string `json:"mode,omitempty"`
}

type Question struct {
	ID  string `json:"id"`
	Ask string `json:"ask"`
}

type QuestionSet struct {
	ResponseFormat map[string]map[string]string `json:"response_format"`
	Questions      []Question                   `json:"questions"`
}

type BenchmarkCase struct {
	ID       string
	Mode     string
	Question string
	Answer   Answer
	Records  []string
}

type RecordPosition struct {
	QuestionID  string `json:"question_id"`
	Mode        string `json:"mode"`
	OffsetBytes int    `json:"offset_bytes"`
}

type Manifest struct {
	GeneratorVersion       int              `json:"generator_version"`
	RequestedContextTokens int              `json:"requested_context_tokens"`
	ReservedHeadroomTokens int              `json:"reserved_headroom_tokens"`
	MaterialTokenTarget    int              `json:"material_token_target_estimate"`
	CharsPerTokenEstimate  float64          `json:"chars_per_token_estimate"`
	File                   string           `json:"file"`
	Characters             int              `json:"characters"`
	SHA256                 string           `json:"sha256"`
	AnchorCount            int              `json:"anchor_count"`
	TestModes              []string         `json:"test_modes"`
	RecordPositions        []RecordPosition `json:"record_positions"`
}

func Generate(out string, tier int, seed int64, charsPerToken float64, anchorCount int, testModes []string) (Manifest, error) {
	if err := os.MkdirAll(out, 0o755); err != nil {
		return Manifest{}, err
	}
	cases := benchmarkCases(seed, anchorCount, testModes)
	if err := writeJSON(filepath.Join(out, "questions.json"), questionSet(cases)); err != nil {
		return Manifest{}, err
	}
	if err := writeJSON(filepath.Join(out, "answers.private.json"), answerSet(cases)); err != nil {
		return Manifest{}, err
	}

	material, positions := buildMaterial(
		int(math.Round(float64(tier-config.HeadroomTokens)*charsPerToken)),
		seed+int64(tier),
		cases,
	)
	file := fmt.Sprintf("context_%dk.txt", tier/1000)
	if err := writeText(filepath.Join(out, file), material); err != nil {
		return Manifest{}, err
	}
	digest := sha256.Sum256([]byte(material))
	manifest := Manifest{
		GeneratorVersion:       GeneratorVersion,
		RequestedContextTokens: tier,
		ReservedHeadroomTokens: config.HeadroomTokens,
		MaterialTokenTarget:    tier - config.HeadroomTokens,
		CharsPerTokenEstimate:  charsPerToken,
		File:                   file,
		Characters:             len(material),
		SHA256:                 hex.EncodeToString(digest[:]),
		AnchorCount:            anchorCount,
		TestModes:              append([]string(nil), testModes...),
		RecordPositions:        positions,
	}
	if err := writeJSON(filepath.Join(out, "manifest.json"), []Manifest{manifest}); err != nil {
		return Manifest{}, err
	}
	return manifest, nil
}

func benchmarkCases(seed int64, anchorCount int, modes []string) []BenchmarkCase {
	rng := mathrand.New(mathrand.NewSource(seed))
	colors := []string{"amber", "cobalt", "jade", "ivory", "magenta", "ochre", "saffron", "teal"}
	animals := []string{"badger", "crane", "dolphin", "falcon", "gecko", "heron", "lynx", "otter"}
	people := []string{"Avery Chen", "Blair Imani", "Casey Rowan", "Drew Sato", "Ellis Park", "Finley Noor", "Gray Torres", "Harper Vale", "Jordan Kim", "Morgan Lee", "Reese Patel", "Taylor Brooks", "Quinn Silva", "Riley Moss", "Samir Lane", "Robin West"}
	usedIDs := map[string]bool{}
	nextID := func(prefix string) string {
		for {
			id := fmt.Sprintf("%s-%08X", prefix, rng.Uint32())
			if !usedIDs[id] {
				usedIDs[id] = true
				return id
			}
		}
	}
	newValue := func() string {
		return fmt.Sprintf("%s-%s-%d", pick(rng, colors), pick(rng, animals), rng.Intn(9000)+1000)
	}
	cases := make([]BenchmarkCase, 0, anchorCount)
	assignments := make([]string, anchorCount-1)
	for index := range assignments {
		assignments[index] = modes[index%len(modes)]
	}
	rng.Shuffle(len(assignments), func(i, j int) {
		assignments[i], assignments[j] = assignments[j], assignments[i]
	})

	abstentionIndex := rng.Intn(anchorCount) + 1
	assignmentIndex := 0
	for index := 1; index <= anchorCount; index++ {
		qid := questionID(index)
		if index == abstentionIndex {
			missing := []struct{ field, plural string }{
				{"birth date", "birth dates"},
				{"home address", "home addresses"},
				{"telephone number", "telephone numbers"},
			}
			item := missing[rng.Intn(len(missing))]
			id := nextID("R")
			cases = append(cases, BenchmarkCase{
				ID: qid, Mode: "abstention",
				Question: fmt.Sprintf("What is the project owner's %s?", item.field),
				Answer:   Answer{Answer: "INSUFFICIENT", Evidence: id},
				Records:  []string{fmt.Sprintf("Privacy inventory %s: These records contain no %s.\n", id, item.plural)},
			})
			continue
		}
		mode := assignments[assignmentIndex]
		assignmentIndex++
		value := newValue()
		item := nextID("ITEM")
		project := nextID("PROJECT")
		budget := nextID("BUDGET")
		firstID, secondID := nextID("R"), nextID("R")
		oldOwner := pick(rng, people)
		newOwner := pick(rng, people)
		for newOwner == oldOwner {
			newOwner = pick(rng, people)
		}

		switch mode {
		case "needle":
			question := []string{"What is the registered code for %s?", "Which code is recorded for %s?", "Give the registry code associated with %s."}
			cases = append(cases, BenchmarkCase{
				ID:       qid,
				Mode:     mode,
				Question: fmt.Sprintf(pick(rng, question), item),
				Answer:   Answer{Answer: value, Evidence: firstID},
				Records: []string{
					fmt.Sprintf("Registry record %s: The registered code for %s is %s.\n", firstID, item, value),
				},
			})
		case "adversarial":
			sources := []string{"North", "South", "East"}
			selected := rng.Intn(len(sources))
			usedValues := map[string]bool{value: true}
			records := []string{fmt.Sprintf("Registry rule %s: For %s, use the code from the %s register.\n", firstID, item, sources[selected])}
			answerID := ""
			for sourceIndex, source := range sources {
				id := nextID("R")
				candidate := value
				if sourceIndex == selected {
					answerID = id
				} else {
					for usedValues[candidate] {
						candidate = newValue()
					}
					usedValues[candidate] = true
				}
				records = append(records, fmt.Sprintf("Registry record %s: The %s register lists %s with code %s.\n", id, source, item, candidate))
			}
			cases = append(cases, BenchmarkCase{
				ID:       qid,
				Mode:     mode,
				Question: fmt.Sprintf("According to the registry rule, what code applies to %s?", item),
				Answer:   Answer{Answer: value, Evidence: firstID + " and " + answerID},
				Records:  records,
			})
		case "conflict":
			cases = append(cases, BenchmarkCase{
				ID:       qid,
				Mode:     mode,
				Question: fmt.Sprintf("Who is the current owner of project %s?", project),
				Answer:   Answer{Answer: newOwner, Evidence: firstID + " and " + secondID},
				Records: []string{
					fmt.Sprintf("Project record %s: For %s, the initial owner was %s and the deputy was %s.\n", firstID, project, oldOwner, newOwner),
					fmt.Sprintf("Project update %s: For %s, the initial ownership in %s is superseded; the deputy named in %s is now the owner.\n", secondID, project, firstID, firstID),
				},
			})
		case "multi_hop":
			cases = append(cases, BenchmarkCase{
				ID:       qid,
				Mode:     mode,
				Question: fmt.Sprintf("Who personally approves the %s?", budget),
				Answer:   Answer{Answer: newOwner, Evidence: firstID + " and " + secondID},
				Records: []string{
					fmt.Sprintf("Role record %s: The owner of project %s is %s.\n", firstID, project, newOwner),
					fmt.Sprintf("Approval record %s: Budget %s must be personally approved by the owner of project %s.\n", secondID, budget, project),
				},
			})
		}
	}

	return cases
}

func buildMaterial(targetChars int, seed int64, cases []BenchmarkCase) (string, []RecordPosition) {
	rng := mathrand.New(mathrand.NewSource(seed))
	digest := sha256.Sum256([]byte(strconv.FormatInt(seed, 10)))
	tag := hex.EncodeToString(digest[:])[:12]
	type placedRecord struct{ text, questionID, mode string }
	records := []placedRecord{}
	for _, testCase := range cases {
		for _, record := range testCase.Records {
			records = append(records, placedRecord{record, testCase.ID, testCase.Mode})
		}
	}
	rng.Shuffle(len(records), func(i, j int) {
		records[i], records[j] = records[j], records[i]
	})

	var builder strings.Builder
	builder.Grow(targetChars + 4096)
	builder.WriteString("LONG-CONTEXT EVALUATION CORPUS " + tag + "\nFollow each registry rule and explicit project update when interpreting the records.\n\n")
	positions := make([]RecordPosition, 0, len(records))
	targets := make([]int, len(records))
	for i := range records {
		targets[i] = int(float64(targetChars) * (float64(i) + rng.Float64()) / float64(len(records)))
	}
	filler := []string{
		"the operational ledger records routine catalog updates and neutral administrative observations",
		"a quarterly archive indexes correspondence and ordinary shipment summaries",
		"the records office reviews routine filing schedules and internal catalog entries",
		"a maintenance log notes standard inventory checks and document routing",
		"the administrative register lists ordinary meeting notes and archive references",
		"a processing journal describes routine reviews and neutral ledger updates",
	}
	nextAnchor, fillerIndex := 0, 1
	for builder.Len() < targetChars {
		if nextAnchor < len(records) && builder.Len() >= targets[nextAnchor] {
			record := records[nextAnchor]
			positions = append(positions, RecordPosition{record.questionID, record.mode, builder.Len()})
			builder.WriteString(record.text)
			nextAnchor++
			continue
		}
		fmt.Fprintf(&builder, "Archive %s segment %06d: %s.\n", tag, fillerIndex, pick(rng, filler))
		if fillerIndex%17 == 0 {
			fmt.Fprintf(&builder, "Filing note %s-%06d: revision bucket %d remains routine.\n", tag, fillerIndex, rng.Intn(90000)+10000)
		}
		fillerIndex++
	}
	for ; nextAnchor < len(records); nextAnchor++ {
		record := records[nextAnchor]
		positions = append(positions, RecordPosition{record.questionID, record.mode, builder.Len()})
		builder.WriteString(record.text)
	}
	return builder.String(), positions
}

func questionSet(cases []BenchmarkCase) QuestionSet {
	questions := make([]Question, 0, len(cases))
	for _, testCase := range cases {
		questions = append(questions, Question{ID: testCase.ID, Ask: testCase.Question})
	}
	return QuestionSet{
		ResponseFormat: map[string]map[string]string{"question_id": {"answer": "string", "evidence": "record id"}},
		Questions:      questions,
	}
}

func answerSet(cases []BenchmarkCase) map[string]Answer {
	answers := map[string]Answer{}
	for _, testCase := range cases {
		answer := testCase.Answer
		answer.Mode = testCase.Mode
		answers[testCase.ID] = answer
	}
	return answers
}

func pick(rng *mathrand.Rand, values []string) string {
	return values[rng.Intn(len(values))]
}

func questionID(index int) string {
	return fmt.Sprintf("Q%02d", index)
}

func writeJSON(path string, value any) error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o600)
}

func writeText(path, text string) error {
	return os.WriteFile(path, []byte(text), 0o600)
}
