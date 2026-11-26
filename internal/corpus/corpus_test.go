package corpus

import (
	"reflect"
	"regexp"
	"strings"
	"testing"
)

func TestTestModesGenerateExpectedCases(t *testing.T) {
	cases := benchmarkCases(2026110806, 13, []string{"needle", "multi_hop", "conflict", "adversarial"})
	if len(cases) != 13 {
		t.Fatalf("unexpected case count or abstention case: %#v", cases)
	}
	counts := map[string]int{}
	for _, testCase := range cases {
		counts[testCase.Mode]++
	}
	for _, mode := range []string{"needle", "multi_hop", "conflict", "adversarial"} {
		if counts[mode] != 3 {
			t.Fatalf("mode %s received %d cases, want 3", mode, counts[mode])
		}
	}
	if counts["abstention"] != 1 {
		t.Fatalf("expected one abstention, got %d", counts["abstention"])
	}
}

func TestNeedleOnlyConfiguration(t *testing.T) {
	cases := benchmarkCases(2026110807, 13, []string{"needle"})
	for _, testCase := range cases {
		if testCase.Mode != "needle" && testCase.Mode != "abstention" {
			t.Fatalf("needle-only configuration generated %s", testCase.Mode)
		}
	}
}

func TestAdversarialUsesMatchingCandidateFormats(t *testing.T) {
	cases := benchmarkCases(2026110808, 13, []string{"adversarial"})
	for _, testCase := range cases {
		if testCase.Mode != "adversarial" {
			continue
		}
		if len(testCase.Records) != 4 {
			t.Fatalf("expected rule and three candidates: %#v", testCase)
		}
		pattern := regexp.MustCompile(`^Registry record (R-[0-9A-F]{8}): The (North|South|East) register lists (ITEM-[0-9A-F]{8}) with code ([a-z]+-[a-z]+-[0-9]{4})\.$`)
		values := map[string]bool{}
		for _, record := range testCase.Records[1:] {
			match := pattern.FindStringSubmatch(strings.TrimSpace(record))
			if match == nil || values[match[4]] {
				t.Fatalf("invalid or duplicate candidate: %q", record)
			}
			values[match[4]] = true
			if match[4] == testCase.Answer.Answer && (!strings.Contains(testCase.Answer.Evidence, match[1]) || !strings.Contains(testCase.Records[0], "use the code from the "+match[2]+" register")) {
				t.Fatalf("rule does not select the expected answer: %#v", testCase)
			}
		}
		if !values[testCase.Answer.Answer] {
			t.Fatalf("answer not in candidates: %#v", testCase)
		}
	}
}

func TestConflictEvidenceRequiresBothSides(t *testing.T) {
	cases := benchmarkCases(2026110809, 13, []string{"conflict"})
	found := false
	for _, testCase := range cases {
		if testCase.Mode != "conflict" {
			continue
		}
		found = true
		ids := strings.Split(testCase.Answer.Evidence, " and ")
		if len(ids) != 2 || !strings.Contains(testCase.Records[0], ids[0]) || !strings.Contains(testCase.Records[1], ids[1]) || !strings.Contains(testCase.Records[1], ids[0]) || strings.Contains(testCase.Records[1], testCase.Answer.Answer) || !strings.Contains(testCase.Records[0], testCase.Answer.Answer) {
			t.Fatalf("conflict evidence does not require both records: %#v", testCase.Answer)
		}
	}
	if !found {
		t.Fatal("no conflict case generated")
	}
}

func TestGeneratedRecordsAreUniqueReproducibleAndLocated(t *testing.T) {
	idPattern := regexp.MustCompile(`R-[0-9A-F]{8}`)
	for seed := int64(0); seed < 100; seed++ {
		cases := benchmarkCases(seed, 50, []string{"needle", "multi_hop", "conflict", "adversarial"})
		if !reflect.DeepEqual(cases, benchmarkCases(seed, 50, []string{"needle", "multi_hop", "conflict", "adversarial"})) {
			t.Fatalf("seed %d is not reproducible", seed)
		}
		material, positions := buildMaterial(100000, seed, cases)
		materialAgain, positionsAgain := buildMaterial(100000, seed, cases)
		if material != materialAgain || !reflect.DeepEqual(positions, positionsAgain) {
			t.Fatalf("seed %d material is not reproducible", seed)
		}
		seen := map[string]bool{}
		recordsByQuestion := map[string][]string{}
		recordCount := 0
		if len(positions) == 0 {
			t.Fatal("no positions")
		}
		for _, testCase := range cases {
			recordsByQuestion[testCase.ID] = testCase.Records
			for _, record := range testCase.Records {
				id := idPattern.FindString(record)
				if id == "" || seen[id] {
					t.Fatalf("seed %d: invalid or repeated record ID %q", seed, id)
				}
				seen[id] = true
				recordCount++
			}
			for _, id := range strings.Split(testCase.Answer.Evidence, " and ") {
				if !strings.Contains(strings.Join(testCase.Records, ""), id) {
					t.Fatalf("seed %d: invalid evidence %q", seed, id)
				}
			}
		}
		for _, position := range positions {
			if position.OffsetBytes < 0 || position.OffsetBytes >= len(material) {
				t.Fatalf("seed %d: incorrect position: %#v", seed, position)
			}
			found := false
			for _, record := range recordsByQuestion[position.QuestionID] {
				if strings.HasPrefix(material[position.OffsetBytes:], record) {
					found = true
					break
				}
			}
			if !found || position.Mode == "" {
				t.Fatalf("seed %d: incorrect position metadata: %#v", seed, position)
			}
		}
		if len(positions) != recordCount {
			t.Fatalf("seed %d: missing positions", seed)
		}
	}
}

func TestAbstentionQuestionVariesBySeed(t *testing.T) {
	questions, locations := map[string]bool{}, map[string]bool{}
	for seed := int64(0); seed < 100; seed++ {
		for _, testCase := range benchmarkCases(seed, 13, []string{"needle"}) {
			if testCase.Mode != "abstention" {
				continue
			}
			if testCase.Answer.Answer != "INSUFFICIENT" || strings.Contains(testCase.Question, "INSUFFICIENT") {
				t.Fatalf("unexpected abstention: %#v", testCase)
			}
			questions[testCase.Question], locations[testCase.ID] = true, true
		}
	}
	if len(questions) != 3 || len(locations) < 2 {
		t.Fatalf("abstention lacks variation: questions=%d positions=%d", len(questions), len(locations))
	}
}

func TestModeAssignmentIsRandomized(t *testing.T) {
	first := benchmarkCases(2026110810, 21, []string{"needle", "multi_hop", "conflict", "adversarial"})
	second := benchmarkCases(2026110811, 21, []string{"needle", "multi_hop", "conflict", "adversarial"})
	same := true
	for index := range first {
		if first[index].Mode != second[index].Mode {
			same = false
			break
		}
	}
	if same {
		t.Fatal("mode assignment did not vary with the random seed")
	}
}
