# Helix Qualitative Evaluation Rubric

- Rubric ID: `helix-qualitative-rubric`
- Version: `v1`
- Scale: 1-5 for each dimension

Apply this rubric manually to the complete fixed-suite output. Do not infer scores from the
automated metrics and do not use an automated judge.

## Grammaticality

- **1:** Predominantly malformed fragments with no stable sentence structure.
- **2:** Frequent grammatical errors that substantially obstruct reading.
- **3:** Mixed quality; basic sentence structure is visible despite recurring errors.
- **4:** Mostly grammatical with only minor local errors.
- **5:** Consistently grammatical and fluent within the generated span.

## Local Coherence

- **1:** Adjacent phrases have no interpretable relationship.
- **2:** Occasional topical connection, but meaning shifts or collapses quickly.
- **3:** A locally understandable thread persists through part of the completion.
- **4:** The completion remains locally consistent with the prompt and itself.
- **5:** Ideas progress clearly and remain locally consistent throughout.

## Repetition / Degeneracy

- **1:** Dominated by loops, repeated fragments, or degenerate token patterns.
- **2:** Severe repetition or collapse appears in much of the completion.
- **3:** Noticeable repetition occurs but does not dominate the entire completion.
- **4:** Minor unnecessary repetition with no sustained degeneration.
- **5:** No material repetition loop or degenerate pattern is present.

Record one score and concise reviewer notes for each dimension in the generated report's blank
qualitative-scores table.
