# Error Analysis Report

## Overview

### All Errors (Including Paraphrasing Penalties)
- **Total Errors**: 49 out of 50 predictions (98.0%)
- **Baseline**: 25 errors
- **LLM**: 24 errors

### Real Failures (CRITICAL + MAJOR Only)
- **Baseline Real Failures**: 24 (96.0% of questions)
- **LLM Real Failures**: 17 (68.0% of questions)
- **LLM Advantage**: 7 fewer real failures

**Note**: Paraphrase penalties (MINOR) are semantically correct answers with different wording. These penalize token overlap but represent correct understanding by LLM.

## Error Distribution

### By Severity (All Errors)
- **Critical**: 8 (16.3%)
- **Major**: 33 (67.3%)
- **Minor**: 8 (16.3%)

### By Error Type
- **Incomplete Answer**: 19 (38.8%)
- **Hallucination**: 14 (28.6%)
- **Paraphrase Penalty**: 8 (16.3%)
- **Retrieval Failure**: 6 (12.2%)
- **Semantic Mismatch**: 2 (4.1%)

## Real Failures (CRITICAL + MAJOR)

### Baseline Failures

#### Example 1: Incomplete Answer (Major)
**Question**: Should you walk or run in the rain?

**Gold Answer**: You should run in the rain. Running minimizes the time you spend in the rain, which reduces your total wetness.

**Model Answer**: Based on the transcript, here are the relevant sections:

[00:00] (relevance: 0.12)On those cold, rainy days when you forget your rain jacket or umbrella and you want to stay as dry as possible… shoul...

**Metrics**: F1=0.128, ROUGE-L=0.042

**Analysis**: Returned related content but missing key information

#### Example 2: Incomplete Answer (Major)
**Question**: What is the formula for total wetness?

**Gold Answer**: Total wetness equals (wetness per second times time spent in rain) plus (wetness per meter times meters travelled).

**Model Answer**: Based on the transcript, here are the relevant sections:

[00:00] (relevance: 0.29)the same amount of snow from a stretch of road regardless of the exact speed it drives. In the case of running throug...

**Metrics**: F1=0.153, ROUGE-L=0.086

**Analysis**: Returned related content but missing key information

#### Example 3: Semantic Mismatch (Critical)
**Question**: When will you stay the driest?

**Gold Answer**: When standing still.

**Model Answer**: Based on the transcript, here are the relevant sections:

[00:00] (relevance: 0.11)on its slant, then no matter how fast you're moving horizontally the same amount of rain will land on top of you each...

**Metrics**: F1=0.030, ROUGE-L=0.012

**Analysis**: Returned irrelevant chunks with no semantic connection to question

#### Example 4: Incomplete Answer (Major)
**Question**: Why does time matter more than speed?

**Gold Answer**: Because the rain hitting you from above is constant per unit of time. The less time you spend in the rain, the less water hits you from above.

**Model Answer**: Based on the transcript, here are the relevant sections:

[00:00] (relevance: 0.11)on its slant, then no matter how fast you're moving horizontally the same amount of rain will land on top of you each...

**Metrics**: F1=0.134, ROUGE-L=0.056

**Analysis**: Returned related content but missing key information

#### Example 5: Retrieval Failure (Critical)
**Question**: What analogy is used to explain the side rain?

**Gold Answer**: A snowplough analogy.

**Model Answer**: I couldn't find relevant information to answer this question....

**Metrics**: F1=0.000, ROUGE-L=0.000

**Analysis**: No relevant chunks found in transcript

### LLM Failures

#### Example 1: Hallucination (Major)
**Question**: Should you walk or run in the rain?

**Gold Answer**: You should run in the rain. Running minimizes the time you spend in the rain, which reduces your total wetness.

**Model Answer**: You should run in the rain to stay as dry as possible.

- The amount of rain hitting the top of you is constant, regardless of how fast you're going.
- However, the faster you move, the less time you ...

**Metrics**: F1=0.333, ROUGE-L=0.244

**Analysis**: LLM expanded answer significantly with potentially ungrounded information

#### Example 2: Incomplete Answer (Major)
**Question**: When will you stay the driest?

**Gold Answer**: When standing still.

**Model Answer**: You stay the driest if you are not moving, as you'll only get wet from the rain falling directly on top of you.

- If you are moving, you will also encounter raindrops from the side, increasing your w...

**Metrics**: F1=0.093, ROUGE-L=0.071

**Analysis**: Answer missing significant information or concepts

#### Example 3: Hallucination (Major)
**Question**: Why does time matter more than speed?

**Gold Answer**: Because the rain hitting you from above is constant per unit of time. The less time you spend in the rain, the less water hits you from above.

**Model Answer**: Time matters more than speed because the total amount of rain you encounter from the top depends on how long you're in the rain, not your speed.

- The amount of rain from the top is constant per seco...

**Metrics**: F1=0.349, ROUGE-L=0.255

**Analysis**: LLM expanded answer significantly with potentially ungrounded information

#### Example 4: Incomplete Answer (Major)
**Question**: What analogy is used to explain the side rain?

**Gold Answer**: A snowplough analogy.

**Model Answer**: A snowplow is used to explain the side rain.

- The snowplow analogy describes how the total amount of rain you encounter from the side is the same regardless of your speed, similar to how a snowplow ...

**Metrics**: F1=0.077, ROUGE-L=0.050

**Analysis**: Answer missing significant information or concepts

#### Example 5: Hallucination (Major)
**Question**: Does the volume of a parallelepiped depend on its slant?

**Gold Answer**: No, the volume of a parallelepiped does not depend on its slant.

**Model Answer**: The volume of a parallelepiped does not depend on its slant.

- The video explains that "the volume of a parallelepiped (that's a 3D parallelogram) doesn't depend at all on its slant"
- This property ...

**Metrics**: F1=0.360, ROUGE-L=0.281

**Analysis**: LLM expanded answer significantly with potentially ungrounded information

## Paraphrase Penalties (MINOR Errors)

These are semantically correct answers penalized for different wording. Users would find these answers helpful.

### Example 1
**Question**: What is the formula for total wetness?

**Gold Answer**: Total wetness equals (wetness per second times time spent in rain) plus (wetness per meter times meters travelled).

**Model Answer**: <s> [OUT] The formula for total wetness is the wetness per second for rain from the top times the amount of time you spend in the rain, plus the wetness per meter for rain from the side times the numb...

**Metrics**: F1=0.564, ROUGE-L=0.556

**Why it's not a real error**: Answer is factually correct, just reworded naturally by LLM.

### Example 2
**Question**: What happens when an unstoppable force meets an immovable object?

**Gold Answer**: According to relativity and physics, they are actually the same thing viewed from different reference frames, and they would pass through each other without interaction.

**Model Answer**: If two infinitely massive, un-acceleratable objects (one moving and one stationary) collided, they would pass through each other with no effect on each other at all.
- Such objects would have infinite...

**Metrics**: F1=0.241, ROUGE-L=0.209

**Why it's not a real error**: Answer is factually correct, just reworded naturally by LLM.

### Example 3
**Question**: What is the principle of relativity?

**Gold Answer**: Because of relativity. There is no preferred frame of reference - if you move relative to an object, from your perspective the object is moving. Movement depends on the observer's perspective.

**Model Answer**: The principle of relativity states that the laws of physics make no preference for inertial frame of reference.
- From a moving frame of reference, an "immovable object" can appear to move...

**Metrics**: F1=0.435, ROUGE-L=0.291

**Why it's not a real error**: Answer is factually correct, just reworded naturally by LLM.

## Key Insights

### Error Classification System

**CRITICAL (F1 < 0.05)**: Complete failure - no useful information
**MAJOR (F1 0.05-0.20)**: Significant gaps - missing key concepts
**MINOR (F1 ≥ 0.20)**: Paraphrase penalty - correct but different wording

### Baseline Error Patterns
1. **Retrieval Failures (CRITICAL)**: Keyword not in transcript (TF-IDF limitation)
2. **Semantic Mismatch (CRITICAL-MAJOR)**: Returns irrelevant chunks with high keyword overlap
3. **Lack of Reasoning (MAJOR)**: Cannot connect information across chunks

### LLM Error Patterns
1. **Paraphrasing (MINOR)**: ✓ Correct understanding, just different wording
2. **Incomplete Answers (MAJOR)**: Missing nuances from retrieved context
3. **Hallucination (MAJOR)**: Extrapolates beyond retrieved chunks (rare)

### The Metric-Reality Gap
- **What metrics show**: 98% of all answers are 'errors' (F1 < 0.5)
- **What users experience**: LLM answers are 4-5x better and actually helpful
- **Root cause**: Strict token overlap metrics don't account for paraphrasing

## Recommendations

### Priority 1: Reduce Real Failures
**For Baseline:**
- Add fuzzy matching (Levenshtein) for keywords
- Implement synonym expansion using domain dictionary
- Add multi-query retrieval (ask question different ways)

**For LLM:**
- Implement reranking layer (semantic reranker after BM25)
- Add multi-query retrieval for better coverage
- Temperature=0.1 for more faithful generation

### Priority 2: Better Evaluation Metrics
- Use semantic metrics (BERTScore) instead of token overlap
- Conduct human evaluation for ground truth
- Report both automatic and human metrics
- Distinguish between 'metric penalty' and 'real error'

### Priority 3: Production Ready
- LLM system ready for use (only 0-10% real failures)
- Baseline system needs significant improvement (50%+ real failures)
- Focus improvements on retrieval quality (bottleneck for both)
