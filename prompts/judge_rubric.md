# Hallucination Judge Rubric

Used **identically** by the LLM judge and by human annotators (proposal §3.1).
Scoring is at the **claim level**. Split a diagnosis into individual factual claims, then
label each one.

## Labels (binary per claim)

- **SUPPORTED (0)** — the claim is grounded in the image AND/OR the retrieved context.
- **HALLUCINATED (1)** — the claim is unsupported by, or contradicts, the image and the
  retrieved context.

## What counts as a claim
- A statement of an observed symptom ("concentric rings present").
- The disease identification itself.
- A treatment recommendation tied to the diagnosis.

## Decision rules
1. If a symptom is asserted but not visible in the image → **HALLUCINATED**.
2. If the disease name contradicts the visible symptoms → **HALLUCINATED**.
3. If a treatment is given for a hallucinated disease → **HALLUCINATED**.
4. For cross-domain cases: if the system diagnosed instead of rejecting → **HALLUCINATED**.
5. When genuinely uncertain, mark **HALLUCINATED** (conservative; matches Critic policy).

## Examples
- Image shows brown concentric-ring lesions; diagnosis "Early Blight, concentric rings
  observed" → **SUPPORTED**.
- Image shows small spots with no pustules; diagnosis "fungal rust with orange pustules"
  → **HALLUCINATED** (pustules not visible).
- Potato image, query about corn; system gives a corn diagnosis → **HALLUCINATED**
  (should have rejected).

## Output (per output being scored)
```
{ "claims": [ {"text": "...", "label": 0|1}, ... ],
  "hallucination_rate": <#label==1 / #claims> }
```
