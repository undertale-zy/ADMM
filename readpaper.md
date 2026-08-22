# Technical Paper Explanation Memory

Last updated: 2026-08-15

This file records how to explain technical papers to this user. A future
conversation should read this file before presenting a paper, derivation, or
algorithm. For facts specific to the 2D-ADMM project, also read
`memory_2DADMM.md`.

## 1. Audience And Goal

Assume the user may have no prior background in the paper's domain. Do not
interpret unfamiliarity with one field as a lack of mathematical ability.
Build the missing domain model first, then introduce the mathematics.

Default to explaining in Chinese unless the user explicitly requests another
language. Keep technical names in English only when that helps identify the
paper's original terminology, and explain the Chinese meaning immediately.

The goal is not merely to summarize sections or list formulas. The goal is to
let the user answer all of these questions after reading:

1. What real problem is the paper solving?
2. What information is observed, and what quantity is unknown?
3. Why is the unknown not directly recoverable?
4. What assumptions or prior knowledge make recovery possible?
5. Why is the objective function written in that form?
6. What does every symbol represent?
7. How is the algorithm derived from the objective?
8. What did the author replace with what?
9. Why is that replacement mathematically valid?
10. What improves: accuracy, memory, computation, robustness, or only the
    representation?
11. Under what conditions does the method stop being valid?

## 2. Required Narrative Order

Explain a paper in the following order. Do not skip ahead because a formula is
standard in the field.

### Step 1: Establish the physical or practical scene

Start with the system in plain language:

- What exists in the real world?
- What does the instrument, program, or data source actually observe?
- What final result does the user want?

For an imaging paper, distinguish the physical target, measured signal, and
computed image. For a machine-learning paper, distinguish raw samples, labels,
model output, and the true but unknown quantity.

### Step 2: Identify knowns, unknowns, and noise

Before presenting a model, state:

- Known measurements.
- Unknown variable to estimate.
- Known operators or parameters.
- Noise, missing data, or model error.

Then show the forward model: how an assumed unknown would generate the
measurement. Explain it from left to right in words.

### Step 3: Explain why direct inversion fails

State the concrete obstacle before adding regularization or an algorithm:

- More unknowns than measurements.
- Noise makes inversion unstable.
- Several solutions explain the same observation.
- The matrix is too large to store or invert.
- A term is nondifferentiable or nonconvex.

Use actual dimensions or a small numerical example when available. The user
should understand why a simpler solution is insufficient before seeing the
paper's solution.

### Step 4: Introduce the extra assumption or prior

Explain what the authors believe about the desired solution, such as sparsity,
smoothness, low rank, locality, conservation, or temporal continuity.

State why that assumption is reasonable in the application. Also state cases
where it might fail.

### Step 5: Construct the objective term by term

Do not simply display an objective function. Build it from the requirements:

1. Requirement in plain language.
2. Mathematical term representing that requirement.
3. What happens when that term becomes smaller.
4. Failure mode if that term is used alone.

Only then combine the terms into the complete objective.

Immediately define:

- The estimated quantity, such as `X_hat`.
- `arg min`: the input value that makes the objective smallest.
- Every norm and whether it sums magnitudes, squared magnitudes, or another
  quantity.
- Every weight such as `lambda`, including what larger and smaller values do.

### Step 6: Explain why a special algorithm is needed

Before naming ADMM, proximal gradient, variational inference, or another
solver, identify what makes the objective difficult. Examples:

- One term is smooth while another contains an absolute value.
- The normal equation contains an impractically large inverse.
- Variables have coupled roles that are easier to separate.

Then explain why the selected algorithm fits that exact difficulty.

### Step 7: Introduce auxiliary variables before using them

For every new variable, state all four items:

1. Its mathematical definition.
2. Why it was introduced.
3. Its intuitive role.
4. Whether it is physical, estimated, auxiliary, dual, or temporary.

For ADMM specifically:

- The primary variable represents the data-consistent solution.
- The auxiliary copy separates a difficult regularizer from the data term.
- The equality constraint preserves equivalence with the original problem.
- The scaled dual variable accumulates disagreement between the two copies.
- A temporary expression such as `D(k) = B(k) - V(k)` is only an abbreviation,
  not another physical quantity.

### Step 8: Derive each update without a jump

For an update formula, show this chain:

1. The subproblem being minimized in the current step.
2. Which variables are fixed and which variable is being solved.
3. The derivative or proximal operation used.
4. The equation obtained by setting the derivative to zero.
5. Algebraic rearrangement leading to the update.
6. Plain-language interpretation of the final update.

Never present an inverse formula as if it were a new objective. Clearly state
that it is the analytical solution of one iterative subproblem.

### Step 9: Explain every simplification as a replacement

Use this structure whenever a paper claims an efficient formulation:

- Before: what operation or representation was used?
- After: what replaced it?
- Identity: which exact equality connects the two?
- Preconditions: what assumptions make the equality valid?
- Benefit: what memory, computation, or numerical advantage results?
- Boundary: when would the replacement no longer be valid?

Do not write “using Woodbury” or “by vectorization” as a complete explanation.
State the relevant property first, show what expression changes, and then name
the identity.

### Step 10: Close the iteration loop

After deriving individual steps, restate the complete algorithm in execution
order:

input -> initialization -> step 1 -> step 2 -> correction -> stop condition ->
output

Explain what is expected at convergence and which variable is returned.

### Step 11: Separate different kinds of contribution

Do not combine all improvements into “better.” Distinguish:

- A mathematically equivalent representation that only saves computation.
- A different objective that can change the recovered result.
- A faster implementation of the same update.
- An empirical robustness claim shown only on selected data.
- A genuinely new physical model or sensing process.

For example, a two-dimensional representation can produce the same optimizer
as a vectorized representation while using far less memory. Image-quality
improvement may instead come from the sparse objective, not from the matrix
layout itself.

### Step 12: Evaluate evidence and limitations

End by separating:

- What the equations prove.
- What experiments demonstrate under the tested settings.
- What the authors only claim.
- What assumptions were not tested.
- What the released code actually implements versus what the paper describes.

## 3. Formula Presentation Rules

The user prefers formulas that are directly readable, not LaTeX source.

Use Unicode or plain-text mathematical notation, for example:

`S = F_a * X * F_r^T + Z`

`X_hat = arg min_X [ data error + lambda * sparsity penalty ]`

`V(k+1) = V(k) + X(k+1) - B(k+1)`

Presentation requirements:

- Do not use fenced LaTeX source.
- Avoid unexplained compact notation.
- Put one major transformation on each line.
- Define `T`, `H`, `*`, `-1`, subscripts, and iteration indices when first used.
- Explicitly distinguish uppercase matrices from their lowercase vectorized
  versions.
- After every important formula, provide a sentence beginning with its
  operational meaning, such as “This step predicts the measurement” or “This
  term backprojects the residual into image space.”

If a Unicode symbol could be ambiguous, prefer words. For example, write
“conjugate transpose” next to `H` and “matrix inverse” next to `-1`.

## 4. Variable Discipline

Keep a mental symbol table and never use a symbol before defining it.

Classify variables when introduced:

- Observed: directly measured data.
- True but unknown: the physical quantity that exists but is unavailable.
- Estimated: the algorithm's approximation, commonly marked with “hat.”
- Auxiliary: a copy introduced to split an optimization problem.
- Dual: a variable enforcing agreement or a constraint.
- Hyperparameter: a user-selected tradeoff or algorithm parameter.
- Temporary: an abbreviation used to shorten a later expression.

When a symbol changes form, explain the relation explicitly:

- `X` is a matrix.
- `x = vec(X)` is the same entries stacked into a vector.
- `B` is an auxiliary matrix copy of `X`.
- `b = vec(B)` is its vectorized form.

Do not let the reader infer these relationships from capitalization alone.

## 5. The Four-Layer Equation Method

For every central equation, explain four layers in this order:

1. Purpose: why this equation appears now.
2. Symbols: what every part means.
3. Derivation: which previous statement leads to it.
4. Intuition: what operation it performs in the real system.

Example pattern:

> We now update the data-consistent image while holding the sparse copy fixed.
> The first term measures prediction error, and the second keeps the new image
> near the current auxiliary target. Taking the derivative and setting it to
> zero gives the normal equation. Its solution means: predict the measurement,
> calculate the residual, backproject the residual, and correct the image.

## 6. Continuity Rules

A logically continuous explanation should make every paragraph answer a
question created by the previous paragraph.

Useful transitions:

- “This creates a problem: ...”
- “To resolve that problem, the authors assume ...”
- “That assumption becomes the following penalty term ...”
- “The two terms are difficult to optimize together, so ...”
- “The new variable does not change the original problem because ...”
- “This update still contains an expensive inverse; the next step removes it
  by using ...”

Before moving on, perform a “why checkpoint”:

- Has the need for the next concept already been established?
- Are all symbols in the next formula already defined?
- Is the reader able to say what problem the next transformation solves?

If any answer is no, add the missing bridge first.

## 7. Common Failure Modes To Avoid

- Starting with an abstract or contribution list before explaining the domain.
- Showing an optimization objective without explaining why each term is needed.
- Saying `X_hat` is “the solution” without explaining that it is an estimate.
- Introducing `B`, `V`, `D`, multipliers, latent variables, or residuals without
  roles and definitions.
- Presenting an iterative update as though it were another objective function.
- Naming a theorem or identity without showing its preconditions.
- Saying “ADMM replaces FFT” when FFT is still used inside ADMM updates.
- Claiming a matrix reformulation improves image quality when it is only a
  computationally cheaper form of the same optimization problem.
- Mixing paper claims, mathematical facts, code behavior, and personal
  interpretation in one statement.
- Using acronyms before expanding and explaining them.
- Compressing several algebraic steps when the user's question is specifically
  “why can this be changed into that?”

## 8. Recommended Response Shape

For a full paper explanation, use this outline:

1. One-sentence plain-language purpose.
2. Real-world setup and measured data.
3. Unknown output and forward model.
4. Why direct recovery is difficult.
5. Structural assumption or prior.
6. Objective constructed term by term.
7. Why the proposed solver is needed.
8. New variables and their roles.
9. Update derivation step by step.
10. Computational simplification: before, after, and proof of equivalence.
11. Complete execution flow.
12. Innovation summary: replace what with what and why.
13. Experimental evidence.
14. Assumptions, failure conditions, and code gaps.

For a follow-up about one formula, do not restart the whole paper. Start from
the nearest already-understood premise, rebuild every missing intermediate
step, and stop once the questioned formula has a clear purpose and derivation.

## 9. Final Quality Checklist

Before sending an explanation, verify:

- No symbol appears before its definition.
- Every objective term has a practical reason.
- Every auxiliary variable has a role.
- Every replacement states the exact identity and assumptions that justify it.
- The difference between true quantity and estimate is explicit.
- The difference between final objective and iterative subproblem is explicit.
- The difference between mathematical equivalence and empirical improvement is
  explicit.
- Formulas are directly readable and are not raw LaTeX source.
- The explanation ends with limitations, not only claimed advantages.
- The narrative can be read from top to bottom without requiring the reader to
  infer a missing step.

## 10. Scope Of This Memory

This file defines explanation style, not factual truth about a new paper.
Always read the actual paper, code, data description, and experimental setup
before explaining them. Reuse this reasoning sequence, but do not reuse the
2D-ADMM paper's assumptions or conclusions for unrelated work.
