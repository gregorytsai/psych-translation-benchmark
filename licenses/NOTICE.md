# Third-party and archived-source notices

This file is a notice and provenance index. It is not a blanket relicensing of
the archived benchmark source. The repository root `LICENSE` applies only to
the new website code and public-release validation code that is owned by this
project and explicitly covered there.

## Archived source inventory

`benchmark_code/` contains 114 archived source/template files reviewed for the
public release:

| extension | count | treatment |
|---|---:|---|
| `.py` | 89 | archived source; no blanket MIT assertion |
| `.jinja` | 7 | upstream/model-derived or project adapter templates; see below |
| `.sh` | 5 | archived source; no blanket MIT assertion |
| `.swift` | 4 | archived source; no blanket MIT assertion |
| `.mjs` | 9 | archived source; no blanket MIT assertion |

The 96-file `*.py`/`*.jinja` subset is the reviewed code/template set used by
the publication gate. Historical tools remain for inspection and
reproducibility; they do not supply the site's latest ranking.

## Gemma and TranslateGemma

TranslateGemma/Gemma 3 templates and model-derivative context are subject to
the Google Gemma Terms of Use. The required distribution notice is:

> Gemma is provided under and subject to the Gemma Terms of Use found at
> ai.google.dev/gemma/terms

For machine-readable and copy/paste use, the same required notice in one line
is: `Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms`.

The snapshot is reproduced in [`GEMMA_TERMS_OF_USE.txt`](GEMMA_TERMS_OF_USE.txt)
and the official URL is controlling if the terms change. A model-weight,
GGUF, mmproj, or other model-derivative distribution is outside this public
code archive and requires a separate supply-chain and license review.

Gemma 4 is recorded separately under the official Apache License 2.0 page:
[`GEMMA4_APACHE-2.0.txt`](GEMMA4_APACHE-2.0.txt) and the canonical text in
[`APACHE-2.0.txt`](APACHE-2.0.txt). The Gemma 4 model-card license statement
does not automatically relicense every local QAT/GGUF/mmproj/converter,
quantizer, or extracted template.

## Template-by-template status

| path | status and required treatment |
|---|---|
| `translation_benchmark_20260916_luna/embedded_templates/translategemma-12b-it.Q8_0.jinja` | TranslateGemma/Gemma 3 upstream or extracted template; retain Gemma Terms notice and upstream attribution. |
| `translation_benchmark_20260916_luna/embedded_templates/translategemma-4b-it.Q8_0.jinja` | TranslateGemma/Gemma 3 upstream or extracted template; retain Gemma Terms notice and upstream attribution. |
| `translation_benchmark_20260916_luna/candidate_translategemma-official.jinja` | Candidate copy of upstream template; retain upstream rights and do not call it project MIT. |
| `translation_benchmark_20260916_luna/candidate_translategemma-en-zh-tw-server.jinja` | **MODIFIED PROJECT ADAPTER**: fixed English-to-`zh-TW` service contract; upstream-derived portions retain their original terms; project-owned wrapper portions are separate. |
| `translategemma-en-zh-tw.jinja` | **MODIFIED PROJECT ADAPTER**: project English-to-`zh-TW` adapter with TranslateGemma/Gemma 3 context; upstream-derived portions retain their original terms. |
| `translation_benchmark_20260916_luna/embedded_templates/gemma-4-12B-it-qat-UD-Q4_K_XL.jinja` | Gemma 4 upstream/extracted template; Apache 2.0 model-family record is separate from Gemma Terms. Preserve upstream provenance and modification notice. |
| `translation_benchmark_20260916_luna/embedded_templates/gemma-4-E4B-it-qat-UD-Q4_K_XL.jinja` | Gemma 4 upstream/extracted template; Apache 2.0 model-family record is separate from Gemma Terms. Preserve upstream provenance and modification notice. |

No single MIT statement in this repository covers these seven templates or
the remaining archived source files. If a file is modified in a future public
revision, add a prominent modified-file notice and update the provenance
record before publishing.

## Source and model references

- Gemma Terms: <https://ai.google.dev/gemma/terms/>
- Gemma 4 Apache 2.0 page: <https://ai.google.dev/gemma/apache_2>
- TranslateGemma model card: <https://huggingface.co/google/translategemma-4b-it>
- Gemma 4 model card: <https://ai.google.dev/gemma/docs/core/model_card_4>
- Official TranslateGemma announcement: <https://blog.google/innovation-and-ai/technology/developers-tools/translategemma/>
- WHO and public-health source rights are separate from this code notice; see
  `docs/licenses.html` and `PUBLICATION_BOUNDARY.md`.

No model weights, private model inventory, credential store, browser profile,
raw request/response archive, or private binding map is included by this
notice.

