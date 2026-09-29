# English README and Assembly Image Design

Date: 2026-09-29

## Goal

Translate the complete repository README from Chinese to English and display the Stewart-platform assembly image near the top of the document.

## README scope

- Translate every Chinese heading, paragraph, table heading, formula explanation, instruction, warning, and limitation in `README.md` into clear technical English.
- Preserve the existing section order and overall document structure.
- Preserve all commands, file paths, URLs, model parameters, dataset counts, metric values, and license references exactly.
- Keep the public metric name as “Boundary accuracy”; do not restore bilateral or millimetre-threshold wording.
- Insert `CAD/装配体.jpg` immediately after the introductory paragraph with English alt text and a bounded display width suitable for GitHub.
- Do not add a CAD-assets section or describe individual STL files.

## CAD files

- Add `CAD/装配体.jpg` and all 45 existing `CAD/*.STL` files to Git.
- Preserve every existing CAD filename and binary file unchanged.
- Use normal Git storage because the directory is approximately 11 MB and its largest file is approximately 4.8 MB, below GitHub's ordinary per-file limit.

## Tests and validation

- Update README-focused tests so they require the translated English headings, commands, metrics, limitation statement, and image reference.
- Add checks that the README no longer contains Chinese prose while allowing the Chinese image filename inside the Markdown/HTML path.
- Verify the image path resolves and all expected CAD files are tracked.
- Run the complete test suite and published-artifact verifier.
- Confirm that datasets, model files, the JSON report, and the Markdown accuracy report are unchanged.
