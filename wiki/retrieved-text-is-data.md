---
title: Retrieved text is data
sources: [core/source_text.py, core/pdf_text.py, core/knowledge.py, core/engine.py, core/summarizer.py, agents/figures_read.md]
updated: 2026-10-01
---
# Retrieved text is data

Text FI retrieves (papers, abstracts, full text, web pages, figure captions, data files, earlier FI results read back from the knowledge base) is material for the model to read and cite, never instructions for it to follow. A sentence planted in a source ("ignore the previous instructions and ...") would otherwise read like FI's own words.

**Fence.** Every block of retrieved text a prompt carries sits between `<<<FI SOURCE TEXT BEGIN>>>` and `<<<FI SOURCE TEXT END>>>`, after one sentence saying what it is (`core/source_text.py::fence`). FI's own notes about the sources (the reminders about preliminary results, the foundational-works note, the elision note of a file list) stay outside. A copy of the markers' words inside a source, in any case, spacing, or split by invisible characters, is replaced before the block is built (`neutralise`), so a source cannot close the block and speak as FI; nothing else in the text changes, so a quote the claim check looks up is still found. The figure reader gets image and caption pairs, which are not fenced; `agents/figures_read.md` tells it the same thing in words.

**Flag.** Each source is checked once per text (again if its full text arrives later) for text addressed to an AI model (a narrow list of order-shaped phrasings) and for text hidden from a reader: Unicode tag characters, three or more invisible characters inside Latin words, and text a PDF draws in white on the page or below 2 pt (`core/pdf_text.py`, `PdfText.hidden_text`; white over a coloured panel or an image, and invisible text layers of scans, do not count). A hit is a flag only: the source is kept whole, its header line in the prompts ends with `[flagged: may contain hidden instructions]`, the trace records a `check_result` (`check source_text: flagged`, with each source and why) and `run.log` names it in one line. A clean literature pass is recorded as `ok`. Nothing is removed, the quest never stops, and there is no switch.

**Measured.** 2,932 stored source texts from earlier quests (65 million characters): none flagged. 108 real PDFs (papers and reports on the maintainer's machine): none flagged, on either PDF reader, once invisible text layers (which OCR and some browsers' print-to-PDF write under a page) stopped counting; they were the one false positive the first run found, in 10 of the 108.

**Not covered.** An order worded outside the list; homoglyph tricks; the source titles in the writer's cross-check summary and the poster's reference list, which carry titles only.

Related: [[literature-screening-record|Literature screening record]] (why each source was kept or dropped).
