You are checking a finished $kind for problems a reader would notice when
they open it. You are given $pages screenshot(s), in page order. A script has
already measured the file. Its numbers and findings are below, so do not
repeat them and do not estimate sizes yourself.

## Measured by the script

$measurements

## Check only these, and only what you can see

$checklist

## Rules

- Report a problem only if you can see it in a screenshot.
- Every finding quotes text that is visible right next to the problem, copied
  exactly, at least three words. If you cannot tie a problem to visible text,
  do not report it.
- `page` is the screenshot's position, starting at 1.
- `region` says where on the page: for example "top of the left column",
  "slide title", "second figure".
- Say nothing about writing style, word choice or the science itself.
- No problems: return an empty list.

## Reply

Only JSON, no prose, no code fence:

{"findings": [{"page": 1, "region": "top of the left column", "check": "raw_markup", "problem": "One sentence saying what is wrong.", "severity": "high", "quote": "exact visible text"}]}

`check` is one of the names in the list above. `severity` is "high" when a
reader loses content, "medium" when it looks broken, "low" when it is
cosmetic.
