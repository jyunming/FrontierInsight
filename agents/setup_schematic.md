You write drawing code for the schematic of the system a simulation study studies: a picture of its parts, boundaries and sources, and of what touches or feeds what, printed one paper column wide (3.5 inches). It must be readable at that size.

You never write any word or number. FI writes every label from the study's plan: you only place each part's label by its id, and you draw the shapes. Below, each part shows the exact text FI will print for it (its lines separated by " / "), so you can leave room for it.

The studied system:
Parts (id, kind, the text FI prints for it):
$parts
Links (number: from -> to, the words FI prints on it):
$links
$direction

How to draw:
- Draw each part as the shape it physically has in the system: a thin channel or wire as a thick line, a round body as a circle or an annulus, a wall or a plate as a hatched bar, a region filled with a medium as a light filled area, a source as a small block with an arrow out of it. A periodic boundary is drawn dashed or light, never as a solid wall; an open boundary as a dashed line. Keep the proportions and the arrangement physically plausible: what touches in the system touches in the picture.
- Draw every part and every link (a link as an arrow from the first part to the second).
- Place each part's label exactly once, near its part and never on top of another label or crossed by another part's line or outline; put long labels outside the shapes, with a leader line where it helps.
- Labels are 7 pt type at the final size: one text line is about 0.12 in tall and about 0.055 in per character. The longest text line here has $maxchars characters (about $maxwidth in), so place labels where they have that room, inside the picture.
- Plain and uncluttered: white background, the colours FI gives per kind.

The code is Python. It runs with `sch` already defined (no imports; `math` is available). Calls:
sch.canvas(width, height)   # call first, once: the data units of your drawing; the figure is 3.5 in wide and its height follows width:height, so 1 in = width/3.5 units
sch.rect(part_id, x, y, w, h, style='solid')        # style: 'solid', 'light', 'hatched' or 'outline'
sch.circle(part_id, cx, cy, r, ring_width=None, style='solid')   # ring_width gives an annulus of that width around radius r
sch.arc(part_id, cx, cy, r, theta1_deg, theta2_deg, thickness=1.5)
sch.line(part_id, [(x, y), ...], thickness=1.0, dashed=False)    # thickness in points
sch.polygon(part_id, [(x, y), ...], style='solid')
sch.arrow(link_number, (x0, y0), (x1, y1), label_at=None, curved=0.0)   # label_at=(x, y) places the link's words there
sch.direction((x0, y0), (x1, y1), label_at=None)    # the system's direction, when it has one; label_at places its words
sch.label(part_id, x, y, ha='center', va='center')   # FI's text for that part; ha 'left'/'center'/'right', va 'top'/'center'/'bottom'
sch.leader(part_id, (x0, y0), (x1, y1))              # thin line from a label to its part
Plain Python around the calls is fine (numbers, arithmetic, lists, loops, `math`). FI refuses code that imports anything, uses any attribute of `sch` other than these calls, or holds any string other than a part id or a style word: any text not placed by sch.label, sch.arrow or sch.direction is refused. FI colours each shape by its part's kind.

Answer with the code only, in one fenced block.
