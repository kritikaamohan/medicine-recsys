"""PDF summary report for a MediGuide check."""
import html
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

DISCLAIMER = ("MediGuide gives general information only. It is not a diagnosis or medical "
              "advice. Always check with a doctor or pharmacist before taking any medicine.")


def nice(s):
    return s.replace("_", " ")


def match_text(share):
    return "Strong match" if share >= 0.7 else "Possible match" if share >= 0.4 else "Weak match"


def _drugs(r):
    """Medicines as a list of dicts, whether stored as a DataFrame or a list."""
    d = r.get("drugs", [])
    return d.to_dict("records") if hasattr(d, "to_dict") else list(d)


TEAL = colors.HexColor("#0F9D8A")


def _esc(text):
    """Escape for reportlab markup and keep to characters the built-in font supports."""
    return html.escape(str(text).encode("latin-1", "replace").decode("latin-1"))


def build_pdf(res, username=None):
    """Return the report as PDF bytes."""
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="MediGuide summary", author="MediGuide")
    ss = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=ss["Title"], textColor=TEAL, fontSize=22, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=TEAL, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=10, leading=14)
    small = ParagraphStyle("s", parent=body, fontSize=8.5, textColor=colors.HexColor("#52606D"))
    warn = ParagraphStyle("w", parent=body, backColor=colors.HexColor("#FFF4E5"),
                          borderPadding=6, spaceBefore=6, spaceAfter=8)

    story = [Paragraph("MediGuide summary", title), Spacer(1, 8),
             Paragraph(_esc(DISCLAIMER), warn)]
    meta = f"<b>Date:</b> {_esc(res['time'])} &nbsp;&nbsp; <b>Age:</b> {int(res['age'])}"
    if username:
        meta += f" &nbsp;&nbsp; <b>Account:</b> {_esc(username)}"
    story += [Paragraph(meta, body),
              Paragraph("<b>Symptoms:</b> " + _esc(", ".join(nice(s) for s in res["symptoms"])), body)]

    if res.get("red_flags") or any(r["emergency"] for r in res["results"]):
        story.append(Paragraph(
            "<b>Urgent:</b> some of the symptoms entered can be serious. If this is happening "
            "now or getting worse, seek urgent medical care.", warn))

    for n, r in enumerate(res["results"], 1):
        total = len(r["matched"]) + len(r["others"])
        story.append(Paragraph(f"{n}. {_esc(r['label'])}", h2))
        story.append(Paragraph(
            f"<b>{match_text(r['share'])}</b>: matched {len(r['matched'])} of {total} "
            "typical symptoms", body))
        if r["description"]:
            story.append(Paragraph(_esc(r["description"]), body))
        if r["matched"]:
            story.append(Paragraph("<b>You mentioned:</b> " +
                                   _esc(", ".join(nice(s) for s in r["matched"])), body))
        if r["others"]:
            story.append(Paragraph("<b>Also common with this condition:</b> " +
                                   _esc(", ".join(nice(s) for s in r["others"][:6])), body))
        if r["emergency"]:
            story.append(Paragraph("<b>This can be an emergency. Seek urgent medical care.</b>", warn))
            continue
        if r["precautions"]:
            story.append(Paragraph("<b>What you can do:</b> " +
                                   _esc("; ".join(t.capitalize() for t in r["precautions"])), body))
        if r["removed"]:
            story.append(Paragraph("Left out because of your allergy: " +
                                   _esc(", ".join(r["removed"])), small))
        drugs = _drugs(r)
        if drugs:
            rows = [["Medicine", "Reviews", "Avg rating (/10)"]] + [
                [_esc(x["drugName"]), int(x["n_reviews"]), x["mean_rating"]] for x in drugs[:5]]
            t = Table(rows, colWidths=[95 * mm, 25 * mm, 40 * mm], repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), TEAL), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD2D9")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F7F6")]),
                ("ALIGN", (1, 0), (-1, -1), "CENTER")]))
            story += [Spacer(1, 4), Paragraph("<b>Medicines other patients rated well:</b>", body), t,
                      Paragraph("Based on patient reviews, not medical evidence. Ask a pharmacist "
                                "or doctor before taking any medicine.", small)]
        else:
            story.append(Paragraph("No patient-review data for this condition. "
                                   "Please speak to a doctor about treatment.", small))

    story += [Spacer(1, 10),
              Paragraph("See a doctor if your symptoms are severe, getting worse, or last more "
                        "than a few days. Please discuss this summary with a doctor or pharmacist.",
                        small)]
    doc.build(story)
    return buf.getvalue()
