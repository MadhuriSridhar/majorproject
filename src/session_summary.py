"""
Phase 2: Session summary + PDF export.

Builds a human-readable summary from SessionStats and exports it to a local
PDF using reportlab. PDF generation runs after STOP DRIVING and must not
block the live camera loop (the UI calls it in a background thread).
"""

from __future__ import annotations

from pathlib import Path

from driving_duration import format_hms
from session_manager import SessionStats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


def build_summary_text(stats: SessionStats) -> str:
    """Return a plain-text session summary (used by UI and PDF)."""
    start = stats.start_time.strftime("%Y-%m-%d %H:%M:%S") if stats.start_time else "N/A"
    end = stats.end_time.strftime("%Y-%m-%d %H:%M:%S") if stats.end_time else "N/A"
    night = "ACTIVE" if stats.night_context_active else "INACTIVE"

    lines = [
        "=" * 40,
        "          SESSION SUMMARY",
        "=" * 40,
        "",
        f"Session ID:            {stats.session_id}",
        f"Date:                  {start[:10] if start != 'N/A' else 'N/A'}",
        f"Start Time:            {start}",
        f"End Time:              {end}",
        f"Driving Time:          {format_hms(stats.driving_duration_s)}",
        "",
        f"Average Alertness:      {stats.average_alertness:.0f} / 100",
        f"Minimum Alertness:      {stats.minimum_alertness:.0f} / 100",
        f"Maximum Alertness:      {stats.maximum_alertness:.0f} / 100",
        f"Average Fatigue Score:  {stats.average_fatigue_risk:.0f} / 100",
        f"Maximum Fatigue Score:  {stats.maximum_fatigue_risk:.0f} / 100",
        f"Average Fatigue Risk:   {stats.average_fatigue_risk:.0f} / 100",
        f"Maximum Fatigue Risk:   {stats.maximum_fatigue_risk:.0f} / 100",
        f"Average Distraction:    {stats.average_distraction:.0f} / 100",
        f"Environmental Risk:     {stats.average_environmental_risk:.0f} / 100",
        f"Final Status:          {stats.final_status}",
        f"Safe / Good Time:       {format_hms(stats.time_safe_s)}",
        f"Moderate-Risk Time:     {format_hms(stats.time_moderate_s)}",
        f"High-Risk Time:         {format_hms(stats.time_high_s)}",
        f"Critical Time:          {format_hms(stats.time_critical_s)}",
        "",
        f"Total Blinks:          {stats.blink_count}",
        f"Average Blink Rate:    {stats.average_blink_rate:.1f} / min",
        f"Total Yawns:           {stats.yawn_count}",
        "",
        f"Low Fatigue Events:    {stats.low_fatigue_count}",
        f"Alert Events:          {stats.alert_count}",
        f"Warnings:              {stats.warning_count}",
        f"Drowsy Events:         {stats.drowsy_count}",
        f"Alert Episodes:        {len(stats.alert_events)}",
        f"Distraction Events:    {stats.distraction_event_count}",
        f"Fatigue Events:        {stats.fatigue_event_count}",
        f"Recovery Events:       {stats.recovery_event_count}",
        f"High-Risk Zone Events: {stats.high_risk_zone_event_count}",
        "",
        f"Night Context:         {night}",
        "",
        "Risk-factor summary:",
        f"  Eye:                 {_factor_text(_last_factor(stats, 'eye_risk'))}",
        f"  Blink:               {_factor_text(_last_factor(stats, 'blink_risk'))}",
        f"  Yawn:                {_factor_text(_last_factor(stats, 'yawn_risk'))}",
        f"  Night:               {_factor_text(_last_factor(stats, 'night_risk'))}",
        f"  Driving Duration:    {_factor_text(_last_factor(stats, 'duration_risk'))}",
        "",
        "=" * 40,
    ]
    return "\n".join(lines)


def _last_factor(stats: SessionStats, key: str) -> float | None:
    """
    Return the most recent factor value, or None if the latest sample is
    unavailable (e.g. face not detected).
    """
    series = stats.risk_factor_history.get(key, [])
    if not series:
        return 0.0
    return series[-1]


def _factor_text(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.0f}"


def export_pdf(stats: SessionStats, output_dir: Path | None = None) -> Path:
    """
    Generate a local PDF report for the session.

    Returns the path to the generated PDF. Raises on failure so the caller
    can show an error (but never crash the app).
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    target_dir = Path(output_dir) if output_dir else REPORTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_id = stats.session_id or "session"
    pdf_path = target_dir / f"session_{safe_id}.pdf"

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleX",
        parent=styles["Title"],
        fontSize=18,
        textColor=colors.HexColor("#1a1a2e"),
        spaceAfter=6,
    )
    heading_style = ParagraphStyle(
        "HeadingX",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#16213e"),
        spaceBefore=10,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "BodyX",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
    )

    start = stats.start_time.strftime("%Y-%m-%d %H:%M:%S") if stats.start_time else "N/A"
    end = stats.end_time.strftime("%Y-%m-%d %H:%M:%S") if stats.end_time else "N/A"
    night = "ACTIVE" if stats.night_context_active else "INACTIVE"

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    story: list = []
    story.append(Paragraph("DRIVER ALERTNESS MONITOR", title_style))
    story.append(Paragraph("Session Report", heading_style))
    story.append(Spacer(1, 4))

    # Session info table.
    info_rows = [
        ["Session ID", stats.session_id],
        ["Date", start[:10] if start != "N/A" else "N/A"],
        ["Start Time", start],
        ["End Time", end],
        ["Driving Duration", format_hms(stats.driving_duration_s)],
        ["Average Alertness", f"{stats.average_alertness:.0f} / 100"],
        ["Minimum Alertness", f"{stats.minimum_alertness:.0f} / 100"],
        ["Maximum Alertness", f"{stats.maximum_alertness:.0f} / 100"],
        ["Average Fatigue Score", f"{stats.average_fatigue_risk:.0f} / 100"],
        ["Maximum Fatigue Score", f"{stats.maximum_fatigue_risk:.0f} / 100"],
        ["Average Distraction", f"{stats.average_distraction:.0f} / 100"],
        ["Environmental Risk", f"{stats.average_environmental_risk:.0f} / 100"],
        ["Final Status", stats.final_status],
        ["Total Blinks", str(stats.blink_count)],
        ["Average Blink Rate", f"{stats.average_blink_rate:.1f} / min"],
        ["Total Yawns", str(stats.yawn_count)],
        ["Low Fatigue Events", str(stats.low_fatigue_count)],
        ["Alert Events", str(stats.alert_count)],
        ["Warning Events", str(stats.warning_count)],
        ["Drowsy Events", str(stats.drowsy_count)],
        ["Alert Episodes", str(len(stats.alert_events))],
        ["Distraction Events", str(stats.distraction_event_count)],
        ["Fatigue Events", str(stats.fatigue_event_count)],
        ["Recovery Events", str(stats.recovery_event_count)],
        ["High-Risk Zone Events", str(stats.high_risk_zone_event_count)],
        ["Night Context", night],
    ]
    info_table = Table(info_rows, colWidths=[55 * mm, 100 * mm])
    info_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2f7")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c8d0dc")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(info_table)
    story.append(Spacer(1, 10))

    # Risk-factor summary.
    story.append(Paragraph("Risk-Factor Summary", heading_style))
    factor_rows = [
        ["Factor", "Risk"],
        ["Eye", _factor_text(_last_factor(stats, "eye_risk"))],
        ["Blink", _factor_text(_last_factor(stats, "blink_risk"))],
        ["Yawn", _factor_text(_last_factor(stats, "yawn_risk"))],
        ["Night", _factor_text(_last_factor(stats, "night_risk"))],
        ["Driving Duration", _factor_text(_last_factor(stats, "duration_risk"))],
    ]
    factor_table = Table(factor_rows, colWidths=[55 * mm, 100 * mm])
    factor_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c8d0dc")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(factor_table)
    story.append(Spacer(1, 10))

    # Alertness trend (simple text representation if history exists).
    if stats.alertness_history:
        story.append(Paragraph("Alertness Trend (sampled)", heading_style))
        # Downsample to at most 40 points for readability.
        history = stats.alertness_history
        step = max(1, len(history) // 40)
        sampled = history[::step]
        trend_lines = []
        for sample in sampled:
            alertness = sample.alertness if sample.alertness is not None else 0.0
            bar_len = int(round(alertness / 5.0))
            bar = "#" * bar_len
            trend_lines.append(f"{sample.severity:<12} {alertness:5.0f}  {bar}")
        trend_text = "\n".join(trend_lines)
        story.append(Paragraph(trend_text.replace("\n", "<br/>"), body_style))

    if stats.alert_events:
        story.append(Paragraph("Alert Episodes", heading_style))
        event_rows = [["Severity", "Primary Reason", "Minimum Alertness", "Action"]]
        for event in stats.alert_events:
            event_rows.append([
                event.severity,
                event.primary_reason or "Attention risk elevated",
                f"{event.minimum_alertness:.0f}",
                event.action or "Stay focused",
            ])
        event_table = Table(event_rows, colWidths=[28 * mm, 65 * mm, 32 * mm, 30 * mm])
        event_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c8d0dc")),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(event_table)

    doc.build(story)
    return pdf_path