import pytest

from reviewers import Finding, ReviewReport


class TestFinding:
    def test_create_with_critical_severity(self):
        finding = Finding(
            category="security",
            severity="critical",
            line=42,
            description="SQL injection risk",
            suggestion="Use parameterized queries",
        )
        assert finding.category == "security"
        assert finding.severity == "critical"
        assert finding.line == 42
        assert finding.description == "SQL injection risk"
        assert finding.suggestion == "Use parameterized queries"

    def test_create_with_major_severity(self):
        finding = Finding(
            category="quality",
            severity="major",
            line=10,
            description="Dead code",
            suggestion="Remove unused function",
        )
        assert finding.severity == "major"

    def test_create_with_minor_severity(self):
        finding = Finding(
            category="style",
            severity="minor",
            line=5,
            description="Inconsistent naming",
            suggestion="Use snake_case",
        )
        assert finding.severity == "minor"

    def test_create_with_suggestion_severity(self):
        finding = Finding(
            category="performance",
            severity="suggestion",
            line=100,
            description="Consider caching",
            suggestion="Add memoization",
        )
        assert finding.severity == "suggestion"


class TestReviewReport:
    def test_create_report_with_defaults(self):
        report = ReviewReport(
            filename="main.py",
            language="Python",
            total_lines=50,
        )
        assert report.filename == "main.py"
        assert report.language == "Python"
        assert report.total_lines == 50
        assert report.findings == []

    def test_create_report_with_findings(self):
        findings = [
            Finding(category="quality", severity="major", line=10, description="Bug", suggestion="Fix it"),
            Finding(category="style", severity="minor", line=20, description="Format", suggestion="Reformat"),
        ]
        report = ReviewReport(
            filename="app.js",
            language="JavaScript",
            total_lines=100,
            findings=findings,
        )
        assert len(report.findings) == 2

    def test_summary_returns_dict_with_correct_keys(self):
        findings = [
            Finding(category="quality", severity="major", line=1, description="A", suggestion="Fix"),
            Finding(category="security", severity="critical", line=2, description="B", suggestion="Fix"),
            Finding(category="quality", severity="minor", line=3, description="C", suggestion="Fix"),
            Finding(category="performance", severity="suggestion", line=4, description="D", suggestion="Fix"),
            Finding(category="style", severity="minor", line=5, description="E", suggestion="Fix"),
        ]
        report = ReviewReport(
            filename="test.py",
            language="Python",
            total_lines=10,
            findings=findings,
        )
        summary = report.summary
        assert isinstance(summary, dict)
        assert set(summary.keys()) == {"quality", "security", "style", "performance"}
        assert summary["quality"] == 2
        assert summary["security"] == 1
        assert summary["performance"] == 1
        assert summary["style"] == 1

    def test_summary_empty_findings(self):
        report = ReviewReport(filename="empty.py", language="Python", total_lines=0)
        summary = report.summary
        assert all(count == 0 for count in summary.values())

    def test_str_representation(self):
        report = ReviewReport(filename="hello.py", language="Python", total_lines=10)
        output = str(report)
        assert "Code Review Report" in output
        assert "hello.py" in output
        assert "No issues found" in output