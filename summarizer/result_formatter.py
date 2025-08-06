#!/usr/bin/env python3
"""
Results formatting and output utilities
"""

import json
from pathlib import Path
from typing import Optional
from data_models import ScanResults, FileAnalysis, SemanticInsight


class ResultFormatter:
    """Handles formatting and displaying scan results"""

    def __init__(self):
        pass

    def display_results(self, results: ScanResults, output_option: Optional[str] = None):
        """Display comprehensive scan results in terminal"""
        self._print_header(results)
        self._print_summary(results)
        self._print_statistics(results)
        self._print_top_files(results)
        self._print_semantic_insights(results)
        self._print_recommendations(results)

        # Save results if requested
        if output_option:
            self.save_results(results, output_option)

    def _print_header(self, results: ScanResults):
        """Print scan completion header"""
        print(f'\n\033[1;32m[Scan Complete - Session {results.session_id}]\033[0m')
        print(f'\033[1;34m[Goal]:\033[0m {results.goal}')
        print(
            f'\033[1;34m[Focus]:\033[0m {results.strategy.scan_focus.title()} | {results.strategy.analysis_depth.title()} Analysis')
        print('-' * 60)

    def _print_summary(self, results: ScanResults):
        """Print executive summary"""
        print(f'\033[1;35m[Summary]:\033[0m')
        print(f'  {results.summary}')
        print()

    def _print_statistics(self, results: ScanResults):
        """Print processing statistics"""
        stats = results.get_summary_stats()

        print(f'\033[1;34m[Statistics]:\033[0m')
        print(f'  Files Found: {results.total_files}')
        print(f'  Files Analyzed: {results.analyzed_files}')
        if results.skipped_files > 0:
            print(f'  Files Skipped: {results.skipped_files}')
        print(f'  High Relevance: {len(results.high_relevance_files)} files')
        print(f'  Semantic Insights: {len(results.semantic_insights)} patterns')
        print(f'  Processing Time: {results.processing_time_seconds:.2f}s')
        if stats['files_per_second'] > 0:
            print(f'  Speed: {stats["files_per_second"]:.1f} files/sec')
        print()

    def _print_top_files(self, results: ScanResults):
        """Print top relevant files"""
        if not results.high_relevance_files:
            print(f'\033[1;33m[No High-Relevance Files Found]\033[0m')
            return

        print(f'\033[1;33m[Top Relevant Files]:\033[0m')

        for i, analysis in enumerate(results.get_top_files(8)):
            file_name = Path(analysis.file_path).name
            score_color = self._get_score_color(analysis.relevance_score)

            print(f'  {i + 1:2d}. {file_name} {score_color}({analysis.relevance_score:.2f})\033[0m')

            # Show file summary
            if analysis.content_summary:
                summary = analysis.content_summary[:80] + "..." if len(
                    analysis.content_summary) > 80 else analysis.content_summary
                print(f'      \033[1;90m{summary}\033[0m')

            # Show key findings
            if analysis.key_findings:
                for finding in analysis.key_findings[:2]:  # Show top 2 findings
                    finding_short = finding[:100] + "..." if len(finding) > 100 else finding
                    print(f'      • {finding_short}')

            # Show semantic matches if any
            if analysis.semantic_matches:
                high_sim_matches = [m for m in analysis.semantic_matches if m.get('similarity', 0) > 0.8]
                if high_sim_matches:
                    print(f'      \033[1;36m↳ {len(high_sim_matches)} high-confidence semantic matches\033[0m')

            print()

    def _print_semantic_insights(self, results: ScanResults):
        """Print semantic analysis insights"""
        if not results.semantic_insights:
            return

        print(f'\033[1;35m[Semantic Insights]:\033[0m')

        for i, insight in enumerate(results.semantic_insights[:5]):  # Show top 5
            print(f'  {i + 1}. Query: "\033[1;96m{insight.query}\033[0m"')
            print(f'     Total matches: {insight.total_matches}, High confidence: {insight.high_confidence_matches}')

            if insight.top_match_similarity > 0.7:
                top_file = Path(insight.top_match_file).name
                print(f'     Best match: {top_file} (similarity: {insight.top_match_similarity:.2f})')

                # Show content preview
                content_preview = insight.top_match_content[:150] + "..." if len(
                    insight.top_match_content) > 150 else insight.top_match_content
                print(f'     \033[1;90m"{content_preview}"\033[0m')

            print()

    def _print_recommendations(self, results: ScanResults):
        """Print actionable recommendations"""
        if not results.recommendations:
            return

        print(f'\033[1;36m[Recommendations]:\033[0m')

        for i, recommendation in enumerate(results.recommendations[:6]):  # Show top 6
            print(f'  {i + 1}. {recommendation}')

        print()

    def _get_score_color(self, score: float) -> str:
        """Get color code based on relevance score"""
        if score >= 0.8:
            return '\033[1;32m'  # Green - High relevance
        elif score >= 0.6:
            return '\033[1;33m'  # Yellow - Medium relevance
        else:
            return '\033[1;31m'  # Red - Low relevance

    def save_results(self, results: ScanResults, output_path: str):
        """Save results to file in various formats"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            if output_path.lower().endswith('.json'):
                self._save_json(results, output_file)
            elif output_path.lower().endswith('.md'):
                self._save_markdown(results, output_file)
            elif output_path.lower().endswith('.html'):
                self._save_html(results, output_file)
            else:
                # Default to JSON
                self._save_json(results, output_file)

            print(f'\033[1;32m[Saved]:\033[0m Results written to {output_path}')

        except Exception as e:
            print(f'\033[1;31m[Error]:\033[0m Failed to save results: {str(e)}')

    def _save_json(self, results: ScanResults, output_file: Path):
        """Save results as JSON"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results.to_dict(), f, indent=2, default=str, ensure_ascii=False)

    def _save_markdown(self, results: ScanResults, output_file: Path):
        """Save results as Markdown report"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# File Scan Results - {results.session_id}\n\n")

            # Executive Summary
            f.write(f"**Goal:** {results.goal}\n")
            f.write(f"**Timestamp:** {results.scan_timestamp}\n")
            f.write(f"**Focus:** {results.strategy.scan_focus.title()}\n")
            f.write(f"**Analysis Depth:** {results.strategy.analysis_depth.title()}\n\n")

            # Summary
            f.write(f"## Summary\n\n{results.summary}\n\n")

            # Statistics
            stats = results.get_summary_stats()
            f.write("## Statistics\n\n")
            f.write(f"- **Files Found:** {results.total_files}\n")
            f.write(f"- **Files Analyzed:** {results.analyzed_files}\n")
            f.write(f"- **High Relevance Files:** {len(results.high_relevance_files)}\n")
            f.write(f"- **Semantic Insights:** {len(results.semantic_insights)}\n")
            f.write(f"- **Processing Time:** {results.processing_time_seconds:.2f}s\n")
            f.write(f"- **Average Relevance Score:** {stats['avg_relevance_score']:.2f}\n\n")

            # High Relevance Files
            if results.high_relevance_files:
                f.write("## High Relevance Files\n\n")

                for i, analysis in enumerate(results.get_top_files(10)):
                    file_name = Path(analysis.file_path).name
                    f.write(f"### {i + 1}. {file_name} (Score: {analysis.relevance_score:.2f})\n\n")

                    f.write(f"**File Type:** {analysis.file_type}\n")
                    f.write(f"**Size:** {analysis.size_bytes:,} bytes\n")
                    f.write(f"**Summary:** {analysis.content_summary}\n\n")

                    if analysis.key_findings:
                        f.write("**Key Findings:**\n")
                        for finding in analysis.key_findings:
                            f.write(f"- {finding}\n")
                        f.write("\n")

                    if analysis.recommendations:
                        f.write("**Recommendations:**\n")
                        for rec in analysis.recommendations:
                            f.write(f"- {rec}\n")
                        f.write("\n")

                    if analysis.semantic_matches:
                        f.write("**Semantic Matches:**\n")
                        for match in analysis.semantic_matches[:3]:
                            f.write(f"- Query: '{match['query']}' (similarity: {match['similarity']:.2f})\n")
                        f.write("\n")

            # Semantic Insights
            if results.semantic_insights:
                f.write("## Semantic Insights\n\n")

                for insight in results.semantic_insights:
                    f.write(f"### Query: \"{insight.query}\"\n\n")
                    f.write(f"- **Total Matches:** {insight.total_matches}\n")
                    f.write(f"- **High Confidence Matches:** {insight.high_confidence_matches}\n")
                    f.write(
                        f"- **Top Match:** {Path(insight.top_match_file).name} ({insight.top_match_similarity:.2f})\n\n")

                    if insight.related_files:
                        f.write("**Related Files:**\n")
                        for related in insight.related_files[:5]:
                            f.write(f"- {Path(related['file_path']).name} (similarity: {related['similarity']:.2f})\n")
                        f.write("\n")

            # Recommendations
            if results.recommendations:
                f.write("## Recommendations\n\n")
                for i, rec in enumerate(results.recommendations):
                    f.write(f"{i + 1}. {rec}\n")
                f.write("\n")

            # Strategy Details
            f.write("## Scan Strategy\n\n")
            f.write(f"**Keywords:** {', '.join(results.strategy.keywords)}\n")
            f.write(f"**File Priorities:** {', '.join(results.strategy.file_priorities)}\n")
            f.write(f"**Patterns to Find:** {', '.join(results.strategy.patterns_to_find)}\n")
            f.write(f"**Semantic Queries:** {', '.join(results.strategy.semantic_queries)}\n")

    def _save_html(self, results: ScanResults, output_file: Path):
        """Save results as HTML report"""
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>File Scan Results - {results.session_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #007acc; padding-bottom: 10px; }}
        h2 {{ color: #555; border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 30px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: #f8f9fa; padding: 15px; border-radius: 6px; border-left: 4px solid #007acc; }}
        .file-item {{ background: #f8f9fa; margin: 10px 0; padding: 15px; border-radius: 6px; border-left: 4px solid #28a745; }}
        .file-item.medium {{ border-left-color: #ffc107; }}
        .file-item.low {{ border-left-color: #dc3545; }}
        .score {{ font-weight: bold; color: #007acc; }}
        .findings {{ margin: 10px 0; }}
        .findings li {{ margin: 5px 0; }}
        .semantic-insight {{ background: #e3f2fd; padding: 15px; margin: 10px 0; border-radius: 6px; }}
        .recommendations {{ background: #f1f8e9; padding: 20px; border-radius: 6px; }}
        .recommendations li {{ margin: 8px 0; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: 'Monaco', 'Consolas', monospace; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>File Scan Results - {results.session_id}</h1>

        <div class="stats">
            <div class="stat-card">
                <h3>Goal</h3>
                <p>{results.goal}</p>
            </div>
            <div class="stat-card">
                <h3>Files Analyzed</h3>
                <p>{results.analyzed_files} / {results.total_files}</p>
            </div>
            <div class="stat-card">
                <h3>High Relevance</h3>
                <p>{len(results.high_relevance_files)} files</p>
            </div>
            <div class="stat-card">
                <h3>Processing Time</h3>
                <p>{results.processing_time_seconds:.2f}s</p>
            </div>
        </div>

        <h2>Summary</h2>
        <p>{results.summary}</p>
"""

        # Add high relevance files
        if results.high_relevance_files:
            html_content += "<h2>High Relevance Files</h2>\n"

            for analysis in results.get_top_files(10):
                file_name = Path(analysis.file_path).name
                score_class = "high" if analysis.relevance_score >= 0.8 else (
                    "medium" if analysis.relevance_score >= 0.6 else "low")

                html_content += f"""
                <div class="file-item {score_class}">
                    <h3>{file_name} <span class="score">({analysis.relevance_score:.2f})</span></h3>
                    <p><strong>Type:</strong> {analysis.file_type} | <strong>Size:</strong> {analysis.size_bytes:,} bytes</p>
                    <p>{analysis.content_summary}</p>
"""

                if analysis.key_findings:
                    html_content += "<div class='findings'><strong>Key Findings:</strong><ul>\n"
                    for finding in analysis.key_findings:
                        html_content += f"<li>{finding}</li>\n"
                    html_content += "</ul></div>\n"

                html_content += "</div>\n"

        # Add semantic insights
        if results.semantic_insights:
            html_content += "<h2>Semantic Insights</h2>\n"

            for insight in results.semantic_insights:
                html_content += f"""
                <div class="semantic-insight">
                    <h3>Query: "{insight.query}"</h3>
                    <p><strong>Matches:</strong> {insight.total_matches} total, {insight.high_confidence_matches} high confidence</p>
                    <p><strong>Top Match:</strong> {Path(insight.top_match_file).name} (similarity: {insight.top_match_similarity:.2f})</p>
                </div>
"""

        # Add recommendations
        if results.recommendations:
            html_content += """
        <h2>Recommendations</h2>
        <div class="recommendations">
            <ul>
"""
            for rec in results.recommendations:
                html_content += f"<li>{rec}</li>\n"

            html_content += "</ul></div>\n"

        html_content += """
    </div>
</body>
</html>"""

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)


def format_results(results: ScanResults, output_option: Optional[str] = None):
    """Convenience function to format and display results"""
    formatter = ResultFormatter()
    formatter.display_results(results, output_option)