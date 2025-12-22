"""
TEI Evaluation Framework - Visualization Module

This module generates visualizations for TEI LLM evaluation:
- Unified evaluation heatmap (per-file scores across dimensions)
- Processing comparison (grouped bars across processing runs for one model)
- Cross-model comparison (grouped bars across models)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


class TEIEvaluationVisualizer:
    """Handles visualization generation for TEI evaluation reports."""

    def __init__(self, output_dir: Path):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory where visualizations will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style - use available matplotlib style
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except (OSError, KeyError, ValueError):
            # Fallback to basic seaborn style or default
            try:
                sns.set_style("darkgrid")
            except (ValueError, KeyError):
                pass  # Use matplotlib default
        sns.set_palette("husl")

    def create_unified_evaluation_heatmap(self, file_results: List[Dict[str, Any]],
                                          filename: str = "unified_evaluation_heatmap.png") -> Path:
        """
        Create heatmap showing per-file scores across 5 key evaluation aspects.

        Args:
            file_results: List of per-file results from unified reporter
            filename: Output filename

        Returns:
            Path to saved visualization
        """
        # Check if file_results is empty or None
        if not file_results:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, 'No File Results Available',
                   ha='center', va='center', fontsize=18, fontweight='bold',
                   color='#999999', transform=ax.transAxes,
                   bbox=dict(boxstyle='round,pad=1', facecolor='#F0F0F0',
                            edgecolor='#CCCCCC', linewidth=2))
            ax.text(0.5, 0.35, 'Run unified_eval.py to generate file results',
                   ha='center', va='center', fontsize=12, color='#666666',
                   transform=ax.transAxes)
            ax.axis('off')
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            return output_path

        # Extract data for heatmap - 5 key aspects
        files = []
        data_rows = []
        annotation_rows = []  # Store annotations separately for boolean vs score columns

        for result in file_results:
            file_name = result.get('file_name', 'Unknown')
            files.append(file_name)

            # Extract 5 key scores, handling N/A values
            def safe_score(key, default=0):
                val = result.get(key, default)
                if val == "N/A" or val is None:
                    return default
                return float(val) if isinstance(val, (int, float)) else default

            # Helper to convert boolean to score (TRUE = 100, FALSE = 0)
            def bool_to_score(key, default=0):
                val = result.get(key, default)
                if val == "N/A" or val is None:
                    return default
                if isinstance(val, bool):
                    return 100.0 if val else 0.0
                # Try to convert string "True"/"False"
                if isinstance(val, str):
                    if val.lower() == 'true':
                        return 100.0
                    elif val.lower() == 'false':
                        return 0.0
                return default

            # 1. Well-formedness: boolean (TRUE = green/100, FALSE = red/0)
            d0_well_formed = bool_to_score('d0_well_formed')
            d0_annotation = "TRUE" if d0_well_formed == 100 else "FALSE"

            # 2. Content Preservation: score
            d1_score = safe_score('d1_source_fidelity_score')
            d1_annotation = f"{d1_score:.1f}"

            # 3. Schema Validation: boolean (prefer TEI valid, fallback to project valid)
            d2_tei_valid = result.get('d2_tei_valid', False)
            d2_project_valid = result.get('d2_project_valid', False)
            if d2_tei_valid != "N/A" and d2_tei_valid is not None:
                d2_valid = bool(d2_tei_valid) if isinstance(d2_tei_valid, bool) else (str(d2_tei_valid).lower() == 'true')
            elif d2_project_valid != "N/A" and d2_project_valid is not None:
                d2_valid = bool(d2_project_valid) if isinstance(d2_project_valid, bool) else (str(d2_project_valid).lower() == 'true')
            else:
                d2_valid = False
            d2_score = 100.0 if d2_valid else 0.0
            d2_annotation = "TRUE" if d2_valid else "FALSE"

            # 4. Structural Match: score
            d3_score = safe_score('d3_structural_score')
            d3_annotation = f"{d3_score:.1f}"

            # 5. Content Matching: score
            d4_score = safe_score('d4_semantic_score')
            d4_annotation = f"{d4_score:.1f}"

            # 6. Overall Score: score
            overall_score = safe_score('overall_score')
            overall_annotation = f"{overall_score:.1f}" if overall_score > 0 else "N/A"

            row = [
                d0_well_formed,  # 1. Well-formedness (boolean)
                d1_score,  # 2. Content Preservation (score)
                d2_score,  # 3. Schema Validation (boolean)
                d3_score,  # 4. Structural Match (score)
                d4_score,  # 5. Content Matching (score)
                overall_score,  # 6. Overall Score (score)
            ]
            data_rows.append(row)

            annotation_row = [
                d0_annotation,
                d1_annotation,
                d2_annotation,
                d3_annotation,
                d4_annotation,
                overall_annotation,
            ]
            annotation_rows.append(annotation_row)

        # Create DataFrame
        metrics = [
            'Well-Formedness\n(D0)',
            'Source Fidelity Score\n(D1)',
            'Schema Validation\n(D2)',
            'Structural Fidelity Score\n(D3)',
            'Semantic Score\n(D4)',
            'TEI Encoding Score'
        ]

        df = pd.DataFrame(data_rows, index=files, columns=metrics)
        df_annotations = pd.DataFrame(annotation_rows, index=files, columns=metrics)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, max(6, len(files) * 0.4)))

        # Custom colormap: red -> yellow -> green
        cmap = sns.diverging_palette(10, 140, s=80, l=50, as_cmap=True)

        # Create heatmap with custom annotations
        sns.heatmap(df, annot=df_annotations, fmt='', cmap=cmap, center=70,
                   vmin=0, vmax=100, cbar_kws={'label': 'Score (%)'},
                   linewidths=1, linecolor='white', ax=ax,
                   annot_kws={'fontsize': 9, 'fontweight': 'bold'})

        ax.set_title('Unified Evaluation - Dimension Heatmap',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Evaluation Dimensions', fontsize=13, fontweight='bold')
        ax.set_ylabel('Files', fontsize=13, fontweight='bold')

        # Rotate y-axis labels for readability
        plt.yticks(rotation=0, fontsize=9)
        plt.xticks(rotation=0, fontsize=10, ha='center')

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        return output_path


# -------------------------
# Comparison visualization functions
# -------------------------

def _calc_metric_values_for_run(file_results: List[Dict[str, Any]], metric: Dict[str, str]) -> float:
    """Compute a single metric value for one processing run."""
    if not file_results:
        return 0.0

    if metric['type'] == 'binary':
        field = metric['field']
        passed = sum(1 for f in file_results if f.get(field) is True)
        failed = sum(1 for f in file_results if f.get(field) is False)
        evaluated = passed + failed
        return (passed / evaluated) * 100 if evaluated > 0 else 0.0

    if metric['type'] == 'score':
        field = metric['field']
        scores = [f.get(field) for f in file_results if f.get(field) != "N/A" and isinstance(f.get(field), (int, float))]
        return (sum(scores) / len(scores)) if scores else 0.0

    if metric['type'] == 'rate':
        total_exact = sum(f.get('d4_exact_content_matches', 0) for f in file_results if isinstance(f.get('d4_exact_content_matches'), int))
        total_elements = sum(f.get('d4_total_elements', 0) for f in file_results if isinstance(f.get('d4_total_elements'), int))
        return (total_exact / total_elements) * 100 if total_elements > 0 else 0.0

    if metric['type'] == 'gold_rate':
        # Percentage of "Gold TEIs" = wellformed AND (TEI valid OR Project valid) AND overall_score == 100
        def is_true(x: Any) -> bool:
            return x is True
        def is_hundred(x: Any) -> bool:
            return isinstance(x, (int, float)) and abs(float(x) - 100.0) < 1e-6
        def is_gold(f: Dict[str, Any]) -> bool:
            return is_true(f.get('d0_well_formed')) and (is_true(f.get('d2_tei_valid')) or is_true(f.get('d2_project_valid'))) and is_hundred(f.get('overall_score'))
        total = len(file_results)
        if total == 0:
            return 0.0
        count_gold = sum(1 for f in file_results if is_gold(f))
        return (count_gold / total) * 100.0

    return 0.0


def _parse_metadata_label(label: str) -> Dict[str, str]:
    """Parse a metadata label string into a dictionary of fields."""
    result = {
        'Model': '',
        'Temp': '',
        'Processing Time': '',
        'API-Costs': '',
        'Prompt': '',
        'Thinking': ''
    }

    # Split by " | " to get parts
    parts = label.split(' | ')
    if not parts:
        return result

    # First part is model name or timestamp
    result['Model'] = parts[0] if parts else ''

    # Parse remaining parts - handle fixed-width format with colons
    for part in parts[1:]:
        part = part.strip()
        # Check for fixed-width format (key with colon padded to fixed width)
        if part.startswith('Temp:'):
            # Extract value after the fixed-width key (8 chars: "Temp:   ")
            result['Temp'] = part[8:].strip() if len(part) > 8 else ''
        elif part.startswith('Processing Time:'):
            # Extract value after the fixed-width key (20 chars: "Processing Time:  ")
            result['Processing Time'] = part[20:].strip() if len(part) > 20 else ''
        elif part.startswith('API-Costs:'):
            # Extract value after the fixed-width key (11 chars: "API-Costs: ")
            result['API-Costs'] = part[11:].strip() if len(part) > 11 else ''
        elif part.startswith('Prompt:'):
            # Extract value after the fixed-width key (9 chars: "Prompt:  ")
            result['Prompt'] = part[9:].strip() if len(part) > 9 else ''
        elif part.startswith('Thinking:'):
            # Extract value after the fixed-width key (12 chars: "Thinking:  ")
            result['Thinking'] = part[12:].strip() if len(part) > 12 else ''

    return result


def _metrics_for_mode(schema_mode: str) -> List[Dict[str, str]]:
    """
    Define the metrics to display for both charts based on unified summary keys:
    - XML-Wellformedness (binary) -> d0_well_formed
    - Project Validity (binary) -> d2_project_valid
    - Source Fidelity (binary) -> d1_source_fidelity
    - Avg Source Similarity (score) -> d1_source_fidelity_score
    - Structural Fidelity (binary) -> d3_structural_match
    - Avg Structure Similarity (score) -> d3_structural_score
    - Element-Content Match (binary) -> d4_content_match
    - Adjusted Semantic Score (score) -> d4_adjusted_semantic_score
    - Overall Encoding (score) -> overall_score
    - Gold TEIs (rate) -> percentage of gold files as defined above
    """
    metrics = [
        {'type': 'binary', 'field': 'd0_well_formed', 'label': 'XML-\nWellformedness'},
        {'type': 'binary', 'field': 'd2_project_valid', 'label': 'Project\nValidity'},
        {'type': 'binary', 'field': 'd1_source_fidelity', 'label': 'Source\nFidelity'},
        {'type': 'score', 'field': 'd1_source_fidelity_score', 'label': 'Source\nSimilarity'},
        {'type': 'binary', 'field': 'd3_structural_match', 'label': 'Structural\nFidelity'},
        {'type': 'score', 'field': 'd3_structural_score', 'label': 'Structure\nSimilarity'},
        {'type': 'binary', 'field': 'd4_content_match', 'label': 'Exact Element-\nContent Match'},
        {'type': 'rate', 'label': 'Avg Exact\nMatch Rate'},
        {'type': 'score', 'field': 'd4_adjusted_semantic_score', 'label': 'Adjusted\nSemantic Score'},
        {'type': 'score', 'field': 'overall_score', 'label': 'Overall\nEncoding'},
    ]
    # Add Gold TEIs metric right after Overall Encoding
    metrics.append({'type': 'gold_rate', 'label': 'Gold\nTEIs'})
    return metrics


def create_processing_comparison(
    processing_results: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
    schema_mode: str = "both",
    metadata_labels: Optional[Dict[str, str]] = None,
    model_name: Optional[str] = None
) -> Path:
    """
    Create a 4-box dashboard comparing multiple processing runs of one model.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = "processing_comparison.png"

    if not processing_results:
        fig, ax = plt.subplots(figsize=(18, 10))
        ax.axis('off')
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        return output_path

    run_names = list(processing_results.keys())
    num_runs = len(run_names)

    colors_palette = ['#2E86AB', '#A23B72', '#F18F01', '#27AE60', '#E74C3C', '#9B59B6', '#16A085', '#F39C12']
    run_colors = {name: colors_palette[i % len(colors_palette)] for i, name in enumerate(run_names)}
    labels_for_runs = {name: (metadata_labels.get(name, name) if metadata_labels else name) for name in run_names}

    fig = plt.figure(figsize=(18, 11))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.2], hspace=0.35, top=0.92, bottom=0.08, left=0.06, right=1.0)
    top = outer[0].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.2)   # 50/50
    bottom = outer[1].subgridspec(1, 1)  # Full width for Distribution

    # Box 1 (top-left): Binary counts
    ax1 = fig.add_subplot(top[0, 0])
    binary_metrics = [('d0_well_formed', 'D0 Wellformedness\n(XML Valid)')]
    if schema_mode in ['tei', 'both']:
        binary_metrics.append(('d2_tei_valid', 'D2 TEI Validity\n(TEI Schema Valid)'))
    if schema_mode in ['project', 'both']:
        binary_metrics.append(('d2_project_valid', 'D2 Project Validity\n(Schema Valid)'))
    binary_metrics.extend([('d1_source_fidelity', 'D1 Source Fidelity\n(Source Text Unaltered)'), ('d3_structural_match', 'D3 Structural Fidelity\n(XML Structure as in GT)')])
    x1 = np.arange(len(binary_metrics))
    bar_width1 = 0.5 / max(num_runs, 1)
    total_files = {run: len(processing_results[run]) for run in run_names}
    for i, run in enumerate(run_names):
        files = processing_results[run]
        counts = [sum(1 for f in files if f.get(field) is True) for field, _ in binary_metrics]
        offset = (i - num_runs / 2) * bar_width1 + bar_width1 / 2
        bars = ax1.bar(x1 + offset, counts, bar_width1, label=labels_for_runs[run],
                color=run_colors[run], edgecolor='black', linewidth=0.8, alpha=0.9)
        for bar, count in zip(bars, counts):
            if count > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2, count + 0.1, str(count),
                         ha='center', va='bottom', fontsize=6)
    ax1.set_xticks(x1)
    ax1.set_xticklabels([''] * len(binary_metrics))  # Clear default labels
    # Create custom labels with different formatting for main text and parentheses
    for i, (_, lbl) in enumerate(binary_metrics):
        # Split label into main part and parenthetical part
        if '\n(' in lbl:
            parts = lbl.split('\n(', 1)
            main_text = parts[0]
            parenthetical = '(' + parts[1] if len(parts) > 1 else ''
        else:
            main_text = lbl
            parenthetical = ''

        # Position for main text (bold)
        ax1.text(i, -0.05, main_text, transform=ax1.get_xaxis_transform(),
                 ha='center', va='top', fontsize=10, fontweight='bold')
        # Position for parenthetical text (smaller, not bold) - handle multi-line
        if parenthetical:
            # Count lines in parenthetical text to adjust vertical position
            lines = parenthetical.split('\n')
            line_height = 0.06
            for j, line in enumerate(lines):
                ax1.text(i, -0.10 - j * line_height, line, transform=ax1.get_xaxis_transform(),
                         ha='center', va='top', fontsize=8)
    ax1.set_ylabel('Number of Files', fontsize=11, fontweight='bold')
    ax1.set_title('Validation Summary', fontsize=14, fontweight='bold')

    # Adjust ylim to leave space for descriptive text
    max_count = max(total_files.values()) if total_files else 1
    ax1.set_ylim(0, max_count * 1.15)
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.set_axisbelow(True)

    # Box 2 (top-right): Scoreboard (averages)
    ax2 = fig.add_subplot(top[0, 1])
    score_metrics = [
        ('d1_source_fidelity_score', 'D1 Source\nSimilarity\n(String Comparison)'),
        ('d3_structural_score', 'D3 Structure\nSimilarity\n(Normalized XML LCS)'),
        ('d4_adjusted_semantic_score', 'D4 Adjusted\nSemantic Score\n(Element-Content Relation as in GT)'),
        ('overall_score', 'TEI Encoding\nScore'),
    ]
    x2 = np.arange(len(score_metrics))
    bar_width2 = 0.7 / max(num_runs, 1)
    for i, run in enumerate(run_names):
        files = processing_results[run]
        values = []
        for field, _ in score_metrics:
            scores = [f.get(field) for f in files if isinstance(f.get(field), (int, float))]
            values.append((sum(scores) / len(scores)) if scores else 0.0)
        offset = (i - num_runs / 2) * bar_width2 + bar_width2 / 2
        bars = ax2.bar(x2 + offset, values, bar_width2, label=labels_for_runs[run],
                   color=run_colors[run], edgecolor='black', linewidth=0.8, alpha=0.9)
        for bar, val in zip(bars, values):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, val + 1, f'{val:.1f}',
                         ha='center', va='bottom', fontsize=6)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([''] * len(score_metrics))  # Clear default labels
    # Create custom labels with different formatting for main text and parentheses
    for i, (_, lbl) in enumerate(score_metrics):
        # Split label into main part and parenthetical part
        if '\n(' in lbl:
            parts = lbl.rsplit('\n(', 1)  # Split from right to handle multi-line main text
            main_text = parts[0]
            parenthetical = '(' + parts[1] if len(parts) > 1 else ''
        else:
            main_text = lbl
            parenthetical = ''

        # Position for main text (bold) - handle multi-line main text
        main_lines = main_text.split('\n')
        main_line_height = 0.035
        for j, line in enumerate(main_lines):
            ax2.text(i, -0.05 - j * main_line_height, line, transform=ax2.get_xaxis_transform(),
                     ha='center', va='top', fontsize=10, fontweight='bold')
        # Position for parenthetical text (smaller, not bold) - handle multi-line
        if parenthetical:
            lines = parenthetical.split('\n')
            line_height = 0.05
            # Smaller gap between main label and parenthetical part
            start_pos = -0.02 - len(main_lines) * main_line_height - 0.04
            for j, line in enumerate(lines):
                ax2.text(i, start_pos - j * line_height, line, transform=ax2.get_xaxis_transform(),
                         ha='center', va='top', fontsize=8)
    ax2.set_ylabel('Mean (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Scoreboard', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_axisbelow(True)
    # tighten padding inside the right box
    ax2.margins(x=0.02)
    ax2.tick_params(axis='x', pad=2)
    ax2.set_ylabel('Mean (%)', fontsize=11, fontweight='bold', labelpad=6)

    # Box 3 (bottom): Distribution (full width)
    ax3 = fig.add_subplot(bottom[0])
    bands = [
        ('100% GT Overlap', 100.0, 100.0),
        ('91–99', 91.0, 99.9999),
        ('81–90', 81.0, 90.9999),
        ('71–80', 71.0, 80.9999),
        ('61–70', 61.0, 70.9999),
        ('51–60', 51.0, 60.9999),
        ('41–50', 41.0, 50.9999),
        ('31–40', 31.0, 40.9999),
        ('21–30', 21.0, 30.9999),
        ('11–20', 11.0, 20.9999),
        ('1–10', 1.0, 10.9999),
        ('0', 0.0, 0.0),
        ('Not Processed', None, None),  # Special marker for unprocessed files
    ]
    x3 = np.arange(len(bands))
    bar_width3 = 0.7 / max(num_runs, 1)
    def is_true(x: Any) -> bool:
        return x is True
    def is_gold(f: Dict[str, Any]) -> bool:
        def is_hundred(x: Any) -> bool:
            return isinstance(x, (int, float)) and abs(float(x) - 100.0) < 1e-6
        return is_true(f.get('d0_well_formed')) and (is_true(f.get('d2_tei_valid')) or is_true(f.get('d2_project_valid'))) and is_hundred(f.get('overall_score'))
    for i, run in enumerate(run_names):
        files = processing_results[run]
        counts = []
        for label, lo, hi in bands:
            if label == 'Gold TEIs':
                counts.append(sum(1 for f in files if is_gold(f)))
            elif label == 'Not Processed':
                # Count files that were not processed (no valid overall_score or file_not_found)
                not_processed = sum(1 for f in files if (
                    not isinstance(f.get('overall_score'), (int, float)) or
                    f.get('d0_file_not_found') is True or
                    f.get('evaluation_status') == "INVALID"
                ))
                counts.append(not_processed)
            elif lo == hi == 0.0:
                counts.append(sum(1 for f in files if isinstance(f.get('overall_score'), (int, float)) and abs(float(f.get('overall_score')) - 0.0) < 1e-6))
            else:
                counts.append(sum(1 for f in files if isinstance(f.get('overall_score'), (int, float)) and lo <= float(f.get('overall_score')) <= hi))
        offset = (i - num_runs / 2) * bar_width3 + bar_width3 / 2
        bars = ax3.bar(x3 + offset, counts, bar_width3, label=labels_for_runs[run],
                color=run_colors[run], edgecolor='black', linewidth=0.7, alpha=0.9)
        for bar, count in zip(bars, counts):
            if count > 0:
                ax3.text(bar.get_x() + bar.get_width() / 2, count + 0.1, str(count),
                         ha='center', va='bottom', fontsize=6)
    ax3.set_xticks(x3)
    ax3.set_xticklabels([b[0] for b in bands], fontsize=10, fontweight='bold')
    ax3.set_ylabel('Number of Files', fontsize=11, fontweight='bold')
    ax3.set_title('Overall Score Distribution', fontsize=14, fontweight='bold')
    ax3.grid(True, axis='y', alpha=0.3)
    ax3.set_axisbelow(True)
    # Add legend inside the Distribution box in the right corner
    try:
        from matplotlib.patches import Patch

        handles = [
            Patch(facecolor=run_colors[run], edgecolor='black', label=labels_for_runs[run])
            for run in run_names
        ]
        legend_title = f"Model: {model_name}" if model_name else 'Processing Runs'
        ax3.legend(
            handles=handles,
            loc='upper right',
            fontsize=10,
            framealpha=0.95,
            title=legend_title,
            title_fontsize=11,
            borderpad=0.6,
            handletextpad=0.8,
            borderaxespad=0.6,
            columnspacing=0.8,
        )
    except (ImportError, ValueError, KeyError, AttributeError):
        pass  # Legend creation failed, continue without it

    fig.suptitle('Processing Run Comparison - Single Model', fontsize=16, fontweight='bold', y=0.98)

    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return output_path


def create_cross_model_comparison(
    model_results: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
    schema_mode: str = "both",
    metadata_labels: Optional[Dict[str, str]] = None
) -> Path:
    """
    Create a 4-box dashboard comparing multiple models (one run per model).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = sorted(model_results.keys())
    filename_suffix = '_'.join(model_names)
    filename = f"cross_model_comp_{filename_suffix}.png"

    if not model_results:
        fig, ax = plt.subplots(figsize=(18, 10))
        ax.axis('off')
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        return output_path

    num_models = len(model_names)

    colors_palette = ['#2E86AB', '#A23B72', '#F18F01', '#27AE60', '#E74C3C', '#9B59B6', '#16A085', '#F39C12']
    model_colors = {name: colors_palette[i % len(colors_palette)] for i, name in enumerate(model_names)}
    labels_for_models = {name: (metadata_labels.get(name, name) if metadata_labels else name) for name in model_names}

    fig = plt.figure(figsize=(18, 11))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.2], hspace=0.35, top=0.92, bottom=0.08, left=0.06, right=1.0)
    top = outer[0].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.2)   # 50/50
    bottom = outer[1].subgridspec(1, 1)  # Full width for Distribution

    # Box 1 (top-left): Binary counts
    ax1 = fig.add_subplot(top[0, 0])
    # Use the same metric labels and explanations as processing comparison
    binary_metrics = [('d0_well_formed', 'D0 Wellformedness\n(XML Valid)')]
    if schema_mode in ['tei', 'both']:
        binary_metrics.append(('d2_tei_valid', 'D2 TEI Validity\n(TEI Schema Valid)'))
    if schema_mode in ['project', 'both']:
        binary_metrics.append(('d2_project_valid', 'D2 Project Validity\n(Schema Valid)'))
    binary_metrics.extend([
        ('d1_source_fidelity', 'D1 Source Fidelity\n(Source Text Unaltered)'),
        ('d3_structural_match', 'D3 Structural Fidelity\n(XML Structure as in GT)'),
    ])
    x1 = np.arange(len(binary_metrics))
    bar_width1 = 0.5 / max(num_models, 1)
    total_files = {model: len(model_results[model]) for model in model_names}
    for i, model in enumerate(model_names):
        files = model_results[model]
        counts = [sum(1 for f in files if f.get(field) is True) for field, _ in binary_metrics]
        offset = (i - num_models / 2) * bar_width1 + bar_width1 / 2
        bars = ax1.bar(x1 + offset, counts, bar_width1, label=labels_for_models[model],
                color=model_colors[model], edgecolor='black', linewidth=0.8, alpha=0.9)
        for bar, count in zip(bars, counts):
            if count > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2, count + 0.1, str(count),
                         ha='center', va='bottom', fontsize=6)
    ax1.set_xticks(x1)
    ax1.set_xticklabels([''] * len(binary_metrics))  # Clear default labels
    # Create custom labels with different formatting for main text and parentheses
    for i, (_, lbl) in enumerate(binary_metrics):
        # Split label into main part and parenthetical part
        if '\n(' in lbl:
            parts = lbl.split('\n(', 1)
            main_text = parts[0]
            parenthetical = '(' + parts[1] if len(parts) > 1 else ''
        else:
            main_text = lbl
            parenthetical = ''

        # Position for main text (bold)
        ax1.text(i, -0.05, main_text, transform=ax1.get_xaxis_transform(),
                 ha='center', va='top', fontsize=10, fontweight='bold')
        # Position for parenthetical text (smaller, not bold) - handle multi-line
        if parenthetical:
            # Count lines in parenthetical text to adjust vertical position
            lines = parenthetical.split('\n')
            line_height = 0.06
            for j, line in enumerate(lines):
                ax1.text(i, -0.10 - j * line_height, line, transform=ax1.get_xaxis_transform(),
                         ha='center', va='top', fontsize=7)
    ax1.set_ylabel('Number of Files', fontsize=11, fontweight='bold')
    ax1.set_title('Validation Summary', fontsize=14, fontweight='bold')
    # Adjust ylim to leave space for labels
    max_count = max(total_files.values()) if total_files else 1
    ax1.set_ylim(0, max_count * 1.15)
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.set_axisbelow(True)

    # Box 2 (top-right): Scoreboard (averages)
    ax2 = fig.add_subplot(top[0, 1])
    # Use the same metric labels and explanations as processing comparison
    score_metrics = [
        ('d1_source_fidelity_score', 'D1 Source\nSimilarity\n(String Comparison)'),
        ('d3_structural_score', 'D3 Structural\nSimilarity\n(Normalized XML LCS)'),
        ('d4_adjusted_semantic_score', 'D4 Adjusted\nSemantic Score\n(Element-Content Relation as in GT)'),
        ('overall_score', 'TEI Encoding\nScore'),
    ]
    x2 = np.arange(len(score_metrics))
    bar_width2 = 0.7 / max(num_models, 1)
    for i, model in enumerate(model_names):
        files = model_results[model]
        values = []
        for field, _ in score_metrics:
            scores = [f.get(field) for f in files if isinstance(f.get(field), (int, float))]
            values.append((sum(scores) / len(scores)) if scores else 0.0)
        offset = (i - num_models / 2) * bar_width2 + bar_width2 / 2
        bars = ax2.bar(x2 + offset, values, bar_width2, label=labels_for_models[model],
                   color=model_colors[model], edgecolor='black', linewidth=0.8, alpha=0.9)
        for bar, val in zip(bars, values):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, val + 1, f'{val:.1f}',
                         ha='center', va='bottom', fontsize=6)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([''] * len(score_metrics))  # Clear default labels
    # Create custom labels with different formatting for main text and parentheses
    for i, (_, lbl) in enumerate(score_metrics):
        # Split label into main part and parenthetical part
        if '\n(' in lbl:
            parts = lbl.rsplit('\n(', 1)  # Split from right to handle multi-line main text
            main_text = parts[0]
            parenthetical = '(' + parts[1] if len(parts) > 1 else ''
        else:
            main_text = lbl
            parenthetical = ''

        # Position for main text (bold) - handle multi-line main text
        main_lines = main_text.split('\n')
        main_line_height = 0.035
        for j, line in enumerate(main_lines):
            ax2.text(i, -0.05 - j * main_line_height, line, transform=ax2.get_xaxis_transform(),
                     ha='center', va='top', fontsize=10, fontweight='bold')
        # Position for parenthetical text (smaller, not bold) - handle multi-line
        if parenthetical:
            lines = parenthetical.split('\n')
            line_height = 0.05
            start_pos = -0.02 - len(main_lines) * main_line_height - 0.04
            for j, line in enumerate(lines):
                ax2.text(i, start_pos - j * line_height, line, transform=ax2.get_xaxis_transform(),
                         ha='center', va='top', fontsize=7)
    ax2.set_ylabel('Mean (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Scoreboard', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_axisbelow(True)
    # tighten padding inside the right box
    ax2.margins(x=0.02)
    ax2.tick_params(axis='x', pad=2)
    ax2.set_ylabel('Mean (%)', fontsize=11, fontweight='bold', labelpad=6)

    # Box 3 (bottom): Distribution (full width)
    ax3 = fig.add_subplot(bottom[0])
    bands = [
        ('100% GT Overlap', 100.0, 100.0),
        ('91–99', 91.0, 99.9999),
        ('81–90', 81.0, 90.9999),
        ('71–80', 71.0, 80.9999),
        ('61–70', 61.0, 70.9999),
        ('51–60', 51.0, 60.9999),
        ('41–50', 41.0, 50.9999),
        ('31–40', 31.0, 40.9999),
        ('21–30', 21.0, 30.9999),
        ('11–20', 11.0, 20.9999),
        ('1–10', 1.0, 10.9999),
        ('0', 0.0, 0.0),
        ('Not Processed', None, None),  # Special marker for unprocessed files
    ]
    x3 = np.arange(len(bands))
    bar_width3 = 0.7 / max(num_models, 1)
    def is_true(x: Any) -> bool:
        return x is True
    def is_gold(f: Dict[str, Any]) -> bool:
        def is_hundred(x: Any) -> bool:
            return isinstance(x, (int, float)) and abs(float(x) - 100.0) < 1e-6
        return is_true(f.get('d0_well_formed')) and (is_true(f.get('d2_tei_valid')) or is_true(f.get('d2_project_valid'))) and is_hundred(f.get('overall_score'))
    for i, model in enumerate(model_names):
        files = model_results[model]
        counts = []
        for label, lo, hi in bands:
            if label == '100% GT Overlap':
                counts.append(sum(1 for f in files if is_gold(f)))
            elif label == 'Not Processed':
                # Count files that were not processed (no valid overall_score or file_not_found)
                not_processed = sum(1 for f in files if (
                    not isinstance(f.get('overall_score'), (int, float)) or
                    f.get('d0_file_not_found') is True or
                    f.get('evaluation_status') == "INVALID"
                ))
                counts.append(not_processed)
            elif lo == hi == 0.0:
                counts.append(sum(1 for f in files if isinstance(f.get('overall_score'), (int, float)) and abs(float(f.get('overall_score')) - 0.0) < 1e-6))
            else:
                counts.append(sum(1 for f in files if isinstance(f.get('overall_score'), (int, float)) and lo <= float(f.get('overall_score')) <= hi))
        offset = (i - num_models / 2) * bar_width3 + bar_width3 / 2
        bars = ax3.bar(x3 + offset, counts, bar_width3, label=labels_for_models[model],
                color=model_colors[model], edgecolor='black', linewidth=0.7, alpha=0.9)
        for bar, count in zip(bars, counts):
            if count > 0:
                ax3.text(bar.get_x() + bar.get_width() / 2, count + 0.1, str(count),
                         ha='center', va='bottom', fontsize=6)
    ax3.set_xticks(x3)
    ax3.set_xticklabels([b[0] for b in bands], fontsize=10, fontweight='bold')
    ax3.set_ylabel('Number of Files', fontsize=11, fontweight='bold')
    ax3.set_title('Overall Score Distribution', fontsize=14, fontweight='bold')
    ax3.grid(True, axis='y', alpha=0.3)
    ax3.set_axisbelow(True)
    # Add metadata table instead of legend (styled similar to technical details)
    try:
        # Parse metadata for each model
        table_data = []
        headers = ['Model', 'Prompt', 'Temp', 'Time', 'API-Costs']

        for model_name in model_names:
            label = labels_for_models.get(model_name, model_name)
            parsed = _parse_metadata_label(label)
            # Get model name and append Thinking indicator if present
            display_model_name = parsed.get('Model', model_name)
            thinking_value = parsed.get('Thinking', '').strip().upper()
            if thinking_value == 'TRUE':
                display_model_name += ' (Thinking)'
            elif thinking_value == 'FALSE':
                display_model_name += ' (No-thinking)'
            # Use short names for table
            api_costs = parsed.get('API-Costs', '').strip()
            row = [
                display_model_name,
                parsed.get('Prompt', 'N/A'),
                parsed.get('Temp', 'N/A'),
                parsed.get('Processing Time', 'N/A'),
                api_costs if api_costs else 'N/A'
            ]
            table_data.append(row)

        # Create table
        table = ax3.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='upper right',
                         bbox=[0.63, 0.72, 0.35, min(0.3, 0.05 * (len(model_names) + 1))])

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)

        # Style header row
        for i in range(len(headers)):
            cell = table[(0, i)]
            cell.set_facecolor('white')
            cell.set_text_props(weight='bold', fontsize=9)
            cell.set_edgecolor('black')
            cell.set_linewidth(1.5)

        # Style data rows with model colors
        for row_idx, model_name in enumerate(model_names, start=1):
            row_color = model_colors[model_name]
            for col_idx in range(len(headers)):
                cell = table[(row_idx, col_idx)]
                cell.set_facecolor(row_color)
                cell.set_alpha(0.8)
                cell.set_edgecolor('black')
                cell.set_linewidth(1.0)
                cell.set_text_props(fontsize=8)

        # Style table frame
        table.auto_set_column_width(col=list(range(len(headers))))

    except (ValueError, KeyError, AttributeError, TypeError):
        # Fallback to basic legend if table creation fails
        try:
            from matplotlib.patches import Patch

            handles = [
                Patch(facecolor=model_colors[m], edgecolor='black', label=labels_for_models[m])
                for m in model_names
            ]
            ax3.legend(
                handles=handles,
                loc='upper right',
                fontsize=10,
                framealpha=0.95,
                title='Models',
                title_fontsize=11,
                borderpad=0.6,
                handletextpad=0.8,
                borderaxespad=0.6,
                columnspacing=0.8,
            )
        except (ImportError, ValueError, KeyError, AttributeError):
            pass  # Legend creation also failed, continue without it

    fig.suptitle('Cross-Model Comparison', fontsize=16, fontweight='bold', y=0.98)

    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return output_path
