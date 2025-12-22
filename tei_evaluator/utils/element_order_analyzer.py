# utils/element_order_analyzer.py

from typing import List, Dict, Any, Tuple
from lxml import etree

class ElementOrderAnalyzer:
    """Analyzes element order compliance against reference"""
    
    def __init__(self, ignore_elements: set = None):
        self.ignore_elements = ignore_elements or {'note'}
    
    def analyze_element_order(self, xml_body: etree._Element, ref_body: etree._Element) -> Dict[str, Any]:
        """
        Compare element order between XML and reference
        
        Returns comprehensive order analysis including:
        - Order similarity percentage (LCS-based)
        - Sequence alignment details
        - Out-of-order elements list
        - Position shift analysis
        """
        
        # Extract element sequences
        xml_sequence = self._extract_element_sequence(xml_body)
        ref_sequence = self._extract_element_sequence(ref_body)
        
        # Perform multiple order analyses
        analysis = {
            "xml_sequence": xml_sequence,
            "ref_sequence": ref_sequence,
            "sequence_lengths": {
                "xml": len(xml_sequence),
                "ref": len(ref_sequence)
            }
        }
        
        # 1. Longest Common Subsequence analysis
        lcs_analysis = self._lcs_analysis(xml_sequence, ref_sequence)
        analysis["lcs_analysis"] = lcs_analysis
        
        # 2. Position-based analysis (for same-length sequences)
        if len(xml_sequence) == len(ref_sequence):
            position_analysis = self._position_analysis(xml_sequence, ref_sequence)
            analysis["position_analysis"] = position_analysis
        else:
            analysis["position_analysis"] = {"available": False, "reason": "different_lengths"}
        
        # 3. Element displacement analysis
        displacement_analysis = self._displacement_analysis(xml_sequence, ref_sequence)
        analysis["displacement_analysis"] = displacement_analysis
        
        # 4. Overall order compliance score
        overall_score = self._calculate_order_score(analysis)
        analysis["order_compliance_score"] = overall_score
        
        return analysis
    
    def _extract_element_sequence(self, element: etree._Element, path: str = "") -> List[str]:
        """Extract ordered sequence of element names"""
        sequence = []
        
        # Get element name without namespace
        element_name = element.tag.split('}')[-1] if '}' in element.tag else element.tag
        
        # Skip ignored elements
        if element_name in self.ignore_elements:
            # Process children but don't add this element
            for child in element:
                sequence.extend(self._extract_element_sequence(child, f"{path}/{element_name}"))
            return sequence
        
        # Add current element with path context
        current_path = f"{path}/{element_name}" if path else element_name
        sequence.append(current_path)
        
        # Process children in document order
        for child in element:
            sequence.extend(self._extract_element_sequence(child, current_path))
        
        return sequence
    
    def _lcs_analysis(self, xml_seq: List[str], ref_seq: List[str]) -> Dict[str, Any]:
        """Longest Common Subsequence analysis for order similarity"""
        
        # Calculate LCS length
        lcs_length = self._longest_common_subsequence_length(xml_seq, ref_seq)
        
        # Calculate order similarity percentage
        max_length = max(len(xml_seq), len(ref_seq))
        if max_length == 0:
            order_similarity = 100.0
        else:
            order_similarity = (lcs_length / max_length) * 100
        
        # Get actual LCS for detailed analysis
        lcs_sequence = self._get_lcs_sequence(xml_seq, ref_seq)
        
        return {
            "lcs_length": lcs_length,
            "max_sequence_length": max_length,
            "order_similarity_percentage": order_similarity,
            "common_subsequence": lcs_sequence[:10],  # First 10 for brevity
            "interpretation": self._interpret_order_similarity(order_similarity)
        }
    
    def _longest_common_subsequence_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate LCS length using dynamic programming"""
        m, n = len(seq1), len(seq2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _get_lcs_sequence(self, seq1: List[str], seq2: List[str]) -> List[str]:
        """Get the actual LCS sequence"""
        m, n = len(seq1), len(seq2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        # Backtrack to get LCS
        lcs = []
        i, j = m, n
        while i > 0 and j > 0:
            if seq1[i-1] == seq2[j-1]:
                lcs.append(seq1[i-1])
                i -= 1
                j -= 1
            elif dp[i-1][j] > dp[i][j-1]:
                i -= 1
            else:
                j -= 1
        
        return lcs[::-1]  # Reverse to get correct order
    
    def _position_analysis(self, xml_seq: List[str], ref_seq: List[str]) -> Dict[str, Any]:
        """Compare position-by-position for same-length sequences"""
        
        if len(xml_seq) != len(ref_seq):
            return {"available": False, "reason": "different_lengths"}
        
        correct_positions = 0
        misplaced_elements = []
        
        for i, (xml_elem, ref_elem) in enumerate(zip(xml_seq, ref_seq)):
            if xml_elem == ref_elem:
                correct_positions += 1
            else:
                misplaced_elements.append({
                    "position": i,
                    "expected": ref_elem,
                    "found": xml_elem
                })
        
        position_accuracy = (correct_positions / len(xml_seq)) * 100 if xml_seq else 100
        
        return {
            "available": True,
            "total_positions": len(xml_seq),
            "correct_positions": correct_positions,
            "position_accuracy_percentage": position_accuracy,
            "misplaced_elements": misplaced_elements[:10],  # First 10 for brevity
            "total_misplacements": len(misplaced_elements)
        }
    
    def _displacement_analysis(self, xml_seq: List[str], ref_seq: List[str]) -> Dict[str, Any]:
        """Analyze how far elements are displaced from their expected positions"""
        
        # Create position maps
        ref_positions = {elem: i for i, elem in enumerate(ref_seq)}
        xml_positions = {elem: i for i, elem in enumerate(xml_seq)}
        
        displacements = []
        common_elements = set(xml_seq) & set(ref_seq)
        
        for elem in common_elements:
            ref_pos = ref_positions[elem]
            xml_pos = xml_positions[elem]
            displacement = abs(xml_pos - ref_pos)
            
            if displacement > 0:
                displacements.append({
                    "element": elem,
                    "expected_position": ref_pos,
                    "actual_position": xml_pos,
                    "displacement": displacement
                })
        
        # Sort by displacement magnitude
        displacements.sort(key=lambda x: x["displacement"], reverse=True)
        
        # Calculate average displacement
        avg_displacement = sum(d["displacement"] for d in displacements) / len(displacements) if displacements else 0
        max_displacement = max(d["displacement"] for d in displacements) if displacements else 0
        
        return {
            "total_displaced_elements": len(displacements),
            "average_displacement": avg_displacement,
            "max_displacement": max_displacement,
            "most_displaced_elements": displacements[:5],  # Top 5 most displaced
            "displacement_distribution": self._displacement_distribution(displacements)
        }
    
    def _displacement_distribution(self, displacements: List[Dict]) -> Dict[str, int]:
        """Categorize displacements by magnitude"""
        distribution = {
            "minor_displacement_1-2": 0,
            "moderate_displacement_3-5": 0,
            "major_displacement_6-10": 0,
            "severe_displacement_10+": 0
        }
        
        for d in displacements:
            displacement = d["displacement"]
            if displacement <= 2:
                distribution["minor_displacement_1-2"] += 1
            elif displacement <= 5:
                distribution["moderate_displacement_3-5"] += 1
            elif displacement <= 10:
                distribution["major_displacement_6-10"] += 1
            else:
                distribution["severe_displacement_10+"] += 1
        
        return distribution
    
    def _calculate_order_score(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall order compliance score"""
        
        lcs_analysis = analysis["lcs_analysis"]
        position_analysis = analysis.get("position_analysis", {})
        displacement_analysis = analysis["displacement_analysis"]
        
        # Primary score from LCS analysis
        order_similarity = lcs_analysis["order_similarity_percentage"]
        
        # Adjust based on displacement severity
        severe_displacements = displacement_analysis["displacement_distribution"]["severe_displacement_10+"]
        major_displacements = displacement_analysis["displacement_distribution"]["major_displacement_6-10"]
        
        # Penalty for severe displacements
        displacement_penalty = (severe_displacements * 5) + (major_displacements * 2)
        adjusted_score = max(0, order_similarity - displacement_penalty)
        
        return {
            "base_order_similarity": order_similarity,
            "displacement_penalty": displacement_penalty,
            "final_order_score": adjusted_score,
            "interpretation": self._interpret_order_score(adjusted_score)
        }
    
    def _interpret_order_similarity(self, similarity: float) -> str:
        """Interpret order similarity percentage"""
        if similarity >= 95:
            return "Excellent order compliance"
        elif similarity >= 85:
            return "Good order compliance with minor variations"
        elif similarity >= 70:
            return "Moderate order compliance with some deviations"
        elif similarity >= 50:
            return "Poor order compliance with significant deviations"
        else:
            return "Very poor order compliance - major restructuring"
    
    def _interpret_order_score(self, score: float) -> str:
        """Interpret final order compliance score"""
        if score >= 90:
            return "High order compliance"
        elif score >= 75:
            return "Acceptable order compliance"
        elif score >= 60:
            return "Marginal order compliance"
        elif score >= 40:
            return "Poor order compliance"
        else:
            return "Unacceptable order compliance"
    
    def format_order_analysis_report(self, analysis: Dict[str, Any]) -> str:
        """Format order analysis into readable report"""
        report = []
        
        # Header
        report.append("📋 ELEMENT ORDER ANALYSIS:")
        
        # Sequence info
        seq_lengths = analysis["sequence_lengths"]
        report.append(f"Sequence lengths: XML={seq_lengths['xml']}, Reference={seq_lengths['ref']}")
        
        # LCS Analysis
        lcs = analysis["lcs_analysis"]
        report.append(f"\n🔗 ORDER SIMILARITY:")
        report.append(f"Order Similarity: {lcs['order_similarity_percentage']:.1f}%")
        report.append(f"Common Subsequence Length: {lcs['lcs_length']}/{lcs['max_sequence_length']}")
        report.append(f"Assessment: {lcs['interpretation']}")
        
        # Position Analysis (if available)
        pos_analysis = analysis.get("position_analysis", {})
        if pos_analysis.get("available"):
            report.append(f"\n📍 POSITION ACCURACY:")
            report.append(f"Correct Positions: {pos_analysis['correct_positions']}/{pos_analysis['total_positions']}")
            report.append(f"Position Accuracy: {pos_analysis['position_accuracy_percentage']:.1f}%")
            if pos_analysis['total_misplacements'] > 0:
                report.append(f"Misplaced Elements: {pos_analysis['total_misplacements']}")
        
        # Displacement Analysis
        disp = analysis["displacement_analysis"]
        if disp["total_displaced_elements"] > 0:
            report.append(f"\n🔄 ELEMENT DISPLACEMENTS:")
            report.append(f"Displaced Elements: {disp['total_displaced_elements']}")
            report.append(f"Average Displacement: {disp['average_displacement']:.1f} positions")
            report.append(f"Maximum Displacement: {disp['max_displacement']} positions")
            
            # Show displacement distribution
            dist = disp["displacement_distribution"]
            if any(dist.values()):
                report.append("Displacement Breakdown:")
                for category, count in dist.items():
                    if count > 0:
                        category_clean = category.replace("_", " ").replace("-", "-")
                        report.append(f"  {category_clean}: {count}")
        
        # Overall Score
        score = analysis["order_compliance_score"]
        report.append(f"\n[TARGET] OVERALL ORDER COMPLIANCE:")
        report.append(f"Final Score: {score['final_order_score']:.1f}%")
        report.append(f"Assessment: {score['interpretation']}")
        
        return "\n".join(report)