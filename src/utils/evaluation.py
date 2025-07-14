"""
Evaluation utilities for synthetic data quality assessment.
Supports both manual annotation and automatic evaluation metrics.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from sklearn.metrics import cohen_kappa_score
import json
from datetime import datetime
import re

@dataclass
class ManualAnnotation:
    """Structure for manual annotation results."""
    article_id: str
    annotator_id: str
    fact_extraction_quality: str  # "appropriate", "inappropriate", "in-between"
    fact_modification_quality: str  # "appropriate", "inappropriate", "in-between"
    synthetic_article_quality: str  # "appropriate", "inappropriate", "in-between"
    comments: str = ""
    timestamp: str = ""

@dataclass
class AutomaticEvaluation:
    """Structure for automatic evaluation metrics."""
    correctness: float  # Factual accuracy score
    coherence: float    # Text coherence score
    dissimilarity: float  # Dissimilarity from original

class EvaluationManager:
    """Manages both manual and automatic evaluation of synthetic data."""
    
    def __init__(self):
        self.manual_annotations = []
        self.automatic_evaluations = []
    
    def create_annotation_template(self, synthetic_results: List[Dict], 
                                 output_file: str = "annotation_template.csv") -> pd.DataFrame:
        """
        Create template for manual annotation.
        
        Args:
            synthetic_results: List of synthetic data generation results
            output_file: Output CSV file for annotations
        """
        annotation_data = []
        
        for i, result in enumerate(synthetic_results):
            annotation_data.append({
                'article_id': f"article_{i+1}",
                'original_text': result.get('original_article', ''),
                'extracted_facts': json.dumps(result.get('extracted_facts', []), indent=2),
                'modified_facts': json.dumps(result.get('modified_facts', []), indent=2),
                'synthetic_text': result.get('modified_article', ''),
                'fact_extraction_quality': '',  # To be filled by annotators
                'fact_modification_quality': '',  # To be filled by annotators  
                'synthetic_article_quality': '',  # To be filled by annotators
                'annotator_id': '',  # To be filled by annotators
                'comments': ''  # To be filled by annotators
            })
        
        df = pd.DataFrame(annotation_data)
        df.to_csv(output_file, index=False)
        
        print(f"Created annotation template with {len(annotation_data)} articles")
        print(f"Saved to: {output_file}")
        print("\nInstructions for annotators:")
        print("1. Fill in 'annotator_id' with your name/ID")
        print("2. Rate each quality aspect as: 'appropriate', 'inappropriate', or 'in-between'")
        print("3. Add comments explaining your ratings")
        print("4. Save the file when complete")
        
        return df
    
    def load_annotations(self, annotation_file: str) -> List[ManualAnnotation]:
        """Load manual annotations from CSV file."""
        df = pd.read_csv(annotation_file)
        annotations = []
        
        for _, row in df.iterrows():
            if pd.notna(row.get('annotator_id', '')):  # Only process completed annotations
                annotation = ManualAnnotation(
                    article_id=row['article_id'],
                    annotator_id=row['annotator_id'],
                    fact_extraction_quality=row.get('fact_extraction_quality', ''),
                    fact_modification_quality=row.get('fact_modification_quality', ''),
                    synthetic_article_quality=row.get('synthetic_article_quality', ''),
                    comments=row.get('comments', ''),
                    timestamp=datetime.now().isoformat()
                )
                annotations.append(annotation)
        
        self.manual_annotations = annotations
        return annotations
    
    def calculate_inter_annotator_agreement(self, annotations: List[ManualAnnotation]) -> Dict[str, float]:
        """Calculate inter-annotator agreement using Cohen's Kappa."""
        if len(annotations) < 2:
            print("Need at least 2 annotators for agreement calculation")
            return {}
        
        # Group annotations by article
        article_annotations = {}
        for ann in annotations:
            if ann.article_id not in article_annotations:
                article_annotations[ann.article_id] = []
            article_annotations[ann.article_id].append(ann)
        
        # Calculate agreement for each quality aspect
        quality_aspects = ['fact_extraction_quality', 'fact_modification_quality', 'synthetic_article_quality']
        agreements = {}
        
        for aspect in quality_aspects:
            # Get ratings for articles with multiple annotators
            paired_ratings = []
            
            for article_id, anns in article_annotations.items():
                if len(anns) >= 2:
                    # Take first two annotators for simplicity
                    rating1 = getattr(anns[0], aspect)
                    rating2 = getattr(anns[1], aspect)
                    if rating1 and rating2:  # Both rated
                        paired_ratings.append((rating1, rating2))
            
            if len(paired_ratings) > 0:
                # Convert to numeric for kappa calculation
                label_map = {'inappropriate': 0, 'in-between': 1, 'appropriate': 2}
                ratings1 = [label_map.get(r[0], 1) for r in paired_ratings]
                ratings2 = [label_map.get(r[1], 1) for r in paired_ratings]
                
                kappa = cohen_kappa_score(ratings1, ratings2)
                agreements[aspect] = kappa
        
        return agreements
    
    def analyze_manual_annotations(self, annotations: List[ManualAnnotation]) -> Dict[str, Any]:
        """Analyze manual annotation results."""
        if not annotations:
            return {}
        
        # Count ratings for each quality aspect
        aspects = ['fact_extraction_quality', 'fact_modification_quality', 'synthetic_article_quality']
        analysis = {}
        
        for aspect in aspects:
            ratings = [getattr(ann, aspect) for ann in annotations if getattr(ann, aspect)]
            if ratings:
                rating_counts = pd.Series(ratings).value_counts()
                analysis[aspect] = {
                    'total_ratings': len(ratings),
                    'appropriate': rating_counts.get('appropriate', 0),
                    'inappropriate': rating_counts.get('inappropriate', 0),
                    'in_between': rating_counts.get('in-between', 0),
                    'appropriate_percentage': (rating_counts.get('appropriate', 0) / len(ratings)) * 100
                }
        
        return analysis
    
    def calculate_automatic_metrics(self, original_text: str, synthetic_text: str, 
                                  extracted_facts: List[Dict], modified_facts: List[Dict]) -> AutomaticEvaluation:
        """Calculate automatic evaluation metrics."""
        
        # 1. Correctness: Check if facts were properly incorporated
        correctness = self._calculate_correctness(synthetic_text, modified_facts)
        
        # 2. Coherence: Text readability and flow
        coherence = self._calculate_coherence(synthetic_text)
        
        # 3. Dissimilarity: How different synthetic is from original
        dissimilarity = self._calculate_dissimilarity(original_text, synthetic_text)
        
        return AutomaticEvaluation(
            correctness=correctness,
            coherence=coherence,
            dissimilarity=dissimilarity
        )
    
    def _calculate_correctness(self, synthetic_text: str, modified_facts: List[Dict]) -> float:
        """Calculate how well modified facts were incorporated."""
        if not modified_facts:
            return 0.0
        
        incorporated_count = 0
        for fact in modified_facts:
            specific_data = fact.get('specific_data', '')
            if specific_data and specific_data.lower() in synthetic_text.lower():
                incorporated_count += 1
        
        return incorporated_count / len(modified_facts)
    
    def _calculate_coherence(self, text: str) -> float:
        """Calculate text coherence based on basic readability metrics."""
        if not text:
            return 0.0
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) == 0:
            return 0.0
        
        # Simple coherence measures
        avg_sentence_length = np.mean([len(s.split()) for s in sentences])
        
        # Penalize very short or very long sentences
        coherence_score = 1.0
        if avg_sentence_length < 5:
            coherence_score -= 0.3
        elif avg_sentence_length > 30:
            coherence_score -= 0.2
        
        # Check for repeated words (might indicate poor generation)
        words = text.lower().split()
        unique_words = set(words)
        repetition_ratio = len(words) / len(unique_words) if unique_words else 1
        
        if repetition_ratio > 2.0:
            coherence_score -= 0.2
        
        return max(0.0, min(1.0, coherence_score))
    
    def _calculate_dissimilarity(self, original_text: str, synthetic_text: str) -> float:
        """Calculate dissimilarity between original and synthetic text."""
        if not original_text or not synthetic_text:
            return 0.0
        
        # Simple word-based dissimilarity
        original_words = set(original_text.lower().split())
        synthetic_words = set(synthetic_text.lower().split())
        
        if not original_words:
            return 1.0
        
        intersection = original_words.intersection(synthetic_words)
        union = original_words.union(synthetic_words)
        
        # Jaccard dissimilarity
        jaccard_similarity = len(intersection) / len(union) if union else 0
        dissimilarity = 1 - jaccard_similarity
        
        return dissimilarity
    
    def generate_evaluation_report(self, manual_annotations: List[ManualAnnotation] = None,
                                 automatic_evaluations: List[AutomaticEvaluation] = None) -> str:
        """Generate comprehensive evaluation report."""
        report = "SYNTHETIC DATA EVALUATION REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Manual evaluation section
        if manual_annotations:
            report += "MANUAL EVALUATION RESULTS\n"
            report += "-" * 30 + "\n"
            
            analysis = self.analyze_manual_annotations(manual_annotations)
            for aspect, results in analysis.items():
                report += f"\n{aspect.replace('_', ' ').title()}:\n"
                report += f"  Total ratings: {results['total_ratings']}\n"
                report += f"  Appropriate: {results['appropriate']} ({results['appropriate_percentage']:.1f}%)\n"
                report += f"  In-between: {results['in_between']}\n"
                report += f"  Inappropriate: {results['inappropriate']}\n"
            
            # Inter-annotator agreement
            agreements = self.calculate_inter_annotator_agreement(manual_annotations)
            if agreements:
                report += "\nInter-Annotator Agreement (Cohen's Kappa):\n"
                for aspect, kappa in agreements.items():
                    report += f"  {aspect.replace('_', ' ').title()}: {kappa:.3f}\n"
        
        # Automatic evaluation section
        if automatic_evaluations:
            report += "\n\nAUTOMATIC EVALUATION RESULTS\n"
            report += "-" * 30 + "\n"
            
            correctness_scores = [eval.correctness for eval in automatic_evaluations]
            coherence_scores = [eval.coherence for eval in automatic_evaluations]
            dissimilarity_scores = [eval.dissimilarity for eval in automatic_evaluations]
            
            report += f"\nCorrectness (fact incorporation):\n"
            report += f"  Mean: {np.mean(correctness_scores):.3f}\n"
            report += f"  Std:  {np.std(correctness_scores):.3f}\n"
            
            report += f"\nCoherence (text quality):\n"
            report += f"  Mean: {np.mean(coherence_scores):.3f}\n"
            report += f"  Std:  {np.std(coherence_scores):.3f}\n"
            
            report += f"\nDissimilarity (from original):\n"
            report += f"  Mean: {np.mean(dissimilarity_scores):.3f}\n"
            report += f"  Std:  {np.std(dissimilarity_scores):.3f}\n"
        
        return report
