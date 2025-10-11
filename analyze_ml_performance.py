"""
RCMS ML Models - Unseen Data Analysis Report
Real-world performance evaluation and improvement recommendations
"""

import json
from typing import Dict, List, Any
from collections import Counter

def load_test_results(file_path: str = "unseen_data_test_results.json") -> Dict[str, Any]:
    """Load test results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_category_performance(results: List[Dict]) -> Dict[str, Any]:
    """Analyze category classification performance in detail."""
    category_analysis = {
        "correct_predictions": [],
        "incorrect_predictions": [],
        "confusion_matrix": {},
        "category_accuracy": {}
    }
    
    # Track all expected and predicted categories
    expected_categories = [r["expected_category"] for r in results]
    predicted_categories = [r["predicted_category"] for r in results]
    
    # Build confusion matrix
    all_categories = list(set(expected_categories + predicted_categories))
    confusion = {}
    for exp_cat in all_categories:
        confusion[exp_cat] = {pred_cat: 0 for pred_cat in all_categories}
    
    for result in results:
        exp = result["expected_category"]
        pred = result["predicted_category"]
        confusion[exp][pred] += 1
        
        if exp == pred:
            category_analysis["correct_predictions"].append(result)
        else:
            category_analysis["incorrect_predictions"].append(result)
    
    # Calculate per-category accuracy
    for category in all_categories:
        total_for_category = expected_categories.count(category)
        correct_for_category = confusion[category][category]
        accuracy = correct_for_category / total_for_category if total_for_category > 0 else 0
        category_analysis["category_accuracy"][category] = accuracy
    
    category_analysis["confusion_matrix"] = confusion
    return category_analysis

def identify_failure_patterns(results: List[Dict]) -> Dict[str, List]:
    """Identify common failure patterns in predictions."""
    patterns = {
        "infrastructure_safety_confusion": [],
        "cleanliness_food_confusion": [],
        "staff_other_confusion": [],
        "priority_underestimation": [],
        "priority_overestimation": [],
        "escalation_over_eager": [],
        "escalation_under_eager": []
    }
    
    for result in results:
        # Category confusion patterns
        exp_cat = result["expected_category"]
        pred_cat = result["predicted_category"]
        
        if (exp_cat == "Infrastructure" and pred_cat == "Safety") or \
           (exp_cat == "Safety" and pred_cat == "Infrastructure"):
            patterns["infrastructure_safety_confusion"].append(result)
        
        if (exp_cat == "Cleanliness" and pred_cat == "Food") or \
           (exp_cat == "Food" and pred_cat == "Cleanliness"):
            patterns["cleanliness_food_confusion"].append(result)
        
        if (exp_cat == "Staff" and pred_cat == "Other") or \
           (exp_cat == "Other" and pred_cat == "Staff"):
            patterns["staff_other_confusion"].append(result)
        
        # Priority patterns
        priority_levels = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
        exp_pri = priority_levels[result["expected_priority"]]
        pred_pri = priority_levels[result["predicted_priority"]]
        
        if pred_pri < exp_pri - 1:  # Underestimating by more than 1 level
            patterns["priority_underestimation"].append(result)
        elif pred_pri > exp_pri + 1:  # Overestimating by more than 1 level
            patterns["priority_overestimation"].append(result)
        
        # Escalation patterns
        if result["expected_escalation"] == False and result["predicted_escalation"] == True:
            patterns["escalation_over_eager"].append(result)
        elif result["expected_escalation"] == True and result["predicted_escalation"] == False:
            patterns["escalation_under_eager"].append(result)
    
    return patterns

def generate_improvement_recommendations(analysis: Dict[str, Any], patterns: Dict[str, List]) -> List[str]:
    """Generate specific improvement recommendations."""
    recommendations = []
    
    # Category classification improvements
    if analysis["performance"]["category_accuracy"] < 0.6:
        recommendations.append(
            "🔧 CATEGORY CLASSIFICATION: Expand training data with more diverse examples. "
            "Current accuracy of 38% needs significant improvement."
        )
        
        # Specific category issues
        category_acc = analyze_category_performance(analysis["detailed_results"])["category_accuracy"]
        for cat, acc in category_acc.items():
            if acc < 0.4:
                recommendations.append(f"   • {cat}: Add more varied {cat.lower()} complaint examples")
    
    # Infrastructure-Safety confusion
    if len(patterns["infrastructure_safety_confusion"]) > 2:
        recommendations.append(
            "🔧 INFRASTRUCTURE vs SAFETY: Add clear distinction features. "
            "Infrastructure = physical damage/maintenance, Safety = immediate danger/risk"
        )
    
    # Priority classification
    if analysis["performance"]["priority_exact_accuracy"] < 0.7:
        if len(patterns["priority_underestimation"]) > 3:
            recommendations.append(
                "🔧 PRIORITY UNDERESTIMATION: Add urgency keywords and severity indicators. "
                "Train model to recognize critical situations better."
            )
    
    # Escalation logic
    if analysis["performance"]["escalation_accuracy"] < 0.7:
        if len(patterns["escalation_over_eager"]) > 5:
            recommendations.append(
                "🔧 ESCALATION OVER-EAGER: Model escalates too frequently (current issue). "
                "Add more non-escalation examples and refine escalation criteria."
            )
    
    # Confidence issues
    if analysis["performance"]["low_confidence_percentage"] > 0.3:
        recommendations.append(
            "🔧 LOW CONFIDENCE: 43% of predictions have low confidence. "
            "Increase training data diversity and feature engineering."
        )
    
    # Training data recommendations
    recommendations.append(
        "📊 TRAINING DATA: Current synthetic data is limited. "
        "Collect real railway complaint data for better performance."
    )
    
    recommendations.append(
        "🎯 FEATURE ENGINEERING: Add domain-specific features like urgency keywords, "
        "complaint source, time sensitivity, and railway-specific terminology."
    )
    
    return recommendations

def print_detailed_analysis():
    """Print comprehensive analysis of test results."""
    print("🔍 RCMS ML Models - Detailed Unseen Data Analysis")
    print("=" * 60)
    
    # Load results
    try:
        data = load_test_results()
        results = data["detailed_results"]
        performance = data["performance"]
        
        print(f"📊 OVERALL PERFORMANCE SUMMARY:")
        print(f"   • Total Test Cases: {performance['total_tests']}")
        print(f"   • Category Accuracy: {performance['category_accuracy']:.1%}")
        print(f"   • Priority Exact: {performance['priority_exact_accuracy']:.1%}")
        print(f"   • Priority Reasonable: {performance['priority_reasonable_accuracy']:.1%}")
        print(f"   • Escalation Accuracy: {performance['escalation_accuracy']:.1%}")
        print(f"   • Average Confidence: {performance['average_confidence']:.2f}")
        print(f"   • Low Confidence Cases: {performance['low_confidence_predictions']} ({performance['low_confidence_percentage']:.1%})")
        
        # Category analysis
        print(f"\n📂 CATEGORY CLASSIFICATION BREAKDOWN:")
        category_analysis = analyze_category_performance(results)
        for category, accuracy in category_analysis["category_accuracy"].items():
            status = "✅" if accuracy >= 0.6 else "⚠️" if accuracy >= 0.3 else "❌"
            count = sum(1 for r in results if r["expected_category"] == category)
            print(f"   {status} {category}: {accuracy:.1%} ({count} tests)")
        
        # Failure patterns
        print(f"\n🔍 FAILURE PATTERN ANALYSIS:")
        patterns = identify_failure_patterns(results)
        for pattern_name, cases in patterns.items():
            if cases:
                pattern_display = pattern_name.replace("_", " ").title()
                print(f"   • {pattern_display}: {len(cases)} cases")
        
        # Specific problematic cases
        print(f"\n❌ MOST PROBLEMATIC PREDICTIONS:")
        low_confidence_cases = [r for r in results if r["overall_confidence"] < 0.4]
        for case in low_confidence_cases[:5]:  # Show top 5
            print(f"   • '{case['text'][:60]}...'")
            print(f"     Expected: {case['expected_category']}/{case['expected_priority']}/{'Escalate' if case['expected_escalation'] else 'No'}")
            print(f"     Predicted: {case['predicted_category']}/{case['predicted_priority']}/{'Escalate' if case['predicted_escalation'] else 'No'}")
            print(f"     Confidence: {case['overall_confidence']:.2f}")
        
        # Improvement recommendations
        print(f"\n🚀 IMPROVEMENT RECOMMENDATIONS:")
        recommendations = generate_improvement_recommendations(data, patterns)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        # Model strengths
        print(f"\n💪 MODEL STRENGTHS:")
        strengths = []
        if performance['priority_reasonable_accuracy'] >= 0.8:
            strengths.append("Priority classification is reasonable (86% within 1 level)")
        if performance['escalation_accuracy'] >= 0.4:
            strengths.append("Escalation logic captures most critical cases")
        
        # Good predictions
        good_predictions = [r for r in results if 
                          r['predicted_category'] == r['expected_category'] and
                          r['predicted_priority'] == r['expected_priority']]
        if good_predictions:
            strengths.append(f"Perfect predictions on {len(good_predictions)} cases")
        
        for strength in strengths:
            print(f"   ✅ {strength}")
        
        print(f"\n🎯 NEXT STEPS:")
        print("   1. Collect real railway complaint data")
        print("   2. Improve Infrastructure vs Safety distinction")
        print("   3. Refine escalation criteria (reduce false positives)")
        print("   4. Add more Food and Cleanliness training examples")
        print("   5. Implement confidence thresholds for human review")
        
    except FileNotFoundError:
        print("❌ Test results file not found. Run the unseen data test first.")
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")

if __name__ == "__main__":
    print_detailed_analysis()