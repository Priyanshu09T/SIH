"""
Test RCMS ML Models with Unseen Real-World Data
Comprehensive validation with diverse railway complaint scenarios
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import json

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.models.complete_ml_models import ComplaintMLPipeline


class UnseenDataTester:
    """Test ML models with unseen, realistic railway complaint data."""
    
    def __init__(self):
        self.pipeline = ComplaintMLPipeline()
        self.test_results = []
        
    def get_unseen_test_data(self) -> List[Dict[str, Any]]:
        """
        Real-world railway complaint scenarios not seen during training.
        Each complaint has expected category, priority, and escalation labels.
        """
        unseen_complaints = [
            # Infrastructure complaints - varied scenarios
            {
                "text": "The overhead wire is sparking dangerously near platform 3. Passengers are scared to board.",
                "expected_category": "Infrastructure",
                "expected_priority": "Critical",
                "expected_escalation": True,
                "scenario": "Electrical safety hazard"
            },
            {
                "text": "Water dripping from coach ceiling throughout the journey. Seats getting wet.",
                "expected_category": "Infrastructure", 
                "expected_priority": "Medium",
                "expected_escalation": False,
                "scenario": "Coach maintenance issue"
            },
            {
                "text": "The automatic doors in metro coach are malfunctioning and closing on passengers.",
                "expected_category": "Infrastructure",
                "expected_priority": "High",
                "expected_escalation": True,
                "scenario": "Door safety issue"
            },
            
            # Cleanliness complaints - real scenarios
            {
                "text": "Cockroaches in the pantry car! This is absolutely disgusting and unhygienic.",
                "expected_category": "Cleanliness",
                "expected_priority": "High", 
                "expected_escalation": True,
                "scenario": "Severe hygiene violation"
            },
            {
                "text": "The washbasin taps are not working and there's no soap dispenser in coach B4.",
                "expected_category": "Cleanliness",
                "expected_priority": "Medium",
                "expected_escalation": False,
                "scenario": "Basic amenity missing"
            },
            {
                "text": "Vomit on the floor near seat 23 hasn't been cleaned for 2 hours despite informing staff.",
                "expected_category": "Cleanliness",
                "expected_priority": "High",
                "expected_escalation": False,
                "scenario": "Cleaning delay"
            },
            
            # Safety complaints - critical scenarios
            {
                "text": "A passenger is threatening other travelers with a knife. Need immediate help!",
                "expected_category": "Safety",
                "expected_priority": "Critical",
                "expected_escalation": True,
                "scenario": "Immediate security threat"
            },
            {
                "text": "Smoke detected in coach S7. Fire alarm not working. Passengers evacuating.",
                "expected_category": "Safety",
                "expected_priority": "Critical", 
                "expected_escalation": True,
                "scenario": "Fire emergency"
            },
            {
                "text": "The emergency chain was pulled but train didn't stop for 10 minutes.",
                "expected_category": "Safety",
                "expected_priority": "Critical",
                "expected_escalation": True,
                "scenario": "Emergency system failure"
            },
            
            # Staff behavior complaints
            {
                "text": "The ticket collector asked for bribe to allow me to travel without reservation.",
                "expected_category": "Staff",
                "expected_priority": "High",
                "expected_escalation": True,
                "scenario": "Corruption incident"
            },
            {
                "text": "Station master was very helpful in resolving my ticket booking issue quickly.",
                "expected_category": "Staff",
                "expected_priority": "Low",
                "expected_escalation": False,
                "scenario": "Positive feedback"
            },
            {
                "text": "Guard refused to help elderly passenger climb into coach despite repeated requests.",
                "expected_category": "Staff",
                "expected_priority": "Medium",
                "expected_escalation": False,
                "scenario": "Unhelpful behavior"
            },
            
            # Food service complaints
            {
                "text": "Found insects in the food served in Rajdhani Express. Multiple passengers fell sick.",
                "expected_category": "Food",
                "expected_priority": "Critical",
                "expected_escalation": True,
                "scenario": "Food contamination"
            },
            {
                "text": "The tea was cold and the vendor charged extra money saying it's special tea.",
                "expected_category": "Food",
                "expected_priority": "Low",
                "expected_escalation": False,
                "scenario": "Overcharging complaint"
            },
            {
                "text": "Dinner was not served despite advance booking and payment. No refund given.",
                "expected_category": "Food",
                "expected_priority": "Medium",
                "expected_escalation": False,
                "scenario": "Service failure"
            },
            
            # Technology/Other complaints
            {
                "text": "The mobile app crashed while booking tickets and money was deducted twice.",
                "expected_category": "Other",
                "expected_priority": "Medium",
                "expected_escalation": False,
                "scenario": "App technical issue"
            },
            {
                "text": "GPS announcement system is not working. Passengers don't know station names.",
                "expected_category": "Other",
                "expected_priority": "Medium",
                "expected_escalation": False,
                "scenario": "Information system failure"
            },
            {
                "text": "Online refund has been pending for 3 months despite multiple follow-ups.",
                "expected_category": "Other",
                "expected_priority": "Medium",
                "expected_escalation": False,
                "scenario": "Refund delay"
            },
            
            # Edge cases and complex scenarios
            {
                "text": "Passenger fell from moving train due to overcrowding. Platform staff didn't help.",
                "expected_category": "Safety",  # Could be staff, but safety is primary
                "expected_priority": "Critical",
                "expected_escalation": True,
                "scenario": "Complex safety-staff issue"
            },
            {
                "text": "The coach has no lights, broken windows, dirty seats, and smells bad.",
                "expected_category": "Infrastructure",  # Multiple issues, but structural primary
                "expected_priority": "High",
                "expected_escalation": False,
                "scenario": "Multiple category overlap"
            },
            {
                "text": "Suggestion: Install more dustbins on platforms to keep them clean.",
                "expected_category": "Other",  # Suggestion, not complaint
                "expected_priority": "Low",
                "expected_escalation": False,
                "scenario": "Constructive suggestion"
            }
        ]
        
        return unseen_complaints
    
    def analyze_model_performance(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze model performance on unseen data."""
        total_tests = len(results)
        
        # Category accuracy
        category_correct = sum(1 for r in results if r['predicted_category'] == r['expected_category'])
        category_accuracy = category_correct / total_tests
        
        # Priority accuracy (exact match)
        priority_correct = sum(1 for r in results if r['predicted_priority'] == r['expected_priority'])
        priority_accuracy = priority_correct / total_tests
        
        # Priority accuracy (reasonable - within 1 level)
        priority_levels = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
        priority_reasonable = sum(1 for r in results 
                                if abs(priority_levels[r['predicted_priority']] - 
                                      priority_levels[r['expected_priority']]) <= 1)
        priority_reasonable_accuracy = priority_reasonable / total_tests
        
        # Escalation accuracy
        escalation_correct = sum(1 for r in results if r['predicted_escalation'] == r['expected_escalation'])
        escalation_accuracy = escalation_correct / total_tests
        
        # Confidence analysis
        avg_confidence = np.mean([r['overall_confidence'] for r in results])
        low_confidence_threshold = 0.4
        low_confidence_count = sum(1 for r in results if r['overall_confidence'] < low_confidence_threshold)
        
        return {
            "total_tests": total_tests,
            "category_accuracy": category_accuracy,
            "priority_exact_accuracy": priority_accuracy,
            "priority_reasonable_accuracy": priority_reasonable_accuracy,
            "escalation_accuracy": escalation_accuracy,
            "average_confidence": avg_confidence,
            "low_confidence_predictions": low_confidence_count,
            "low_confidence_percentage": low_confidence_count / total_tests
        }
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test on unseen data."""
        print("🧪 Testing ML Models with Unseen Real-World Data")
        print("=" * 60)
        
        # Train the pipeline
        print("🔄 Training ML pipeline...")
        self.pipeline.train()
        print("✅ Training completed\n")
        
        # Get test data
        test_data = self.get_unseen_test_data()
        print(f"📊 Testing with {len(test_data)} unseen complaints\n")
        
        # Test each complaint
        results = []
        for i, test_case in enumerate(test_data, 1):
            try:
                # Get ML predictions
                analysis = self.pipeline.analyze_complaint(test_case["text"])
                
                # Store results
                result = {
                    "test_id": i,
                    "text": test_case["text"],
                    "scenario": test_case["scenario"],
                    "expected_category": test_case["expected_category"],
                    "expected_priority": test_case["expected_priority"],
                    "expected_escalation": test_case["expected_escalation"],
                    "predicted_category": analysis["category"]["prediction"],
                    "predicted_priority": analysis["priority"]["prediction"],
                    "predicted_escalation": analysis["escalation"]["should_escalate"],
                    "category_confidence": analysis["category"]["confidence"],
                    "priority_confidence": analysis["priority"]["confidence"],
                    "escalation_confidence": analysis["escalation"]["confidence"],
                    "overall_confidence": analysis["overall_confidence"]
                }
                results.append(result)
                
                # Print individual result
                print(f"Test {i}: {test_case['scenario']}")
                print(f"📝 Text: '{test_case['text'][:80]}{'...' if len(test_case['text']) > 80 else ''}'")
                
                # Category result
                cat_status = "✅" if result['predicted_category'] == result['expected_category'] else "❌"
                print(f"📂 Category: {cat_status} {result['predicted_category']} (expected: {result['expected_category']}) [{result['category_confidence']:.2f}]")
                
                # Priority result  
                pri_status = "✅" if result['predicted_priority'] == result['expected_priority'] else "❌"
                print(f"⚡ Priority: {pri_status} {result['predicted_priority']} (expected: {result['expected_priority']}) [{result['priority_confidence']:.2f}]")
                
                # Escalation result
                esc_status = "✅" if result['predicted_escalation'] == result['expected_escalation'] else "❌"
                print(f"🚨 Escalate: {esc_status} {'Yes' if result['predicted_escalation'] else 'No'} (expected: {'Yes' if result['expected_escalation'] else 'No'}) [{result['escalation_confidence']:.2f}]")
                
                print(f"📊 Overall Confidence: {result['overall_confidence']:.2f}")
                print("-" * 40)
                
            except Exception as e:
                print(f"❌ Error testing complaint {i}: {e}")
                print("-" * 40)
        
        # Analyze overall performance
        performance = self.analyze_model_performance(results)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📈 UNSEEN DATA TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {performance['total_tests']}")
        print(f"Category Accuracy: {performance['category_accuracy']:.1%}")
        print(f"Priority Exact Accuracy: {performance['priority_exact_accuracy']:.1%}")
        print(f"Priority Reasonable Accuracy: {performance['priority_reasonable_accuracy']:.1%}")
        print(f"Escalation Accuracy: {performance['escalation_accuracy']:.1%}")
        print(f"Average Confidence: {performance['average_confidence']:.2f}")
        print(f"Low Confidence Predictions: {performance['low_confidence_predictions']} ({performance['low_confidence_percentage']:.1%})")
        
        # Performance evaluation
        print("\n🎯 PERFORMANCE EVALUATION:")
        if performance['category_accuracy'] >= 0.7:
            print("✅ Category Classification: GOOD")
        else:
            print("⚠️ Category Classification: NEEDS IMPROVEMENT")
            
        if performance['priority_reasonable_accuracy'] >= 0.7:
            print("✅ Priority Classification: GOOD") 
        else:
            print("⚠️ Priority Classification: NEEDS IMPROVEMENT")
            
        if performance['escalation_accuracy'] >= 0.8:
            print("✅ Escalation Logic: GOOD")
        else:
            print("⚠️ Escalation Logic: NEEDS IMPROVEMENT")
        
        return {
            "performance": performance,
            "detailed_results": results
        }


def main():
    """Run the unseen data testing."""
    tester = UnseenDataTester()
    results = tester.run_comprehensive_test()
    
    # Save results for analysis
    with open("unseen_data_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: unseen_data_test_results.json")
    print("🎉 Unseen data testing completed!")


if __name__ == "__main__":
    main()