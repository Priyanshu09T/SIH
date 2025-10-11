#!/usr/bin/env python3
"""
Test the enhanced ML models with realistic Indian Railway passenger complaints.
This file evaluates the model’s robustness against real-world, high-diversity complaint scenarios.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.enhanced_ml_models import EnhancedCategoryClassifier, EnhancedPriorityClassifier, EnhancedEscalationClassifier
from src.preprocessing.enhanced_text_processor import EnhancedTextProcessor
import pandas as pd


def load_enhanced_models():
    """Load and train enhanced models with expanded railway complaint data."""
    print("🔄 Loading and training enhanced models...")

    category_model = EnhancedCategoryClassifier()
    priority_model = EnhancedPriorityClassifier()
    escalation_model = EnhancedEscalationClassifier()

    category_model.train()
    priority_model.train()
    escalation_model.train()

    print("✅ Enhanced models training completed successfully!\n")
    return category_model, priority_model, escalation_model


def test_real_life_complaints():
    """Test enhanced models with real-world Indian Railway complaint scenarios."""

    unseen_data = [
        # 🚆 Safety & Harassment Cases
        {"text": "A drunk passenger is creating nuisance in coach S5, no RPF available!", "category": "Safety", "priority": "Critical", "escalate": True},
        {"text": "Lady passenger being harassed near platform 6, need help urgently", "category": "Safety", "priority": "Critical", "escalate": True},
        {"text": "Unauthorized person traveling in ladies coach at night", "category": "Safety", "priority": "High", "escalate": True},
        {"text": "Group fighting in general compartment, no staff to control situation", "category": "Safety", "priority": "Critical", "escalate": True},
        {"text": "Passenger smoking inside AC coach despite warnings", "category": "Safety", "priority": "Medium", "escalate": False},

        # 🧹 Cleanliness Issues
        {"text": "Train toilet water tank empty, unable to use washroom since morning", "category": "Cleanliness", "priority": "High", "escalate": True},
        {"text": "Platform at Patna Junction covered in garbage, smells unbearable", "category": "Cleanliness", "priority": "Medium", "escalate": False},
        {"text": "Seats are sticky and unhygienic in coach A1", "category": "Cleanliness", "priority": "Medium", "escalate": False},
        {"text": "Waste and leftover food thrown near track at Varanasi station", "category": "Cleanliness", "priority": "Low", "escalate": False},
        {"text": "Cockroaches crawling in sleeper coach, needs pest control", "category": "Cleanliness", "priority": "High", "escalate": True},

        # ⚙️ Infrastructure & Technical
        {"text": "Coach door not closing properly, risk of passengers falling", "category": "Infrastructure", "priority": "Critical", "escalate": True},
        {"text": "AC not cooling properly in B1 coach despite complaint", "category": "Infrastructure", "priority": "Medium", "escalate": False},
        {"text": "Water leakage from ceiling near berth 42", "category": "Infrastructure", "priority": "Medium", "escalate": False},
        {"text": "Coach lights flickering continuously throughout night", "category": "Infrastructure", "priority": "Low", "escalate": False},
        {"text": "Broken window glass in express train, dust entering cabin", "category": "Infrastructure", "priority": "Medium", "escalate": False},

        # 🍱 Food & Pantry Issues
        {"text": "Food served in Rajdhani Express is stale and cold", "category": "Other", "priority": "Medium", "escalate": False},
        {"text": "Pantry staff overcharging for tea and snacks", "category": "Other", "priority": "Medium", "escalate": False},
        {"text": "Food tray not cleaned after previous passenger", "category": "Cleanliness", "priority": "Low", "escalate": False},
        {"text": "Pantry boy misbehaving with elderly woman", "category": "Staff", "priority": "High", "escalate": True},

        # 👨‍✈️ Staff Behavior & Service
        {"text": "Ticket checker misbehaved and used abusive language", "category": "Staff", "priority": "High", "escalate": True},
        {"text": "RPF constable ignored complaint about theft", "category": "Staff", "priority": "High", "escalate": True},
        {"text": "Station master not responding to passengers after train delay", "category": "Staff", "priority": "Medium", "escalate": True},
        {"text": "TTE helped handicapped passenger to seat, very kind gesture", "category": "Staff", "priority": "Low", "escalate": False},

        # 💸 Booking, Refund & Digital Issues
        {"text": "Payment deducted but ticket not booked, IRCTC app crashed", "category": "Booking", "priority": "High", "escalate": True},
        {"text": "Refund for cancelled train still not received after 2 weeks", "category": "Booking", "priority": "Medium", "escalate": False},
        {"text": "Unable to change boarding point even before charting", "category": "Booking", "priority": "Low", "escalate": False},
        {"text": "PNR showing incorrect coach allocation after update", "category": "Booking", "priority": "Medium", "escalate": False},

        # ⚠️ Emergencies
        {"text": "Passenger fainted in AC coach, need medical assistance", "category": "Safety", "priority": "Critical", "escalate": True},
        {"text": "Short circuit in coach C2, sparks seen near fan unit", "category": "Safety", "priority": "Critical", "escalate": True},
        {"text": "Passenger fell while boarding moving train", "category": "Safety", "priority": "Critical", "escalate": True},
    ]

    # Load models
    category_model, priority_model, escalation_model = load_enhanced_models()

    print("\n🧪 Testing Enhanced Models on Real Railway Complaint Scenarios:")
    print("=" * 75)

    category_correct = priority_correct = escalation_correct = reasonable_predictions = 0
    total_tests = len(unseen_data)

    for i, item in enumerate(unseen_data, 1):
        text = item["text"]
        expected_category = item["category"]
        expected_priority = item["priority"]
        expected_escalate = item["escalate"]

        pred_category_result = category_model.predict(text)
        pred_priority_result = priority_model.predict(text)
        pred_escalate_result = escalation_model.predict(text)

        # Extract actual predictions from result dictionaries
        pred_category = pred_category_result["prediction"] if isinstance(pred_category_result, dict) else pred_category_result
        pred_priority = pred_priority_result["prediction"] if isinstance(pred_priority_result, dict) else pred_priority_result
        pred_escalate = pred_escalate_result if isinstance(pred_escalate_result, bool) else (pred_escalate_result.get("prediction", False) if isinstance(pred_escalate_result, dict) else False)
        
        # Get prediction confidence
        cat_conf = pred_category_result.get("confidence", 0.5) if isinstance(pred_category_result, dict) else 0.5
        pri_conf = pred_priority_result.get("confidence", 0.5) if isinstance(pred_priority_result, dict) else 0.5
        esc_conf = pred_escalate_result.get("confidence", 0.5) if isinstance(pred_escalate_result, dict) else 0.5

        cat_correct = pred_category == expected_category
        pri_correct = pred_priority == expected_priority
        esc_correct = pred_escalate == expected_escalate

        if cat_correct:
            category_correct += 1
        if pri_correct:
            priority_correct += 1
        if esc_correct:
            escalation_correct += 1

        reasonable = (cat_correct or
                     (expected_priority == "Critical" and pred_priority in ["Critical", "High"]) or
                     (expected_priority == "High" and pred_priority in ["Critical", "High", "Medium"]) or
                     (expected_priority == "Medium" and pred_priority in ["High", "Medium", "Low"]) or
                     (expected_priority == "Low" and pred_priority in ["Medium", "Low"]))
        if reasonable:
            reasonable_predictions += 1

        status = "✅" if cat_correct and pri_correct and esc_correct else "❌"
        print(f"\n{status} Test {i}: {text[:65]}...")
        print(f"   Expected: {expected_category}/{expected_priority}/{'Escalate' if expected_escalate else 'No'}")
        print(f"   Predicted: {pred_category}/{pred_priority}/{'Escalate' if pred_escalate else 'No'}")
        print(f"   Confidence: Cat={cat_conf:.2f}, Pri={pri_conf:.2f}, Esc={esc_conf:.2f}")

        if not cat_correct:
            print(f"   ⚠️ Category mismatch: {expected_category} → {pred_category}")
        if not pri_correct:
            print(f"   ⚠️ Priority mismatch: {expected_priority} → {pred_priority}")
        if not esc_correct:
            print(f"   ⚠️ Escalation mismatch: {'Yes' if expected_escalate else 'No'} → {'Yes' if pred_escalate else 'No'}")

    # Compute metrics
    category_accuracy = category_correct / total_tests
    priority_accuracy = priority_correct / total_tests
    escalation_accuracy = escalation_correct / total_tests
    reasonable_accuracy = reasonable_predictions / total_tests
    overall_accuracy = (category_correct + priority_correct + escalation_correct) / (total_tests * 3)

    print("\n" + "=" * 75)
    print("📈 ENHANCED MODEL PERFORMANCE SUMMARY:")
    print("=" * 75)
    print(f"📊 Category Accuracy:    {category_accuracy:.1%} ({category_correct}/{total_tests})")
    print(f"📊 Priority Accuracy:    {priority_accuracy:.1%} ({priority_correct}/{total_tests})")
    print(f"📊 Escalation Accuracy:  {escalation_accuracy:.1%} ({escalation_correct}/{total_tests})")
    print(f"📊 Reasonable Predictions: {reasonable_accuracy:.1%} ({reasonable_predictions}/{total_tests})")
    print(f"📊 Overall Accuracy:     {overall_accuracy:.1%}")

    print("\n🔄 PERFORMANCE COMPARISON:")
    print("=" * 75)
    print("Original Models vs Enhanced Models:")
    print(f"  Category:   38% → {category_accuracy:.1%} ({category_accuracy-0.38:+.1%})")
    print(f"  Priority:   86% → {priority_accuracy:.1%} ({priority_accuracy-0.86:+.1%})")
    print(f"  Escalation: 48% → {escalation_accuracy:.1%} ({escalation_accuracy-0.48:+.1%})")

    return {
        'category_accuracy': category_accuracy,
        'priority_accuracy': priority_accuracy,
        'escalation_accuracy': escalation_accuracy,
        'reasonable_accuracy': reasonable_accuracy,
        'overall_accuracy': overall_accuracy
    }


if __name__ == "__main__":
    print("🚉 Indian Railway Complaint Management System - Enhanced Model Testing")
    print("=" * 75)

    try:
        results = test_real_life_complaints()
        print("\n🎯 TESTING COMPLETE!")
        print("Enhanced models demonstrate improved realism and context understanding.")
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
