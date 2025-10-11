"""
Enhanced Training Data for RCMS ML Models
Comprehensive real-world railway complaint examples with better category distinction
"""

from typing import Dict, List, Tuple
import random


class EnhancedTrainingDataGenerator:
    """Generate diverse, realistic training data with better category distinction."""
    
    def __init__(self):
        self.training_data = self._create_comprehensive_training_data()
    
    def _create_comprehensive_training_data(self) -> Dict[str, Dict[str, List[str]]]:
        """Create comprehensive training data with clear category distinctions."""
        
        return {
            "Infrastructure": {
                "Critical": [
                    "Overhead wire snapped and hanging dangerously over platform 2",
                    "Train derailed due to track fault near junction causing major blockage", 
                    "Bridge structure showing cracks and passengers feeling vibrations",
                    "Pantograph broke and damaged overhead catenary system completely",
                    "Signal system completely failed causing trains to stop for hours",
                    "Coach coupling broke during journey and separated from train",
                    "Platform roof collapsed during heavy rain injuring passengers",
                    "Track gauge widened causing wheel derailment risk",
                    "Overhead power supply failed affecting entire section"
                ],
                "High": [
                    "Coach door not opening at stations passengers unable to board",
                    "Air conditioning system completely failed in AC coach", 
                    "Train brake system making grinding noise and delayed stopping",
                    "Platform height difference too much passengers struggling to board",
                    "Coach lighting completely dark even during night journey",
                    "Window glass shattered due to structural vibration",
                    "Seat mounting broken and berth falling on passengers",
                    "Toilet door lock mechanism completely jammed and stuck",
                    "Coach fan making loud noise and wobbling dangerously"
                ],
                "Medium": [
                    "Platform bench broken and passengers have to stand",
                    "Station water tap not working and no drinking water",
                    "Coach window not closing properly during rain",
                    "Berth chain loose and making noise during travel",
                    "Platform display board not showing train timings",
                    "Station announcement system volume too low",
                    "Coach charging point not working for mobile phones",
                    "Platform shelter has broken tiles on roof",
                    "Station waiting room door handle broken"
                ],
                "Low": [
                    "Platform need more seating arrangements for passengers",
                    "Station building paint is old and needs renovation",
                    "Platform number board font size could be bigger",
                    "Station garden area needs better landscaping",
                    "Platform canopy design could be more modern"
                ]
            },
            
            "Cleanliness": {
                "Critical": [
                    "Sewage overflow in train toilet spreading to entire coach",
                    "Dead rat found in water tank passengers noticed bad smell",
                    "Coach floor covered with vomit and no cleaning for hours",
                    "Toilet waste flowing out and contaminating passenger area", 
                    "Platform flooded with drainage water mixed with garbage",
                    "Multiple cockroaches crawling on food serving counter",
                    "Toilet completely blocked with waste overflowing on floor",
                    "Coach water tank contaminated with visible dirt and insects"
                ],
                "High": [
                    "Train toilet extremely dirty with urine smell throughout coach",
                    "Platform littered with garbage and plastic bottles everywhere",
                    "Coach washbasin covered with thick dirt and grime layers",
                    "Station toilet unusable due to filthy condition and no water",
                    "Pantry area unhygienic with food items covered in dust",
                    "Coach floor sticky and dirty passengers shoes getting stuck",
                    "Platform dustbins overflowing and attracting flies",
                    "Train toilet seat broken and covered with stains"
                ],
                "Medium": [
                    "Platform needs sweeping and general cleaning maintenance",
                    "Coach seats have stains and require deep cleaning",
                    "Station toilet needs regular cleaning and soap refill",
                    "Train window glass dirty and affecting outside view",
                    "Platform has scattered paper and needs waste collection",
                    "Coach floor needs mopping and disinfection",
                    "Station waiting room dusty and needs cleaning"
                ],
                "Low": [
                    "Platform could use more frequent cleaning schedule",
                    "Coach interior cleaning could be more thorough",
                    "Station toilet paper needs regular replacement",
                    "Platform flower beds need weeding and maintenance"
                ]
            },
            
            "Safety": {
                "Critical": [
                    "Passenger threatening others with knife demanding money immediately",
                    "Fire broke out in coach S4 smoke filling entire compartment",
                    "Train collision imminent due to signal failure emergency braking",
                    "Passenger fell from moving train due to overcrowding fatality risk",
                    "Bomb threat reported by passenger evacuation needed urgently",
                    "Gas cylinder leak in pantry car explosion risk to passengers",
                    "Train brake failure while descending hill speed increasing dangerously",
                    "Overhead wire touched coach roof electric shock risk to passengers",
                    "Platform stampede during rush hour multiple passengers injured"
                ],
                "High": [
                    "Suspicious unattended bag found under seat in coach",
                    "Passenger harassment case woman asking for help immediately",
                    "Emergency chain pulled but train not stopping for miles",
                    "Passenger medical emergency heart attack needs doctor urgently",
                    "Train door opened during high speed journey safety risk",
                    "Platform edge too close to track passengers at risk",
                    "Coach tilting abnormally due to track defect danger",
                    "Passenger fight escalating and others feeling threatened"
                ],
                "Medium": [
                    "Platform lacks proper lighting during night time",
                    "Station has no security guard during late hours",
                    "Coach emergency equipment location not clearly marked",
                    "Platform announcement not audible during noisy conditions",
                    "Train first aid kit missing from coach emergency area",
                    "Station exit routes not clearly marked for emergencies"
                ],
                "Low": [
                    "Platform could use better security camera coverage",
                    "Station safety guidelines could be more visible",
                    "Train safety demonstration could be more comprehensive"
                ]
            },
            
            "Staff": {
                "Critical": [
                    "Ticket collector demanding bribe threatening to detrain passenger",
                    "Guard sexually harassing female passenger needs immediate action",
                    "Station master drunk on duty and creating dangerous situations",
                    "Conductor physically assaulting passenger over seat dispute",
                    "Staff member stealing passenger belongings from compartment"
                ],
                "High": [
                    "TTE very rude and using abusive language with passengers",
                    "Guard sleeping on duty and not responding to emergencies",
                    "Station announcer giving wrong information about train timings",
                    "Pantry staff preparing food in unhygienic conditions",
                    "Conductor refusing to help elderly passenger despite requests",
                    "Staff member discriminating against passenger based on appearance"
                ],
                "Medium": [
                    "Ticket checker not available in coach for ticket verification",
                    "Station staff not helpful in providing train information",
                    "Guard response time slow for passenger assistance requests",
                    "Pantry vendor overcharging for food items without receipt",
                    "Staff member not following proper uniform guidelines"
                ],
                "Low": [
                    "Station master was very helpful in resolving ticket issue",
                    "Train conductor polite and assisted with luggage placement",
                    "Thank you message for guard who helped during journey",
                    "Appreciate TTE for maintaining discipline in coach",
                    "Staff training could include better customer service skills"
                ]
            },
            
            "Food": {
                "Critical": [
                    "Multiple passengers food poisoning after eating pantry meal",
                    "Live insects found crawling in served food causing health risk",
                    "Food served was rotten and caused vomiting in passengers",
                    "Pantry using expired ingredients visible mold on bread items",
                    "Food contamination suspected several passengers fell sick simultaneously"
                ],
                "High": [
                    "Food quality very poor and not worth the high price",
                    "Meal served completely cold despite paying for hot food",
                    "Pantry staff rude and unhygienic while preparing food",
                    "Food items stale and taste suggests storage for days",
                    "Water bottle provided had suspicious particles floating inside"
                ],
                "Medium": [
                    "Breakfast not available despite advance booking and payment",
                    "Food overpriced compared to quality and quantity provided",
                    "Pantry service delayed and passengers waited for hours",
                    "Limited vegetarian options available for dietary requirements",
                    "Food packaging poor and items spilled during serving"
                ],
                "Low": [
                    "Food variety could be improved with more regional options",
                    "Meal timing could be better coordinated with journey schedule",
                    "Food presentation could be more appealing to passengers",
                    "Pantry could offer more healthy snack alternatives"
                ]
            },
            
            "Other": {
                "Critical": [
                    "Train booking system hacked and personal data compromised",
                    "Emergency communication system completely failed during crisis"
                ],
                "High": [
                    "Mobile app crashed during ticket booking money deducted twice",
                    "Website down for hours unable to check train status",
                    "Reservation system showing wrong berth allocation repeatedly",
                    "WiFi completely not working despite promises of connectivity"
                ],
                "Medium": [
                    "Train running late by 3 hours without proper updates",
                    "PNR status not updating causing confusion for passengers",
                    "Ticket cancellation refund delayed for more than month",
                    "GPS announcement system not working passengers confused",
                    "Platform display board showing incorrect train information"
                ],
                "Low": [
                    "Suggestion to improve online booking user interface design",
                    "Request for more payment options in mobile application",
                    "Train timing could be better optimized for commuters",
                    "Station could provide more passenger information services"
                ]
            }
        }
    
    def get_balanced_training_data(self) -> Tuple[List[str], List[str], List[str]]:
        """Get balanced training data for all categories and priorities."""
        texts = []
        categories = []
        priorities = []
        
        for category, priority_data in self.training_data.items():
            for priority, examples in priority_data.items():
                for example in examples:
                    texts.append(example)
                    categories.append(category)
                    priorities.append(priority)
        
        return texts, categories, priorities
    
    def get_escalation_training_data(self) -> Tuple[List[str], List[str]]:
        """Generate escalation training data based on priority and content."""
        texts = []
        escalation_labels = []
        
        # Define escalation rules
        escalation_keywords = {
            'bribe', 'harassment', 'assault', 'theft', 'discrimination',
            'emergency', 'danger', 'fire', 'accident', 'injury', 'threat',
            'critical', 'urgent', 'help', 'immediately'
        }
        
        for category, priority_data in self.training_data.items():
            for priority, examples in priority_data.items():
                for example in examples:
                    texts.append(example)
                    
                    # Determine escalation based on priority and keywords
                    should_escalate = False
                    
                    # Always escalate Critical issues
                    if priority == "Critical":
                        should_escalate = True
                    
                    # Escalate High priority if safety/staff issues
                    elif priority == "High" and category in ["Safety", "Staff"]:
                        should_escalate = True
                    
                    # Escalate if contains escalation keywords
                    elif any(keyword in example.lower() for keyword in escalation_keywords):
                        should_escalate = True
                    
                    # Don't escalate positive feedback or suggestions
                    elif any(word in example.lower() for word in ['thank', 'appreciate', 'suggestion', 'could']):
                        should_escalate = False
                    
                    escalation_labels.append("Escalate" if should_escalate else "No_Escalation")
        
        return texts, escalation_labels
    
    def get_category_specific_data(self, category: str, min_examples: int = 20) -> List[str]:
        """Get category-specific training data with minimum examples."""
        if category not in self.training_data:
            return []
        
        examples = []
        for priority_data in self.training_data[category].values():
            examples.extend(priority_data)
        
        # If we need more examples, create variations
        while len(examples) < min_examples:
            # Add slight variations of existing examples
            base_example = random.choice(examples[:len(examples)//2])  # Choose from original examples only
            # Simple variation by adding context
            variations = [
                f"Repeated issue: {base_example}",
                f"Urgent: {base_example}",
                f"Please help: {base_example}",
                f"Again today: {base_example}"
            ]
            examples.extend(variations)
        
        return examples[:min_examples]


# Test the enhanced training data
if __name__ == "__main__":
    generator = EnhancedTrainingDataGenerator()
    
    print("📊 Enhanced Training Data Analysis")
    print("=" * 50)
    
    # Get balanced data
    texts, categories, priorities = generator.get_balanced_training_data()
    
    print(f"Total training examples: {len(texts)}")
    print(f"\nCategory distribution:")
    from collections import Counter
    cat_counts = Counter(categories)
    for cat, count in cat_counts.items():
        print(f"  {cat}: {count} examples")
    
    print(f"\nPriority distribution:")
    pri_counts = Counter(priorities)
    for pri, count in pri_counts.items():
        print(f"  {pri}: {count} examples")
    
    # Get escalation data
    esc_texts, esc_labels = generator.get_escalation_training_data()
    esc_counts = Counter(esc_labels)
    print(f"\nEscalation distribution:")
    for label, count in esc_counts.items():
        print(f"  {label}: {count} examples")
    
    # Show sample data for each category
    print(f"\n🧪 Sample Training Examples:")
    print("-" * 50)
    
    for category in generator.training_data.keys():
        examples = generator.get_category_specific_data(category, 2)
        print(f"\n{category}:")
        for i, example in enumerate(examples[:2], 1):
            print(f"  {i}. {example}")
    
    print(f"\n✅ Enhanced training data ready for ML model improvement!")