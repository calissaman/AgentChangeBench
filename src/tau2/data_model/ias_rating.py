"""
Data models for IAS (Instruction Adaptivity Score) ratings.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class IASCriteria(str, Enum):
    """IAS rating criteria based on the AgentChangeBench specification."""
    TONE_MATCH = "tone_match"
    CLARITY = "clarity"
    PACING = "pacing"
    ADAPTIVITY = "adaptivity"


class IASCriteriaScore(BaseModel):
    """Individual score for an IAS criteria."""
    criteria: IASCriteria = Field(description="The criteria being rated")
    score: int = Field(description="Score from 1-5", ge=1, le=5)
    notes: Optional[str] = Field(description="Optional notes for this criteria", default=None)


class IASRating(BaseModel):
    """Complete IAS rating for a simulation."""
    simulation_id: str = Field(description="ID of the simulation being rated")
    rater_name: str = Field(description="Name/ID of the person doing the rating")
    timestamp: str = Field(description="When the rating was completed", default_factory=lambda: datetime.now().isoformat())
    
    # Individual criteria scores
    tone_match: IASCriteriaScore = Field(description="How well agent fits persona's emotional and social expectations")
    clarity: IASCriteriaScore = Field(description="How understandable explanations are to persona")
    pacing: IASCriteriaScore = Field(description="Whether response length/timing is suitable")
    adaptivity: IASCriteriaScore = Field(description="How well agent adjusts when persona needs change")
    
    # Overall rating and notes
    overall_score: float = Field(description="Average of all criteria scores")
    overall_notes: Optional[str] = Field(description="General notes about the rating", default=None)
    
    def calculate_overall_score(self) -> float:
        """Calculate the overall IAS score as average of criteria scores."""
        scores = [
            self.tone_match.score,
            self.clarity.score,
            self.pacing.score,
            self.adaptivity.score
        ]
        return sum(scores) / len(scores)
    
    def __init__(self, **data):
        # Calculate overall score before calling super().__init__
        if 'overall_score' not in data and all(key in data for key in ['tone_match', 'clarity', 'pacing', 'adaptivity']):
            scores = [
                data['tone_match'].score,
                data['clarity'].score,
                data['pacing'].score,
                data['adaptivity'].score
            ]
            data['overall_score'] = sum(scores) / len(scores)
        
        super().__init__(**data)


class IASRatingCollection(BaseModel):
    """Collection of IAS ratings for multiple simulations."""
    ratings: Dict[str, List[IASRating]] = Field(
        description="Mapping of simulation_id to list of ratings",
        default_factory=dict
    )
    
    def add_rating(self, rating: IASRating):
        """Add a rating to the collection."""
        if rating.simulation_id not in self.ratings:
            self.ratings[rating.simulation_id] = []
        self.ratings[rating.simulation_id].append(rating)
    
    def get_ratings_for_simulation(self, simulation_id: str) -> List[IASRating]:
        """Get all ratings for a specific simulation."""
        return self.ratings.get(simulation_id, [])
    
    def get_average_score_for_simulation(self, simulation_id: str) -> Optional[float]:
        """Get the average IAS score for a simulation across all raters."""
        ratings = self.get_ratings_for_simulation(simulation_id)
        if not ratings:
            return None
        return sum(r.overall_score for r in ratings) / len(ratings)
    
    def get_inter_rater_reliability(self, simulation_id: str) -> Optional[float]:
        """Calculate inter-rater reliability (simple standard deviation) for a simulation."""
        ratings = self.get_ratings_for_simulation(simulation_id)
        if len(ratings) < 2:
            return None
        
        scores = [r.overall_score for r in ratings]
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        return variance ** 0.5  # Standard deviation


# Constants for the rating interface
IAS_CRITERIA_DESCRIPTIONS = {
    IASCriteria.TONE_MATCH: {
        "name": "Tone Match",
        "description": "How well the agent fits the persona's emotional and social expectations",
        "examples": {
            1: "Poor match - completely inappropriate tone for persona",
            3: "Adequate match - generally appropriate but some misalignment", 
            5: "Excellent match - perfectly fits persona's emotional and social expectations"
        }
    },
    IASCriteria.CLARITY: {
        "name": "Clarity", 
        "description": "How understandable the agent's explanations are to the persona",
        "examples": {
            1: "Poor clarity - confusing, jargon-heavy, hard to understand",
            3: "Adequate clarity - mostly understandable with some unclear parts",
            5: "Excellent clarity - crystal clear explanations appropriate for persona"
        }
    },
    IASCriteria.PACING: {
        "name": "Pacing",
        "description": "Whether response length and timing are suitable for the persona", 
        "examples": {
            1: "Poor pacing - too verbose/brief, inappropriate timing",
            3: "Adequate pacing - generally appropriate length and timing",
            5: "Excellent pacing - perfect response length and timing for persona"
        }
    },
    IASCriteria.ADAPTIVITY: {
        "name": "Adaptivity",
        "description": "How well the agent adjusts its approach when persona needs change mid-task",
        "examples": {
            1: "Poor adaptivity - rigid, doesn't adjust to changing needs",
            3: "Adequate adaptivity - some adjustment but could be better",
            5: "Excellent adaptivity - seamlessly adjusts to persona's changing needs"
        }
    }
} 