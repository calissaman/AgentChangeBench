import json
from pathlib import Path
from typing import List, Optional

from tau2.data_model.ias_rating import IASRating, IASRatingCollection


class IASManager:
    """Manager for IAS ratings storage and retrieval."""
    
    def __init__(self, ratings_file: Optional[Path] = None):
        """Initialize IAS manager with optional custom ratings file path."""
        if ratings_file is None:
            # Default to storing ratings alongside simulation files
            self.ratings_file = Path("data/simulations/ias_ratings.json")
        else:
            self.ratings_file = ratings_file
        
        # Ensure directory exists
        self.ratings_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing ratings
        self.ratings = self._load_ratings()
    
    def _load_ratings(self) -> IASRatingCollection:
        """Load ratings from file or create new collection."""
        if self.ratings_file.exists():
            try:
                with open(self.ratings_file, 'r') as f:
                    data = json.load(f)
                return IASRatingCollection.model_validate(data)
            except Exception as e:
                print(f"Warning: Could not load ratings file {self.ratings_file}: {e}")
                return IASRatingCollection()
        else:
            return IASRatingCollection()
    
    def save_ratings(self):
        """Save ratings to file."""
        try:
            with open(self.ratings_file, 'w') as f:
                f.write(self.ratings.model_dump_json(indent=2))
        except Exception as e:
            print(f"Error saving ratings to {self.ratings_file}: {e}")
            raise
    
    def add_rating(self, rating: IASRating):
        """Add a new rating and save to file."""
        self.ratings.add_rating(rating)
        self.save_ratings()
    
    def get_ratings_for_simulation(self, simulation_id: str) -> List[IASRating]:
        """Get all ratings for a specific simulation."""
        return self.ratings.get_ratings_for_simulation(simulation_id)
    
    def has_rating_by_rater(self, simulation_id: str, rater_name: str) -> bool:
        """Check if a specific rater has already rated a simulation."""
        ratings = self.get_ratings_for_simulation(simulation_id)
        return any(r.rater_name == rater_name for r in ratings)
    
    def get_rating_summary(self, simulation_id: str) -> dict:
        """Get a summary of ratings for a simulation."""
        ratings = self.get_ratings_for_simulation(simulation_id)
        
        if not ratings:
            return {
                "num_ratings": 0,
                "average_score": None,
                "raters": [],
                "reliability": None
            }
        
        return {
            "num_ratings": len(ratings),
            "average_score": self.ratings.get_average_score_for_simulation(simulation_id),
            "raters": [r.rater_name for r in ratings],
            "reliability": self.ratings.get_inter_rater_reliability(simulation_id)
        }
    
    def get_all_rated_simulations(self) -> List[str]:
        """Get list of all simulation IDs that have been rated."""
        return list(self.ratings.ratings.keys())
    
    def export_ratings_csv(self, output_file: Path):
        """Export all ratings to CSV format for analysis."""
        import csv
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'simulation_id', 'rater_name', 'timestamp', 'overall_score',
                'tone_match_score', 'tone_match_notes',
                'clarity_score', 'clarity_notes', 
                'pacing_score', 'pacing_notes',
                'adaptivity_score', 'adaptivity_notes',
                'overall_notes'
            ])
            
            # Write data
            for sim_id, ratings_list in self.ratings.ratings.items():
                for rating in ratings_list:
                    writer.writerow([
                        sim_id,
                        rating.rater_name,
                        rating.timestamp,
                        rating.overall_score,
                        rating.tone_match.score,
                        rating.tone_match.notes or '',
                        rating.clarity.score,
                        rating.clarity.notes or '',
                        rating.pacing.score,
                        rating.pacing.notes or '',
                        rating.adaptivity.score,
                        rating.adaptivity.notes or '',
                        rating.overall_notes or ''
                    ])


def get_default_ias_manager() -> IASManager:
    """Get the default IAS manager instance."""
    return IASManager() 