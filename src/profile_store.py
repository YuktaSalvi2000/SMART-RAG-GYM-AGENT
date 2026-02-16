import os
import json
from datetime import datetime

def create_profile(age, height, weight, fitness_level, gender, health_issues):
    profile = {
        "age": age,
        "height": height,
        "weight": weight,
        "fitness_level": fitness_level,
        "gender": gender,
        "health_issues": health_issues,
        "created_at": str(datetime.now()),
    }
    os.makedirs("storage/profiles", exist_ok=True)
    path = f"storage/profiles/{gender}_{int(age)}_{int(height)}_{int(weight)}.json"
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)
    return profile, path