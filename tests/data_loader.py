import csv
from typing import List
from pydantic import ValidationError
from app.models import DisasterDeclaration


def read_disaster_declarations_from_csv(file_path: str) -> List[DisasterDeclaration]:
    """
    Load disaster declarations from a CSV file into a list of DisasterDeclaration models.
    Skips invalid rows with validation errors.
    """
    declarations = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                # Convert disasterNumber to int explicitly if present
                if 'disasterNumber' in row and row['disasterNumber']:
                    row['disasterNumber'] = int(row['disasterNumber'])
                declaration = DisasterDeclaration(**row)
                declarations.append(declaration)
            except ValidationError as e:
                print(f"Skipping invalid row: {e}")
    print(f"Loaded {len(declarations)} disaster declarations from CSV.")
    return declarations
