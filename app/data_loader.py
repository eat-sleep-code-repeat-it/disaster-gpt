import csv
from typing import List
from pydantic import ValidationError
from app.models import DisasterDeclaration
 
def read_disaster_declarations_from_csv(file_path: str) -> List[DisasterDeclaration]:
    """
    load declarations as structured date model from csv
    """
    declarations = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                # Convert disasterNumber to int explicitly
                if 'disasterNumber' in row and row['disasterNumber']:
                    row['disasterNumber'] = int(row['disasterNumber'])
                decl = DisasterDeclaration(**row)
                declarations.append(decl)
            except ValidationError as e:
                print(f"Skipping invalid row: {e}")
    print(f"Loaded {len(declarations)} disaster declarations from CSV")
    return declarations
