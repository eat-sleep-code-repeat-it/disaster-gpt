from typing import Optional
from pydantic import BaseModel, Field, field_validator
from datetime import date

class DisasterDeclaration(BaseModel):
    """
    Use this model when working with individual disaster declaration.
    """
    disasterNumber: int = Field(description="Sequentially assigned number used to designate an event or incident declared as a disaster")
    declarationTitle: str =  Field(description="Title for the disaster	")
    state: str = Field(description="The name or phrase describing the U.S. state, district, or territory")
    designatedArea: str = Field(description="The name or phrase describing the geographic area that was included in the declaration")
    declarationType: str = Field(description="Two character code that defines if this is a major disaster, fire management, or emergency declaration")
    declarationDate: date = Field(description="Date the disaster was declared")
    incidentBeginDate: date = Field(description="Date the incident itself began")
    incidentEndDate: Optional[date] = Field(default=None, description="Date the incident itself ended")
    fipsStateCode: str = Field(description="FIPS two-digit numeric code used to identify the United States, the District of Columbia, US territories, outlying areas of the US and freely associated states")
    fipsCountyCode: str = Field(description="FIPS three-digit numeric code used to identify counties and county equivalents in the United States, the District of Columbia, US territories, outlying areas of the US and freely associated states")
    ihProgramDeclared: bool = Field(description="Denotes whether the Individuals and Households program was declared for this disaster")
    iaProgramDeclared: bool = Field(description="Denotes whether the Individual Assistance program was declared for this disaster")
    paProgramDeclared: bool = Field(description="Denotes whether the Public Assistance program was declared for this disaster")
    incidentType:str = Field(description="The primary or official type of incident such as fire or flood. Secondary incident types may have been designated. See the designatedIncidentTypes field")
    
    @field_validator('incidentEndDate', mode='before')
    @classmethod
    def parse_incident_end_date(cls, v):
        if v in (None, ''):
            return None
        return v
