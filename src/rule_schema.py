from typing import List, Optional
from pydantic import BaseModel, Field

class ContractRule(BaseModel):
    category: str = Field(description="The formal billing category this rule applies to.")
    rule_statement: str = Field(description="The plain English fundamental logic of the rule.")
    trigger_condition: str = Field(description="The strict logical condition when this rule activates (e.g., 'hours_worked > 40', or 'Always').")
    action_or_limit: str = Field(description="The numeric/financial limit or action taken (e.g., '1.5x base rate' or 'max $100').")
    requires_approval: bool = Field(description="Whether explicit written approval is required before applying this charge.")
    requires_external_lookup: bool = Field(description="Set to true if the text ambiguously references external documents, Appendices, or Excel tables for the exact rates.")
    external_reference_source: Optional[str] = Field(None, description="If requires_external_lookup is true, list the exact name of the Appendix or Table referenced (e.g. 'Appendix E', 'Pricing Matrix Schedule').")

class Rulebook(BaseModel):
    rules: List[ContractRule]
