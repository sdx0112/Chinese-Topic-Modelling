"""
Use small manually-labeled set as examples and try few-shot of GPT model.
"""

from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate