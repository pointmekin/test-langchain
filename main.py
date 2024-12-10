from typing import List, Optional, Union
from fastapi import FastAPI
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field, ValidationError
from dto.prompt import PromptMessage

# MARK: Classes

class Classification(BaseModel):
        sentiment: str = Field(description="The sentiment of the text")
        aggressiveness: int = Field(
            description="How aggressive the text is on a scale from 1 to 10"
        )
        language: str = Field(description="The language the text is written in")

class Person(BaseModel):
    """Information about a person."""

    # ^ Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Person,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    name: Optional[str] = Field(default=None, description="The name of the person")
    hair_color: Optional[str] = Field(
        default=None, description="The color of the person's hair if known"
    )
    height: Optional[float] = Field(
        default=None, description="Height measured in any unit"
    )

class PersonData(BaseModel):
    """Extracted data about people."""

    # Creates a model so that we can extract multiple entities.
    people: List[Person]



# MARK: LLM

# Ollama model
llm = ChatOllama(
    model="llama3.2",
    temperature=0,
    # other params...
)

classification_llm = llm.with_structured_output(
    Classification
)

structured_llm = llm.with_structured_output(schema=Person)

# MARK: Fast API app

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/test-chat")
async def test_chat():
    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to Thai. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]
    ai_msg = llm.invoke(messages)
    return {
        "message": ai_msg
    }
    
@app.post("/test-classification")
async def test_classification(message: PromptMessage):
    tagging_prompt = ChatPromptTemplate.from_template(
    """
    Extract the desired information from the following passage.

    Only extract the properties mentioned in the 'Classification' function.

    Passage:
    {input}

    Examples:
    Input: I really enjoy this movie. It is amazing.
    Output:
    ```json
    {{
        "sentiment": "positive",
        "aggressiveness": 1,
        "language": "english"
    }}
    ```
    """
    )
    
    inp = message.message
    prompt = tagging_prompt.invoke({"input": inp})
    response = classification_llm.invoke(prompt)

    return {
        "message": response
    }

@app.post("/test-extraction")
async def test_extraction(message: PromptMessage):
    # Define a custom prompt to provide instructions and any additional context.
    # 1) You can add examples into the prompt template to improve extraction quality
    # 2) Introduce additional parameters to take context into account (e.g., include metadata
    #    about the document from which the text was extracted.)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert extraction algorithm. "
                "Only extract relevant information from the text. "
                "If you do not know the value of an attribute asked to extract, "
                "return null for the attribute's value.",
            ),
            # Please see the how-to about improving performance with
            # reference examples.
            # MessagesPlaceholder('examples'),
            ("human", "{text}"),
        ]
    )
    
    text = message.message # ex: Alan Smith is 6 feet tall and has blond hair.
    prompt = prompt_template.invoke({"text": text})
    response = structured_llm.invoke(prompt)
    
    return {
        "message": response
    }


@app.post("/test-extraction-multiple-people")
async def test_extraction(message: PromptMessage):
    # Define a custom prompt to provide instructions and any additional context.
    # 1) You can add examples into the prompt template to improve extraction quality
    # 2) Introduce additional parameters to take context into account (e.g., include metadata
    #    about the document from which the text was extracted.)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert extraction algorithm. "
                "Only extract relevant information from the text. "
                "If you do not know the value of an attribute asked to extract, "
                "return null for the attribute's value."
                "You can infer from contextual information provided."
                "If you cannot reasonably infer the value of an attribute asked to extract, "
                "return null for the attribute's value."
            ),
            # Please see the how-to about improving performance with
            # reference examples.
            # MessagesPlaceholder('examples'),
            ("human", "{text}"),
        ]
    )
    
    structured_llm = llm.with_structured_output(schema=PersonData)
    text = message.message # ex: My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me.
    prompt = prompt_template.invoke({"text": text})
    response = structured_llm.invoke(prompt)
    
    return {
        "message": response
    }


