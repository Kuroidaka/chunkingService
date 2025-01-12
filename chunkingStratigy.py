import asyncio
from langchain_openai import ChatOpenAI
from langchain.chains import create_extraction_chain_pydantic
from pydantic import BaseModel
from langchain import hub
from typing import List
from agentic_chunker import AgenticChunker

async def agenticChunking(paragraphs: List[str]):
    obj = hub.pull("wfh/proposal-indexing")
    llm = ChatOpenAI(model='gpt-4o-mini')
    runnable = obj | llm

    class Sentences(BaseModel):
        sentences: list[str]

    # Extraction
    extraction_chain = create_extraction_chain_pydantic(pydantic_schema=Sentences, llm=llm)

    async def get_propositions(text):
        runnable_output = (await asyncio.to_thread(runnable.invoke, {
            "input": text
        })).content
        propositions = (await asyncio.to_thread(extraction_chain.invoke, runnable_output))["text"][0].sentences
        return propositions

    # Concurrent processing
    async def process_paragraph(para, idx):
        propositions = await get_propositions(para)
        print(f"Done with {idx}")
        return propositions

    tasks = []
    for i, para in enumerate(paragraphs):
        try:
            task = process_paragraph(para, i)
            tasks.append(task)
        except Exception:
            # Skip this paragraph on error
            continue

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Flatten results while skipping exceptions
    text_propositions = [
        prop for result in results if not isinstance(result, Exception) for prop in result
    ]

    print(f"You have {len(text_propositions)} propositions")
    print(text_propositions[:10])

    print("#### Agentic Chunking ####")

    ac = AgenticChunker()
    ac.add_propositions(text_propositions)
    print(ac.pretty_print_chunks())
    chunks = ac.get_chunks(get_type='list_of_strings')
    print(chunks)
    return chunks
