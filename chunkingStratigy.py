# Agentic Chunking

# https://arxiv.org/pdf/2312.06648.pdf

from langchain_openai import ChatOpenAI
from langchain.chains import create_extraction_chain_pydantic
from pydantic import BaseModel
from langchain import hub
from agentic_chunker import AgenticChunker
    
def agenticChunking(text):
    obj = hub.pull("wfh/proposal-indexing")
    llm = ChatOpenAI(model='gpt-3.5-turbo')
    runnable = obj | llm

    class Sentences(BaseModel):
        sentences: list[str]
        
    # Extraction
    extraction_chain = create_extraction_chain_pydantic(pydantic_schema=Sentences, llm=llm)
    # extraction_chain = llm.with_structured_output(Sentences)
    
    
    def get_propositions(text):
        runnable_output = runnable.invoke({
            "input": text
        }).content
        propositions = extraction_chain.invoke(runnable_output)["text"][0].sentences
        return propositions
        
    paragraphs = text.split("\n\n")
    text_propositions = []
    for i, para in enumerate(paragraphs):
        propositions = get_propositions(para)
        text_propositions.extend(propositions)
        print (f"Done with {i}")

    print (f"You have {len(text_propositions)} propositions")
    print(text_propositions[:10])

    print("#### Agentic Chunking ####")

    
    ac = AgenticChunker()
    ac.add_propositions(text_propositions)
    print(ac.pretty_print_chunks())
    chunks = ac.get_chunks(get_type='list_of_strings')
    print(chunks)
    return chunks
    # documents = [Document(page_content=chunk, metadata={"source": "local"}) for chunk in chunks]
    # rag(documents, "agentic-chunks")