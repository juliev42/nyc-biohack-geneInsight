import streamlit as st # frontend
from langchain_core.messages import AIMessage, HumanMessage
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import langchain_community # for store message history
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.retrieval import create_retrieval_chain


from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory


# def get_prompt(question):
#     """
#     Builds the prompt for the OpenAI Codex model, including instructions and examples for handling the user's question.
#     """
#     prompt = 'Your task is to use NCBI Web APIs to answer genomic questions.\n'
#     prompt += 'You can call Eutils by: "[https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{esearch|efetch|esummary}.fcgi?db={gene|snp|omim}&retmax={}&{term|id}={term|id}]".\n'
#     prompt += 'For DNA sequences, you can use BLAST by: "[https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?CMD={Put|Get}&PROGRAM=blastn&MEGABLAST=on&DATABASE=nt&FORMAT_TYPE={XML|Text}&QUERY={sequence}&HITLIST_SIZE={max_hit_size}]".\n\n'
#     prompt += f'Now, answer the following question:\nQuestion: Tell everything about this gene {question}\n'
#     prompt += "Don't mention how you did it, just provide information about the gene"
#     return prompt

def get_prompt(gene):
    """
    USCS genome browser API call prompt
    """
    prompt = 'Your task is to use UCSC Genome Browser APIs to answer genomic questions.\n'
    prompt += f'You can call its API by: "[hhtps://api.genome.ucsc.edu/search?search={gene}&genome=hg38]" '
    prompt += f'Provide information about the gene {gene} using USCS Genome Browser'
    prompt += "Don't mention how you did it, just provide information about the gene"
    return prompt

def answer_question(question):
    """
    Takes a user question, retrieves necessary data from API, and uses OpenAI models to answer the question.
    """
    prompt = get_prompt(question)
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {
                "role": "user", 
                "content": prompt
            }
        ],
		max_tokens=512,
		temperature=0,
		n=1
	)

    answer = response.choices[0].message.content
    return answer

def get_response_context_retriever_chain(user_query):
    llm = ChatOpenAI(model="gpt-4o-mini")

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name='chat_history'), 
        ('user', '{input}'),
        ('user', "Generate a query that helps find information relevant to the conversation context above.")
    ])

    conversational_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question based on the conversation history:\n\n{chat_history}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    conversation_rag_chain = create_retrieval_chain(prompt, conversational_prompt, llm)

    st.session_state.chat_history.append(HumanMessage(user_query))
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history, 
        "input": user_query
    })

    st.session_state.chat_history.append(AIMessage(content=response['answer']))
    return response['answer']



def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def get_conversational_response(user_query):
    model = ChatOpenAI(model='gpt-4o-mini')
    
    with_message_history = RunnableWithMessageHistory(model, get_session_history)

    config = {"configurable": {"session_id": "gene"}}

    response = with_message_history.invoke(
        [HumanMessage(content=user_query)],
        config=config,
        )
    
    return response.content


# App config
st.set_page_config(page_title="Chat with bio websites", page_icon="🤖")
st.title("Chat with bio databases")

gene_name = st.text_input("Input gene name and get comprehensive information about the gene.")

store = {}

if gene_name is None or gene_name == "":
    st.info("Please enter a gene name")
else:
    if 'chat_history' not in st.session_state:
        initial_response = answer_question(gene_name)
        st.session_state['chat_history'] = [AIMessage(content=f"Hello, I am GeneXtracotr bot. Here is what I found about {gene_name}"),
                                            AIMessage(content=initial_response)]

    # user input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        # initial_response = answer_question(gene_name)
        # response = "I don't know" # <-- here shuold be the function that respond to my questions using chat history

        response = get_conversational_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))


    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
