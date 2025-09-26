import os
import traceback
from typing import Dict, List
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectordb import VectorDB
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import PyPDF2
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()


def load_documents() -> List[dict]:
    """
    Load documents for demonstration.
    Reads PDF files from the data directory and returns them as a list of dicts.
    """
    results = []
    data_dir = "../data"
    # Directories
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(root, file)
                try:
                    pdf_reader = PdfReader(file_path)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() or ""

                    if text.strip():
                        results.append({
                            "content": text,
                            "metadata": {"source": file_path}
                        })
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    # TODO: Implement document loading
    # HINT: Read the documents from the data directory
    # HINT: Return a list of documents
    # HINT: Your implementation depends on the type of documents you are using (.txt, .pdf, etc.)

    # Your implementation here
    print(f"Loaded {len(results)} documents from {data_dir}")


    return results


class RAGAssistant:
    """
    A simple RAG-based AI assistant using ChromaDB and multiple LLM providers.
    Supports Groq API.
    """

    def __init__(self):
        """Initialize the RAG assistant."""
        # Initialize LLM - check for available API keys in order of preference
        self.llm = self._initialize_llm()
        if not self.llm:
            raise ValueError(
                "No valid API key found. Please set one of: "
                "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

        # Initialize vector database
        self.vector_db = VectorDB()

        # Create RAG prompt template
        # TODO: Implement your RAG prompt template
        # HINT: Use ChatPromptTemplate.from_template() with a template string
        # HINT: Your template should include placeholders for {context} and {question}
        # HINT: Design your prompt to effectively use retrieved context to answer questions
        self.template_text = (
    "You are a helpful chemistry assistant. Use ONLY the provided CONTEXT to answer the QUESTION. "
    "If the CONTEXT contains relevant information, use it directly in your answer. "
    "If the CONTEXT does not contain the answer, say 'I don't know'. "
    "Always include supporting text or examples from the CONTEXT in your answer.\n\n"
    "CONTEXT:\n{context}\n\n"
    "QUESTION:\n{question}\n\n"
    "Answer:"
)
 # Your implementation here
        self.prompt_template = ChatPromptTemplate.from_template(self.template_text)

        # Create the chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        print("RAG Assistant initialized successfully")

    def _initialize_llm(self):
        """
        Initialize the LLM by checking for available API keys.
        Tries OpenAI, Groq, and Google Gemini in that order.
        """
        # Check for OpenAI API key
        # if os.getenv("OPENAI_API_KEY"):
        #     model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        #     print(f"Using OpenAI model: {model_name}")
        #     return ChatOpenAI(
        #         api_key=os.getenv("OPENAI_API_KEY"), model=model_name, temperature=0.0
        #     )

        if os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            print(f"Using Groq model: {model_name}")
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"), model=model_name, temperature=0.0
            )

        # elif os.getenv("GOOGLE_API_KEY"):
        #     model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
        #     print(f"Using Google Gemini model: {model_name}")
        #     return ChatGoogleGenerativeAI(
        #         google_api_key=os.getenv("GOOGLE_API_KEY"),
        #         model=model_name,
        #         temperature=0.0,
        #     )

        else:
            raise ValueError(
                "No valid API key found. Please set one of: OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

    def add_documents(self, documents: List) -> None:
        print("[App] Starting add_documents", flush=True)
        print(f"[App] Document count: {len(documents)}", flush=True)
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents
        """
        self.vector_db.add_documents(documents)
        print("A: After add_documents", flush=True)


    def invoke(self, input: str, n_results: int = 3) -> str:
        """
        Query the RAG assistant.

        Args:
            input: User's input
            n_results: Number of relevant chunks to retrieve

        Returns:
            Dictionary containing the answer and retrieved context
        """
        llm_answer = ""
        # TODO: Implement the RAG query pipeline
        # HINT: Use self.vector_db.search() to retrieve relevant context chunks
        # HINT: Combine the retrieved document chunks into a single context string
        # HINT: Use self.chain.invoke() with context and question to generate the response
        # HINT: Return a string answer from the LLM

        # Your implementation here
        results = self.vector_db.search(input, n_results=n_results)
        # print("\n[DEBUG] Retrieved Chunks:")
        # for i, doc in enumerate(results["documents"]):
        #     print(f"--- Result {i+1} ---\n{doc[:300]}...\n")
        context = "\n\n".join(results["documents"]) if results["documents"] else "No relevant context found."
        res= self.chain.invoke({"context": context, "question": input})
        return res


def main():
    print("MAIN() STARTED", flush=True)

    """Main function to demonstrate the RAG assistant."""
    try:
        # Initialize the RAG assistant
        print("Initializing RAG Assistant...")
        assistant = RAGAssistant()

        # Load sample documents
        print("\nLoading documents...")
        sample_docs = load_documents()
        print(f"Loaded {len(sample_docs)} sample documents")
        print("[DEBUG] before adding documents", flush=True)
        assistant.add_documents(sample_docs)
        print("[DEBUG] Finished adding documents, about to start input loop", flush=True)

        print("Reached before user input loop!", flush=True)
        done = False
        print("DEBUG: About to enter input loop", flush=True)


        while not done: 
            print("\n==========", flush=True)
            print("Ready to answer questions!", flush=True)
            question = input("Enter a question or 'quit' to exit: ")
            if question.lower() == "quit":
                done = True
                print("MAIN() EXITING inside if ", flush=True)

            elif question.strip() == "":
                print("MAIN() EXITING inside elif", flush=True)

                continue
            else:
                result = assistant.invoke(question)
                print(result)
                print("MAIN() EXITING inside else", flush=True)


    except Exception as e:
        print(f"Error running RAG assistant: {e}")
        print("[DEBUG] Exception occurred after document addition but before prompt.")

        traceback.print_exc()
        # print("Make sure you have set up your .env file with at least one API key:")
        # print("- GROQ_API_KEY (Groq Llama models)")
    print("MAIN() EXITING", flush=True)



if __name__ == "__main__":
    main()
