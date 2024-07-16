# Chat with PDF (Gemini and Langchain based approach)

This repository contains the code for a chatbot that interacts with a corpus of PDF documents using Streamlit, LangChain, and Google Generative AI.

## Running this project 

1. Clone this repo using 
```shell
git clone https://github.com/sandyram19/chat-with-pdf.git
```

2. Install the required packages by running:
```shell 
pip install -r requirements.txt
```

3. Run the project using
```shell
streamlit run chat.py
```


### Explanation of the Overall Approach

1. **PDF Text Extraction**: 
   - Used `PyPDF2` to read and extract text from the provided PDF file.

2. **Text Chunking**:
   - Employed `RecursiveCharacterTextSplitter` from LangChain to split the extracted text into manageable chunks. This helps in efficient processing and retrieval of relevant information.

3. **Embedding and Vector Store**:
   - Generated embeddings for text chunks using `GoogleGenerativeAIEmbeddings` and stored them in a FAISS index for fast similarity searches.

4. **Conversational Chain**:
   - Set up a conversational chain using `ChatGoogleGenerativeAI` and `load_qa_chain` from LangChain, which handles the dialogue flow and maintains context using a buffer memory.

5. **User Interaction**:
   - Built an interactive web application using Streamlit, where users can ask questions and get responses based on the processed PDF content.

### Frameworks/Libraries/Tools Used

- **Streamlit**: Provides an easy way to create web applications for data science and machine learning projects.
- **PyPDF2**: A pure Python library for reading and manipulating PDF files.
- **LangChain**: A framework for building applications with large language models, useful for text processing and managing conversational agents.
- **Google Generative AI**: Used for generating embeddings and conversational responses, enhancing the chatbot's capabilities.
- **FAISS**: A library for efficient similarity search and clustering of dense vectors, used here to store and retrieve text embeddings.
- **python-dotenv**: Helps manage environment variables, ensuring secure handling of API keys.

### Problems Faced and Solutions

1. **Embedding API Integration**:
   - Managing API keys securely using `python-dotenv` to avoid hardcoding sensitive information in the code.

2. **TStreamlit Reruns**:
   - Streamlit tends to rerun the entire code whenever new action is made to ensure every change. But it leads to initialisation of functions fresh again leading to lack of memory.<br>
   - Hence, Session State variables are used to ensure that specific codes are run only once. 

### Future Scope

1. **Improved User Interface**:
   - Enhance the Streamlit interface with better styling, responsiveness, and additional interactive features.

2. **Multiple Document Support**:
   - Allow users to upload multiple PDF documents and query across all documents seamlessly.

3. **Contextual Understanding**:
   - Improve the chatbot's ability to maintain and understand context over longer and more complex conversations.

4. **Feedback Mechanism**:
   - Implement a feedback mechanism where users can rate the chatbot's responses, helping in continuous improvement.

5. **Deployment**:
   - Deploy the chatbot to a cloud platform (e.g., AWS, Heroku) for wider accessibility and robustness.

Feel free to use the provided files and information to set up your GitHub repository and enhance your chatbot application!


