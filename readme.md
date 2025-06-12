# Advanced RAG Search for Question Answer Dataset Creation

This is a Streamlit web application for performing Retrieval-Augmented Generation (RAG) search on your PDF documents. It uses the Marker API for high-quality text extraction, Moondream for image understanding, and Qdrant for efficient vector search.

### Key Features

-   **Upload PDFs**: Directly upload your documents through the web interface.
-   **Advanced OCR**: Uses the Marker API for state-of-the-art, layout-aware text and table extraction.
-   **Image Understanding**: Automatically generates descriptions for images within your PDFs using the Moondream API.
-   **Intelligent Chunking**: Uses LangChain to intelligently split documents into context-aware chunks.
-   **Vector Search**: Ask questions in natural language to retrieve the most relevant passages from your documents, powered by Qdrant.

## Prerequisites

Before you begin, ensure you have the following:
1.  **Python 3.9+** installed on your system.
2.  **API Keys** for the following services:
    -   [**Qdrant Cloud**](https://cloud.qdrant.io/): A vector database. You will need the **Cluster URL** and an **API Key**.
    -   [**Marker API From Datalab**](https://marker.services.datalab.tech/): For document-to-markdown conversion.
    -   [**Moondream API**](https://github.com/vikhyat/moondream): For image-to-text description.

## Setup and Installation

Follow these steps to get the application running locally.

**1. Clone the Repository**
   Download the project files or clone the repository to your local machine.

**2. Create a file for the environment variables** 
    Create a file in the root directory with the name: '.env'.

    Go to the respective platforms(qdrant,datalab,moondream) to acquire the free api key's Define your api keys and urls with the following variable names:

    ```bash
    QDRANT_URL = "QDRANT_CLOUD_INSTANCE_URL"
    QDRANT_API_KEY = "QDRANT_CLOUD_INSTANCE_API_KEY"
    #Datalab is the platform to get the api keys to use Marker-OCR.
    DATALAB_API_KEY = "DATALAB_API_KEY"
    DATALAB_MARKER_URL = "https://www.datalab.to/api/v1/marker" #Need not modify this url, this is common for all accounts
    MOONDREAM_API_KEY = "MOONDREAM_API_KEY"
    ```
    
**3. Create a Virtual Environment**
   Navigate into the project directory and create a virtual environment. This keeps your project dependencies isolated.
   ```bash
   # For Windows
   python -m venv venv
   .\venv\Scripts\activate

   # For macOS / Linux
   python -m venv venv
   source venv/bin/activate
   ```

**4. Install all neccesary packages**
    Run the following command in your terminal(root directory) to install all the necceasry python libraries and packages needed
    ```bash
    pip install -r requirements.txt
    ```

**5. Run the streamlit Application**
    Run the application by running the command in the terminal:
    ```bash
    streamlit run app.py
    ```
