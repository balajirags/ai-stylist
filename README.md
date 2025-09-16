
# AI Stylist

## Problem Description

Shopping for fashion online can be overwhelming due to the vast number of products, styles, and brands available. Customers often struggle to find items that match their preferences, occasions, and budgets, especially when searching for specific requirements like business casual attire, wedding outfits, or color and style combinations.

**AI Stylist** solves this problem by providing an intelligent assistant that understands natural language queries about fashion needs. It leverages advanced retrieval and language models to:

- Interpret user questions and extract key concepts (e.g., product type, style, color, budget, occasion).
- Search a large product catalog using hybrid vector search for relevant items.
- Recommend the best-suited products based on user context and catalog data.
- Provide clear, concise, and personalized fashion advice.

This project streamlines the shopping experience, making it easier for users to discover and select products that fit their unique requirements.

## Features
- Natural language understanding for fashion queries
- Concept extraction and context-aware recommendations
- Hybrid vector search for accurate product retrieval
- Integration with Anthropic LLM and Qdrant vector database
- Modular, extensible Python codebase

## Getting Started

### Prerequisites

- Python 3.10.18
- Docker and Docker Compose (for running with Docker)
- ANTHROPIC API KEY
- Qdrant

### Environment Setup

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd ai-stylist
   ```

2. **Set up the environment variables:**

   Create a `.env` file in the root of the project copying `env.example` and add your Anthropic API key:

   ```
   ANTHROPIC_API_KEY=<your-api-key>
   ```

### Running the Application

You can run the application in two ways: with Docker Compose or locally without Docker Compose.

#### Running with Docker Compose (Recommended)

This is the easiest way to get started, as it automatically sets up all the necessary services, including the Qdrant vector database and the PostgreSQL database.

1. **Build and start the services:**

   ```bash
   docker-compose up --build
   ```

   This will build the Docker images for the application and start all the services in the background.

2. **Access the application:**

   Once the services are running, you can access the Flask API at `http://localhost:5001`.

3. **Stopping the services:**

   To stop the services, press `Ctrl+C` in the terminal where the services are running, or run the following command from the project root:

   ```bash
   docker-compose down
   ```

#### Running Locally without Docker Compose

If you prefer to run the application locally without Docker, you'll need to install the dependencies and start the services manually.

1. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Qdrant and PostgreSQL databases:**

   You'll need to have Qdrant and PostgreSQL running on your local machine. You can refer to the official documentation for instructions on how to install and run them:

   - [Qdrant Documentation](https://qdrant.tech/documentation/)
   - [PostgreSQL Documentation](https://www.postgresql.org/docs/)

3. **Run the ingestion script:**

   Before you can use the application, you need to populate the Qdrant database with the product catalog. You can do this by running the following command:

   ```bash
   python -m ingest.ingest_data_prefect
   ```

4. **Run the Flask application:**

   Once the ingestion is complete, you can start the Flask application by running the following command:

   ```bash
   flask run --host=0.0.0.0 --port=5001
   ```

## How it Works

The AI Stylist is a Retrieval-Augmented Generation (RAG) application that uses a combination of a vector database, a large language model (LLM), and a product catalog to provide fashion recommendations.

### Data Ingestion

The data ingestion process is responsible for populating the Qdrant vector database with the product catalog. The product catalog is stored in a CSV file named `data/products_catalog.csv`. The ingestion script reads this file, processes the data, and then uploads it to the Qdrant database.

Each product in the CSV file has the following columns:

- `ProductID`: A unique identifier for the product.
- `ProductName`: The name of the product.
- `ProductBrand`: The brand of the product.
- `Gender`: The gender that the product is intended for.
- `Price (INR)`: The price of the product in Indian Rupees.
- `Description`: A description of the product.
- `PrimaryColor`: The primary color of the product.

The ingestion script combines the `ProductName` and `Description` fields to create a text that is then used to generate vector embeddings for each product. These embeddings are then stored in the Qdrant database, which allows for efficient similarity-based searches.

### RAG Pipeline

The RAG pipeline is the core of the application. It takes a user's question, retrieves relevant products from the Qdrant database, and then uses an LLM to generate a recommendation.

The pipeline consists of the following steps:

1. **Concept Extraction:** The user's question is first passed to an LLM to extract key concepts, such as product types, styles, and colors. This is done using a prompt that instructs the LLM to act as a stylist and product search planner.

2. **Product Retrieval:** The extracted concepts are then used to query the Qdrant database for relevant products. The application uses a hybrid search approach, combining dense and sparse vectors to improve the relevance of the search results.

3. **Recommendation Generation:** The retrieved products are then passed to another LLM, along with the original question, to generate a recommendation. This is done using a prompt that instructs the LLM to act as a fashion advisor and to use the retrieved products to answer the question.

### LLM Usage

The application uses the Anthropic Claude 3.5 Sonnet model for all its LLM tasks. The following prompts are used:

- **Concept Extraction Prompt:** This prompt is used to extract key concepts from the user's question. It instructs the LLM to act as a stylist and product search planner and to return a comma-separated list of product types, styles, colors, or occasions.

- **Recommendation Prompt:** This prompt is used to generate a recommendation based on the retrieved products. It instructs the LLM to act as a fashion advisor and to use the retrieved products to answer the question.

- **Evaluation Prompt:** This prompt is used to evaluate the relevance of the generated answer to the given question. It instructs the LLM to act as an expert evaluator for a RAG system and to classify the relevance of the answer as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

## Usage

Once the application is running, you can interact with it through the CLI or the API.

### CLI

To use the CLI, run the following command:

```bash
python cli.py
```

This will start an interactive prompt that allows you to ask questions to the AI Stylist.

### API

The application also provides a Flask API with the following endpoints:

- `POST /question`: This endpoint takes a question as input and returns a recommendation.
- `POST /feedback`: This endpoint allows you to provide feedback on a recommendation.

### Examples

Here are a few examples of questions you can ask:

- "I am a woman and need business casual attire."
- "Looking for a red dress for a party under 2000 INR."
- "Suggest an outfit for an Indian wedding."
- "I am a man in my 40s, looking for a shirt to match my black jeans and boots."
