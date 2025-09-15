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

- Docker
- Docker Compose
- Python 3.10.18 or above

### Running Locally

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd ai-stylist
   ```

2. **Set up the environment:**

   Create a `.env` file in the root of the project and add your Anthropic API key:

   ```
   ANTHROPIC_API_KEY=<your-api-key>
   ```

3. **Run the application:**

   ```bash
   docker-compose up
   ```

   This will start the Qdrant vector database and run the Prefect ingestion workflow to populate the database with the product catalog.

## Usage

Once the ingestion is complete, you can use the `rag/fashion_assistant.py` or `rag/smart_fashion_assistant.py` scripts to ask questions to the AI Stylist.

### Examples

Here are a few examples of questions you can ask:

- "I am a woman and need business casual attire."
- "Looking for a red dress for a party under 2000 INR."
- "Suggest an outfit for an Indian wedding."
- "I am Men in 40s,looking for Shirt to match my black jeans and boots"

The AI Stylist will recommend products and explain why they are suitable for your needs.