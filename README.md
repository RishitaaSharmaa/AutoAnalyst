# AutoAnalyst

An AI-powered automated data analysis and machine learning tool built with LangGraph and LangChain. AutoAnalyst uses a conversational AI agent to perform exploratory data analysis, data preprocessing, visualization, and machine learning predictions through natural language commands.

## Features

- **🤖 AI-Powered Analysis**: Conversational interface powered by LangGraph and ChatGroq
- **📊 Comprehensive Data Profiling**: Generate detailed dataset summaries and statistics
- **🧹 Automated Preprocessing**: Remove nulls, duplicates, encode categorical variables, and process dates
- **📈 Visualization Tools**: Create distribution plots and correlation heatmaps
- **🔍 Advanced Analytics**: Correlation analysis, outlier detection, and group-by aggregations
- **🎯 Machine Learning**: Train and predict with built-in ML models
- **💾 Stateful Conversations**: Memory-enabled chat for multi-turn interactions
- **🛠️ Tool-Driven Architecture**: Modular design with extensible tool registry

## Architecture

AutoAnalyst uses a graph-based agentic workflow:
- **LangGraph**: Orchestrates the AI agent and tool execution flow
- **LangChain Tools**: Modular tools for data analysis operations
- **ChatGroq**: LLM backend for intelligent tool selection and response generation
- **Registry System**: Centralized dataset and model management

## Project Structure

```
AutoAnalyst/
├── main.py           # Core LangGraph workflow and agent logic
├── tools.py          # Data analysis and ML tool definitions
├── registry.py       # Dataset and model registry
├── frontend.py       # User interface (Streamlit/Gradio)
├── requirements.txt  # Python dependencies
└── .gitignore       # Git ignore rules
```

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/RishitaaSharmaa/AutoAnalyst.git
cd AutoAnalyst
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
Create a `.env` file in the root directory:
```env
API_KEY=your_groq_api_key_here
```

Get your Groq API key from [https://console.groq.com](https://console.groq.com)

### Available Tools

AutoAnalyst provides the following tools that the AI agent can invoke:

#### Data Cleaning
- `rem_null_duplicates`: Remove null values and duplicate rows
- `encode_categorical_tool`: Label-encode categorical columns
- `preprocess_dates_tool`: Convert date columns to numeric features

#### Data Analysis
- `data_profile_tool`: Generate dataset profile with shape, dtypes, and statistics
- `kpi_summary_tool`: Generate high-level KPIs and metrics
- `correlation_tool`: Find strong correlations above a threshold
- `groupby_summary_tool`: Perform group-by aggregations
- `outlier_detection_tool`: Detect outliers using IQR method

#### Visualization
- `plot_distribution_tool`: Plot distribution histograms
- `plot_correlation_heatmap_tool`: Create correlation heatmaps

#### Machine Learning
- `prediction`: Train models and generate predictions

### Example Workflow

```python
from main import load_dataset, run
import pandas as pd

# Load dataset
df = pd.read_csv("sales_data.csv")
load_dataset("sales", df)

# Multi-step analysis
run("First, clean the sales dataset by removing nulls and duplicates")
run("Generate a profile summary of the sales dataset")
run("Encode all categorical variables in sales")
run("Find correlations above 0.7 in sales")
run("Detect outliers in the sales dataset")
run("Train a model to predict sales using the 'revenue' column")
```

## Configuration

### Customizing the LLM

Modify the LLM configuration in `main.py`:

```python
llm = ChatGroq(
    model="openai/gpt-oss-120b",  # Change model here
    api_key=api_key
)
```

### Adding Custom Tools

1. Define your tool in `tools.py`:
```python
@tool
def your_custom_tool(dataset_id: str) -> dict:
    """Your tool description."""
    df = DATASET_REGISTRY[dataset_id]
    # Your logic here
    return {"result": "data"}
```

2. Register the tool in `main.py`:
```python
tools = [..., your_custom_tool]
```

### Adding Custom Models

Register models in `registry.py`:
```python
from sklearn.ensemble import RandomForestClassifier

MODEL_REGISTRY["random_forest"] = RandomForestClassifier()
```

## How It Works

1. **State Management**: Uses LangGraph's `StateGraph` to manage conversation state
2. **Tool Binding**: LLM is bound to available tools and intelligently selects appropriate ones
3. **Conditional Execution**: Graph conditionally routes between chat and tool execution
4. **Memory**: MemorySaver checkpoint enables stateful multi-turn conversations
5. **Registry Pattern**: Datasets and models are stored in registries for efficient access

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



