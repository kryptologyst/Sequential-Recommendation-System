# Sequential Recommendation System

Production-ready implementation of sequential recommendation systems that considers the temporal order of user interactions to predict the next items a user might be interested in.

## Overview

This project implements multiple sequential recommendation approaches:

- **Markov Chain Model**: Simple but effective baseline using item-to-item transition probabilities
- **GRU4Rec**: Deep learning approach using Gated Recurrent Units for sequence modeling
- **Comprehensive Evaluation**: Multiple metrics including precision, recall, NDCG, diversity, and coverage

## Features

- **Modern Architecture**: Clean, modular code with type hints and comprehensive documentation
- **Multiple Models**: From simple Markov chains to advanced neural networks
- **Realistic Data Generation**: Synthetic data with user preferences, popularity bias, and sequential dependencies
- **Comprehensive Evaluation**: Standard recommendation metrics plus diversity and coverage analysis
- **Interactive Demo**: Streamlit-based web interface for exploring recommendations
- **Production Ready**: Proper project structure, configuration management, and testing

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Sequential-Recommendation-System.git
cd Sequential-Recommendation-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Generate sample data:
```bash
python scripts/train.py
```

4. Launch the interactive demo:
```bash
streamlit run scripts/demo.py
```

### Basic Usage

```python
from src.models.sequential_models import MarkovChainModel, GRU4RecModel
from src.data.data_pipeline import generate_sample_data
from src.utils.evaluation import SequentialEvaluator

# Generate sample data
generate_sample_data()

# Load and process data
from src.data.data_pipeline import DataLoader, SequenceProcessor
loader = DataLoader("data")
interactions_df = loader.load_interactions()
processor = SequenceProcessor()
sequences = processor.interactions_to_sequences(interactions_df)

# Train models
markov_model = MarkovChainModel()
markov_model.fit(sequences)

gru_model = GRU4RecModel(num_items=100)
gru_model.fit(sequences)

# Get recommendations
recommendations = markov_model.predict_next([1, 2, 3], k=5)
print(f"Recommended items: {recommendations}")
```

## Project Structure

```
sequential-recommendation-system/
├── src/
│   ├── models/
│   │   └── sequential_models.py    # Model implementations
│   ├── data/
│   │   └── data_pipeline.py        # Data generation and processing
│   └── utils/
│       └── evaluation.py           # Evaluation metrics
├── configs/
│   └── config.yaml                 # Configuration file
├── scripts/
│   ├── train.py                    # Training script
│   └── demo.py                     # Streamlit demo
├── data/                           # Data directory (auto-generated)
├── results/                        # Evaluation results
├── tests/                          # Unit tests
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## Data Format

The system uses three canonical CSV files:

### interactions.csv
```csv
user_id,item_id,timestamp,weight
0,1,1640995200,1.0
0,2,1640995300,1.0
1,1,1640995400,1.0
```

### items.csv
```csv
item_id,title,category,price,rating,features
0,Item_0,Electronics,99.99,4.5,feature_0
1,Item_1,Books,19.99,4.2,feature_1
```

### users.csv
```csv
user_id,age,gender,location,signup_date
0,25,M,US,2023-01-15
1,30,F,UK,2023-02-20
```

## Models

### Markov Chain Model
- **Type**: Baseline sequential model
- **Approach**: Item-to-item transition probabilities
- **Pros**: Simple, interpretable, fast
- **Cons**: Limited to short-term dependencies

### GRU4Rec Model
- **Type**: Deep learning sequential model
- **Approach**: Gated Recurrent Units for sequence modeling
- **Pros**: Captures long-term dependencies, state-of-the-art performance
- **Cons**: Requires more data, computationally intensive

## Evaluation Metrics

### Recommendation Quality
- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Hit Rate@K**: Fraction of users with at least one relevant recommendation

### Diversity & Coverage
- **Intra-list Diversity**: Diversity within recommendation lists
- **Coverage**: Fraction of catalog items that are recommended
- **Popularity Bias**: Bias towards popular items in recommendations

## Configuration

The system uses YAML configuration files for easy customization:

```yaml
# Data configuration
data:
  interactions_file: "data/interactions.csv"
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15

# Model configuration
models:
  markov_chain:
    smoothing: 0.1
    order: 1
  
  gru4rec:
    hidden_size: 128
    num_layers: 2
    dropout: 0.25
    learning_rate: 0.001
    epochs: 50

# Evaluation configuration
evaluation:
  metrics: ["precision@5", "precision@10", "recall@5", "recall@10", "ndcg@5", "ndcg@10"]
  k_values: [5, 10, 20]
```

## Training

### Command Line Training
```bash
python scripts/train.py --config configs/config.yaml
```

### Programmatic Training
```python
from src.models.sequential_models import Config
from scripts.train import main

config = Config.from_yaml("configs/config.yaml")
main()
```

## Interactive Demo

The Streamlit demo provides:

1. **User Recommendations**: Select a user and see personalized recommendations
2. **Item Search**: Search for items and find similar products
3. **Model Performance**: Compare different models' performance metrics
4. **Data Overview**: Explore the dataset statistics and distributions

Launch the demo:
```bash
streamlit run scripts/demo.py
```

## API Usage

For production deployment, the system can be wrapped in a FastAPI service:

```python
from fastapi import FastAPI
from src.models.sequential_models import MarkovChainModel

app = FastAPI()
model = MarkovChainModel()

@app.post("/recommend")
async def recommend(user_sequence: List[int], k: int = 10):
    recommendations = model.predict_next(user_sequence, k=k)
    return {"recommendations": recommendations}
```

## Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src tests/
```

## Performance Considerations

### Data Size
- **Small datasets** (< 10K interactions): Markov Chain models work well
- **Medium datasets** (10K-100K interactions): GRU4Rec recommended
- **Large datasets** (> 100K interactions): Consider distributed training

### Computational Requirements
- **CPU**: Sufficient for Markov Chain models
- **GPU**: Recommended for GRU4Rec training
- **Memory**: 4GB+ RAM for medium datasets

## Extending the System

### Adding New Models
1. Inherit from `BaseSequentialModel`
2. Implement `fit()`, `predict_next()`, and `evaluate()` methods
3. Add to the model registry in `train.py`

### Custom Metrics
1. Add new metric functions to `SequentialEvaluator`
2. Include in the evaluation pipeline
3. Update the demo visualization

### Data Sources
1. Implement custom data loaders
2. Convert to canonical CSV format
3. Use existing processing pipeline

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Data not found**: Run `python scripts/train.py` to generate sample data
3. **Import errors**: Ensure all dependencies are installed

### Performance Issues

1. **Slow training**: Reduce model complexity or use smaller datasets
2. **Poor recommendations**: Check data quality and model parameters
3. **Memory issues**: Process data in batches or reduce sequence length

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{sequential_recommendation_system,
  title={Sequential Recommendation System},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Sequential-Recommendation-System}
}
```

## Acknowledgments

- GRU4Rec paper: "Session-based Recommendations with Recurrent Neural Networks"
- RecBole library for inspiration on evaluation metrics
- Streamlit community for the excellent demo framework
# Sequential-Recommendation-System
