# OpenManus

OpenManus is an AI-powered automation tool that supports multiple AWS Bedrock models including Claude and Amazon Nova Pro.

## Installation

### Prerequisites

1. **Install Git**
   ```bash
   sudo yum install -y git
   ```

2. **AWS Configuration**
   - The EC2 instance should have a role associated to access Bedrock models
   - Configure AWS CLI with your region:
   ```bash
   aws configure
   # Only set region to us-east-1 or your preferred region
   ```

3. **Install UV**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/satishk01/OpenManus.git
   cd OpenManus
   ```

2. **Create and activate virtual environment**
   ```bash
   uv venv --python 3.12
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   uv pip install -r requirements.txt
   ```

4. **Browser Automation Tool (Optional)**
   ```bash
   playwright install
   ```

## Configuration

### Supported Models

OpenManus supports multiple AWS Bedrock models:

- **Claude Models**: `*.anthropic.claude-*` (e.g., `us.anthropic.claude-3-7-sonnet-20250219-v1:0`)
- **Amazon Nova Pro**: `*.amazon.nova-*` (e.g., `us.amazon.nova-pro-v1:0`)

### Model Configuration

Edit `config/config.toml` to configure your preferred model:

#### Claude Configuration (Default)
```toml
[llm]
api_type = "aws"
model = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
base_url = "bedrock-runtime.us-east-1.amazonaws.com"
max_tokens = 8192
temperature = 1.0
api_key = "bear"
```

#### Amazon Nova Pro Configuration
```toml
[llm]
api_type = "aws"
model = "us.amazon.nova-pro-v1:0"
base_url = "bedrock-runtime.us-east-1.amazonaws.com"
max_tokens = 8192
temperature = 0.7
api_key = "bear"
```

### Model Switching

You can easily switch between models by changing the `model` field in `config.toml`. The system automatically detects the model type and applies appropriate formatting and API calls.

### Model-Specific Settings

- **Claude Models**: Work well with temperature 1.0, support streaming and tools
- **Nova Pro Models**: Optimized with temperature 0.7, support streaming and tools

## Running OpenManus

```bash
python main.py
```

## Features

- **Multi-Model Support**: Seamlessly switch between Claude and Nova Pro models
- **Automatic Model Detection**: System automatically detects model type from ID
- **Backward Compatibility**: Existing Claude configurations continue to work
- **Error Handling**: Comprehensive validation and error messages
- **Streaming Support**: Both models support streaming responses
- **Tool Integration**: Function calling support for both model types

## Troubleshooting

### Model Configuration Issues

If you encounter model-related errors:

1. **Invalid Model ID**: Ensure your model ID follows the correct pattern:
   - Claude: `*.anthropic.claude-*`
   - Nova Pro: `*.amazon.nova-*`

2. **Unsupported Model**: Check that your model is available in your AWS region

3. **Backward Compatibility**: Legacy Claude model IDs are supported with fallback handling

### Common Error Messages

- `INVALID_MODEL_ID`: Model ID doesn't match expected patterns
- `UNSUPPORTED_MODEL`: Model type not supported
- `NO_HANDLER`: No handler available for the model type
- `HANDLER_CREATION_ERROR`: Unexpected error during handler creation