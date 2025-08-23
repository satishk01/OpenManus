# Model Migration Guide

This guide helps you migrate between different AWS Bedrock models supported by OpenManus.

## Supported Models

### Claude Models
- **Pattern**: `*.anthropic.claude-*`
- **Examples**:
  - `us.anthropic.claude-3-7-sonnet-20250219-v1:0`
  - `us.anthropic.claude-3-5-sonnet-v2:0`
  - `eu.anthropic.claude-3-haiku-20240307-v1:0`

### Amazon Nova Pro Models
- **Pattern**: `*.amazon.nova-*`
- **Examples**:
  - `us.amazon.nova-pro-v1:0`
  - `eu.amazon.nova-pro-v1:0`

## Migration Steps

### From Claude to Nova Pro

1. **Update config.toml**:
   ```toml
   # Before (Claude)
   [llm]
   api_type = "aws"
   model = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
   temperature = 1.0
   
   # After (Nova Pro)
   [llm]
   api_type = "aws"
   model = "us.amazon.nova-pro-v1:0"
   temperature = 0.7  # Nova Pro works better with lower temperature
   ```

2. **Restart the application**:
   ```bash
   python main.py
   ```

3. **Verify the change**:
   - Check console output for model validation messages
   - Look for: `Validated model: us.amazon.nova-pro-v1:0 (type: nova_pro)`

### From Nova Pro to Claude

1. **Update config.toml**:
   ```toml
   # Before (Nova Pro)
   [llm]
   api_type = "aws"
   model = "us.amazon.nova-pro-v1:0"
   temperature = 0.7
   
   # After (Claude)
   [llm]
   api_type = "aws"
   model = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
   temperature = 1.0  # Claude works well with higher temperature
   ```

2. **Restart the application**

## Model-Specific Configurations

### Claude Optimal Settings
```toml
[llm]
api_type = "aws"
model = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
base_url = "bedrock-runtime.us-east-1.amazonaws.com"
max_tokens = 8192
temperature = 1.0
api_key = "bear"
```

### Nova Pro Optimal Settings
```toml
[llm]
api_type = "aws"
model = "us.amazon.nova-pro-v1:0"
base_url = "bedrock-runtime.us-east-1.amazonaws.com"
max_tokens = 8192
temperature = 0.7
api_key = "bear"
```

## Backward Compatibility

### Legacy Claude Models
If you have an older Claude model ID that doesn't match the new pattern, the system will:
1. Attempt to detect it as a Claude model
2. Fall back to Claude handler with default settings
3. Display a warning message

### Example Legacy Support
```toml
# This will still work with backward compatibility
[llm]
model = "claude-3-sonnet-20240229"  # Legacy format
```

## Troubleshooting Migration

### Common Issues

1. **Model Not Found**
   - **Error**: Model ID not recognized
   - **Solution**: Verify the model is available in your AWS region
   - **Check**: AWS Bedrock console for available models

2. **Invalid Pattern**
   - **Error**: `INVALID_MODEL_ID`
   - **Solution**: Ensure model ID follows correct pattern
   - **Claude**: Must contain `anthropic.claude`
   - **Nova Pro**: Must contain `amazon.nova`

3. **Performance Issues**
   - **Issue**: Responses seem off after migration
   - **Solution**: Adjust temperature settings
   - **Claude**: Try temperature 0.7-1.0
   - **Nova Pro**: Try temperature 0.3-0.7

### Validation Commands

Check your configuration:
```bash
# View current config
cat config/config.toml

# Test model validation (in Python)
python -c "
from app.bedrock import ModelDetector
model_id = 'your-model-id-here'
print(f'Model type: {ModelDetector.get_model_type(model_id)}')
print(f'Valid: {ModelDetector.validate_model_id(model_id)}')
"
```

## Best Practices

1. **Test Before Production**: Always test model changes in a development environment
2. **Monitor Performance**: Compare response quality after migration
3. **Adjust Temperature**: Fine-tune temperature based on your use case
4. **Keep Backups**: Save working configurations before making changes
5. **Regional Availability**: Ensure your target model is available in your AWS region

## Support

If you encounter issues during migration:
1. Check the console output for detailed error messages
2. Verify AWS permissions for the new model
3. Ensure the model is available in your region
4. Review the error handling section in the main README