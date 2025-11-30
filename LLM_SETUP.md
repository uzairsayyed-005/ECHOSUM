# LLM Configuration for Meeting Summarizer

## Quick Setup Guide

### Option 1: OpenAI (Recommended for Best Quality)

1. Get API key from https://platform.openai.com/api-keys
2. Set environment variable:

   ```cmd
   # Windows Command Prompt
   set OPENAI_API_KEY=your_api_key_here

   # Windows PowerShell
   $env:OPENAI_API_KEY="your_api_key_here"

   # Linux/Mac
   export OPENAI_API_KEY=your_api_key_here
   ```

3. Install: `pip install openai>=1.0.0`

### Option 2: Ollama (Free, Local)

1. Install Ollama from https://ollama.ai/
2. Pull model: `ollama pull llama3`
3. Start Ollama service
4. Install: `pip install ollama>=0.1.0`

### Option 3: Anthropic Claude

1. Get API key from https://console.anthropic.com/
2. Set environment variable:

   ```cmd
   # Windows Command Prompt
   set ANTHROPIC_API_KEY=your_api_key_here

   # Windows PowerShell
   $env:ANTHROPIC_API_KEY="your_api_key_here"

   # Linux/Mac
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

3. Install: `pip install anthropic>=0.7.0`

### Option 4: Groq (Fast Inference)

1. Get API key from https://console.groq.com/
2. Set environment variable:

   ```cmd
   # Windows Command Prompt
   set GROQ_API_KEY=your_api_key_here

   # Windows PowerShell
   $env:GROQ_API_KEY="your_api_key_here"

   # Linux/Mac
   export GROQ_API_KEY=your_api_key_here
   ```

3. Install: `pip install groq>=0.4.0`

## Installation Commands

```bash
# Install all LLM dependencies
pip install openai>=1.0.0 anthropic>=0.7.0 ollama>=0.1.0 groq>=0.4.0

# Or install specific providers
pip install openai>=1.0.0        # For OpenAI
pip install anthropic>=0.7.0     # For Anthropic
pip install ollama>=0.1.0        # For Ollama
pip install groq>=0.4.0          # For Groq
```

## Usage Examples

### From Command Line

```python
# Run standalone analysis
python llm_analysis.py

# Or use specific provider
from llm_analysis import analyze_meeting_files

# OpenAI
analyze_meeting_files(provider="openai", model="gpt-3.5-turbo")

# Ollama (local)
analyze_meeting_files(provider="ollama", model="llama3")

# Anthropic
analyze_meeting_files(provider="anthropic", model="claude-3-sonnet-20240229")

# Groq
analyze_meeting_files(provider="groq", model="llama-3.1-70b-versatile")
```

### Integrated Workflow

1. Run `python main.py` as usual
2. When prompted, choose "y" for LLM analysis
3. Select your preferred provider (1=OpenAI, 2=Ollama, 3=Anthropic, 4=Groq)
4. Analysis will be saved to `meeting_analysis.txt`

## Model Recommendations

### For Quality (Paid)

- **OpenAI GPT-4**: Best overall quality, slower, more expensive
- **OpenAI GPT-3.5-turbo**: Good quality, faster, cheaper
- **Anthropic Claude-3**: Excellent for analysis, good quality

### For Privacy/Free (Local)

- **Ollama Llama3**: Good quality, completely local
- **Ollama Mistral**: Lighter weight, faster
- **Ollama CodeLlama**: Better for technical content

## Troubleshooting

### Common Issues

1. **API Key Error**: Make sure environment variable is set correctly
2. **Ollama Connection**: Ensure Ollama service is running
3. **Token Limits**: Long meetings may need chunking (automatically handled)
4. **Rate Limits**: Add delays between API calls if needed

### Performance Tips

- Use `gpt-3.5-turbo` for faster, cheaper analysis
- Use Ollama for completely private analysis
- For very long meetings, consider pre-filtering transcript length

## Cost Estimates (OpenAI)

| Meeting Length | Tokens (Est.) | GPT-3.5-turbo Cost | GPT-4 Cost |
| -------------- | ------------- | ------------------ | ---------- |
| 30 minutes     | ~8,000        | $0.02              | $0.32      |
| 1 hour         | ~15,000       | $0.04              | $0.60      |
| 2 hours        | ~30,000       | $0.08              | $1.20      |

_Costs are approximate and may vary based on transcript complexity_
