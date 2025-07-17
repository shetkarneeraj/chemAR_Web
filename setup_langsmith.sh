#!/bin/bash

# chemAR LangSmith Tracing Setup Script
# Run this script to set up environment variables for LangSmith tracing with Gemini

echo "Setting up LangSmith tracing environment for chemAR..."

# Export LangSmith configuration
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="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"
export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
export LANGSMITH_PROJECT="chemAR"

# Export Gemini API key
export GEMINI_API_KEY="AIzaSyAeBdmZ9yE20s6Ub6m3ZSWg3dcxrCblsWQ"

echo "Environment variables set!"
echo "LangSmith tracing is now enabled for Gemini LLM calls."
echo ""
echo "You can now run the application with:"
echo "python3 app.py"
echo ""
echo "To make these permanent, add the export commands to your ~/.bashrc or ~/.zshrc" 