#!/bin/bash

# chemAR Deployment Script
# This script updates dependencies and restarts the application

echo "🚀 Starting chemAR deployment..."

# Update pip
echo "📦 Updating pip..."
python3 -m pip install --upgrade pip

# Install/Update requirements
echo "📚 Installing requirements..."
python3 -m pip install -r requirements.txt

# Set up LangSmith environment variables (optional)
echo "⚙️  Setting up environment variables..."
export LANGSMITH_TRACING=false  # Set to true to enable tracing
export LANGSMITH_API_KEY="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"
export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
export LANGSMITH_PROJECT="chemAR"
export GEMINI_API_KEY="AIzaSyAeBdmZ9yE20s6Ub6m3ZSWg3dcxrCblsWQ"

# Restart gunicorn service (if using systemd)
echo "🔄 Restarting gunicorn service..."
if systemctl is-active --quiet gunicorn; then
    sudo systemctl restart gunicorn
    echo "✅ Gunicorn service restarted"
else
    echo "⚠️  Gunicorn service not found or not running"
    echo "📝 You may need to start it manually with:"
    echo "   gunicorn --bind 0.0.0.0:8000 app:app"
fi

# Check if the application is running
echo "🔍 Checking application status..."
sleep 3
if curl -f http://localhost:8000 > /dev/null 2>&1; then
    echo "✅ Application is running successfully!"
else
    echo "❌ Application may not be running. Check logs with:"
    echo "   sudo journalctl -u gunicorn -f"
fi

echo "🎉 Deployment complete!"
echo ""
echo "📋 Summary:"
echo "   - Dependencies installed/updated"
echo "   - LangChain fallback enabled (app will work with or without LangChain)"
echo "   - Direct Gemini API calls available"
echo "   - LangSmith tracing available when LANGSMITH_TRACING=true"
echo ""
echo "🌐 Your chemAR application should now be accessible!" 