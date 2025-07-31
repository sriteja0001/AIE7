#!/bin/bash

echo "🎓 Student Health Advisor App"
echo ""
echo "📋 You need to run TWO terminals:"
echo ""
echo "Terminal 1 (Backend - Python):"
echo "  cd /Users/sriteja/aibootcamp/AIE7/11_Certification_Challenge"
echo "  .venv/bin/python server.py"
echo ""
echo "Terminal 2 (Frontend - Next.js):"
echo "  cd /Users/sriteja/aibootcamp/AIE7/11_Certification_Challenge/frontend"
echo "  npm run dev"
echo ""
echo "Then open http://localhost:3000 in your browser!"
echo ""
echo "Would you like me to start the frontend now? (y/n)"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "📱 Starting Next.js frontend..."
    cd frontend
    npm run dev
else
    echo "Frontend not started. Run 'cd frontend && npm run dev' when ready."
fi 