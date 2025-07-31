'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showApiKeys, setShowApiKeys] = useState(false);
  const [apiKeys, setApiKeys] = useState({
    openai: '',
    tavily: '',
    langsmith: ''
  });

  // Load API keys from localStorage on component mount
  useEffect(() => {
    const savedKeys = localStorage.getItem('studentHealthApiKeys');
    if (savedKeys) {
      setApiKeys(JSON.parse(savedKeys));
    }
  }, []);

  // Save API keys to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem('studentHealthApiKeys', JSON.stringify(apiKeys));
  }, [apiKeys]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    // Check if API keys are provided
    if (!apiKeys.openai || !apiKeys.tavily || !apiKeys.langsmith) {
      const errorMessage: Message = {
        id: Date.now().toString(),
        type: 'assistant',
        content: 'Please configure your API keys first. Click the settings icon (⚙️) in the top right.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
      return;
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          question: input,
          apiKeys: apiKeys
        }),
      });

      const data = await response.json();

      if (data.error) {
        const errorMessage: Message = {
          id: (Date.now() + 1).toString(),
          type: 'assistant',
          content: `Error: ${data.error}${data.details ? `\n\nDetails: ${data.details}` : ''}`,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, errorMessage]);
      } else {
        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          type: 'assistant',
          content: data.answer,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, assistantMessage]);
      }
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Header */}
      <div className="bg-white/80 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center">
              <span className="text-white text-xl">🎓</span>
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">Student Health Advisor</h1>
              <p className="text-sm text-gray-600">Your AI wellness expert</p>
            </div>
          </div>
          
          {/* Settings Button */}
          <button
            onClick={() => setShowApiKeys(!showApiKeys)}
            className="p-2 rounded-lg bg-gray-100 hover:bg-gray-200 transition-colors"
          >
            <span className="text-lg">⚙️</span>
          </button>
        </div>
      </div>

      {/* API Keys Collapsible */}
      <AnimatePresence>
        {showApiKeys && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-white border-b border-gray-200"
          >
            <div className="max-w-6xl mx-auto px-4 py-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                🔑 API Configuration
              </h2>
              <p className="text-sm text-gray-600 mb-4">
                Enter your API keys to get started. These are stored locally in your browser.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    OpenAI API Key *
                  </label>
                  <input
                    type="password"
                    value={apiKeys.openai}
                    onChange={(e) => setApiKeys(prev => ({ ...prev, openai: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white text-gray-900"
                    placeholder="sk-..."
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Tavily API Key *
                  </label>
                  <input
                    type="password"
                    value={apiKeys.tavily}
                    onChange={(e) => setApiKeys(prev => ({ ...prev, tavily: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white text-gray-900"
                    placeholder="tvly-..."
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    LangSmith API Key *
                  </label>
                  <input
                    type="password"
                    value={apiKeys.langsmith}
                    onChange={(e) => setApiKeys(prev => ({ ...prev, langsmith: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white text-gray-900"
                    placeholder="ls_..."
                  />
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Chat Area */}
      <div className="max-w-6xl mx-auto px-4 py-6">
        <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
          {/* Messages */}
          <div className="h-[600px] overflow-y-auto p-6 space-y-4">
            {messages.length === 0 && (
              <div className="text-center py-12">
                <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-2xl">💬</span>
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">
                  Welcome to your Student Health Advisor!
                </h3>
                <p className="text-gray-600 mb-4">
                  Ask me anything about nutrition, stress, sleep, exercise, or mental health.
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-w-md mx-auto">
                  <div className="bg-blue-50 rounded-lg p-3 text-sm">
                    <span className="font-medium text-blue-900">🍎 Nutrition</span>
                    <p className="text-blue-700">"What are healthy meal ideas for busy students?"</p>
                  </div>
                  <div className="bg-green-50 rounded-lg p-3 text-sm">
                    <span className="font-medium text-green-900">🧘 Stress</span>
                    <p className="text-green-700">"How can I manage exam stress?"</p>
                  </div>
                  <div className="bg-purple-50 rounded-lg p-3 text-sm">
                    <span className="font-medium text-purple-900">😴 Sleep</span>
                    <p className="text-purple-700">"What's the optimal sleep schedule?"</p>
                  </div>
                  <div className="bg-orange-50 rounded-lg p-3 text-sm">
                    <span className="font-medium text-orange-900">💪 Exercise</span>
                    <p className="text-orange-700">"Quick workout routines for students"</p>
                  </div>
                </div>
              </div>
            )}
            
            {messages.map((message) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-xs lg:max-w-md px-4 py-3 rounded-2xl ${
                    message.type === 'user'
                      ? 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white'
                      : 'bg-gray-100 text-gray-900'
                  }`}
                >
                  <div className="whitespace-pre-wrap text-sm leading-relaxed">{message.content}</div>
                  <div className={`text-xs mt-2 ${
                    message.type === 'user' ? 'text-blue-100' : 'text-gray-500'
                  }`}>
                    {message.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              </motion.div>
            ))}
            
            {isLoading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex justify-start"
              >
                <div className="bg-gray-100 text-gray-900 px-4 py-3 rounded-2xl">
                  <div className="flex items-center space-x-3">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                    </div>
                    <span className="text-sm">Thinking...</span>
                  </div>
                </div>
              </motion.div>
            )}
          </div>

          {/* Input Form */}
          <div className="border-t border-gray-200 p-4 bg-gray-50">
            <form onSubmit={handleSubmit} className="flex space-x-3">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about nutrition, stress, sleep, exercise, or mental health..."
                className="flex-1 px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white text-gray-900"
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim()}
                className="px-6 py-3 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-xl hover:from-blue-600 hover:to-indigo-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 font-medium"
              >
                Send
              </button>
            </form>
          </div>
        </div>

        {/* AI Capabilities Showcase */}
        <div className="mt-8">
          <div className="text-center mb-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-2">AI Capabilities</h2>
            <p className="text-gray-600">Powered by multiple data sources for comprehensive health guidance</p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {/* Student Health Guides */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="bg-white rounded-2xl p-6 shadow-lg hover:shadow-xl transition-all duration-300 border border-gray-100"
            >
              <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center mb-4">
                <span className="text-white text-xl">📚</span>
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Student Health Guides</h3>
              <p className="text-gray-600 text-sm mb-3">Built-in college student health resources and wellness guides</p>
              <div className="flex items-center text-xs text-blue-600">
                <span className="bg-blue-100 px-2 py-1 rounded-full">Local Database</span>
              </div>
            </motion.div>

            {/* Web Search */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="bg-white rounded-2xl p-6 shadow-lg hover:shadow-xl transition-all duration-300 border border-gray-100"
            >
              <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-emerald-600 rounded-xl flex items-center justify-center mb-4">
                <span className="text-white text-xl">🌐</span>
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Web Search</h3>
              <p className="text-gray-600 text-sm mb-3">Current health trends, news, and up-to-date wellness information</p>
              <div className="flex items-center text-xs text-green-600">
                <span className="bg-green-100 px-2 py-1 rounded-full">Real-time Data</span>
              </div>
            </motion.div>

            {/* Research Papers */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="bg-white rounded-2xl p-6 shadow-lg hover:shadow-xl transition-all duration-300 border border-gray-100"
            >
              <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-violet-600 rounded-xl flex items-center justify-center mb-4">
                <span className="text-white text-xl">📄</span>
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Research Papers</h3>
              <p className="text-gray-600 text-sm mb-3">Academic studies and scholarly articles via ArXiv</p>
              <div className="flex items-center text-xs text-purple-600">
                <span className="bg-purple-100 px-2 py-1 rounded-full">Academic Research</span>
              </div>
            </motion.div>

            {/* Medical Research */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="bg-white rounded-2xl p-6 shadow-lg hover:shadow-xl transition-all duration-300 border border-gray-100"
            >
              <div className="w-12 h-12 bg-gradient-to-r from-red-500 to-pink-600 rounded-xl flex items-center justify-center mb-4">
                <span className="text-white text-xl">🏥</span>
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Medical Research</h3>
              <p className="text-gray-600 text-sm mb-3">Clinical studies and healthcare publications via PubMed</p>
              <div className="flex items-center text-xs text-red-600">
                <span className="bg-red-100 px-2 py-1 rounded-full">Clinical Studies</span>
              </div>
            </motion.div>
          </div>
        </div>

        {/* Features Grid */}
        <div className="mt-8 grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-white rounded-xl p-4 text-center shadow-sm hover:shadow-md transition-shadow">
            <div className="text-2xl mb-2">🍎</div>
            <h3 className="font-semibold text-gray-900 text-sm">Nutrition</h3>
            <p className="text-xs text-gray-600">Healthy eating tips</p>
          </div>
          <div className="bg-white rounded-xl p-4 text-center shadow-sm hover:shadow-md transition-shadow">
            <div className="text-2xl mb-2">🧘</div>
            <h3 className="font-semibold text-gray-900 text-sm">Stress Management</h3>
            <p className="text-xs text-gray-600">Coping strategies</p>
          </div>
          <div className="bg-white rounded-xl p-4 text-center shadow-sm hover:shadow-md transition-shadow">
            <div className="text-2xl mb-2">😴</div>
            <h3 className="font-semibold text-gray-900 text-sm">Sleep</h3>
            <p className="text-xs text-gray-600">Optimization tips</p>
          </div>
          <div className="bg-white rounded-xl p-4 text-center shadow-sm hover:shadow-md transition-shadow">
            <div className="text-2xl mb-2">💪</div>
            <h3 className="font-semibold text-gray-900 text-sm">Exercise</h3>
            <p className="text-xs text-gray-600">Fitness routines</p>
          </div>
        </div>
      </div>
    </div>
  );
}
