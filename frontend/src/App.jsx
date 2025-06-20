

import React, { useState, useEffect } from 'react';

function App() {
  // State for all agents
  const [learnerId, setLearnerId] = useState('');
  const [codeInput, setCodeInput] = useState('');
  const [quizLogs, setQuizLogs] = useState('');
  const [detectionResults, setDetectionResults] = useState(null);
  const [classificationResults, setClassificationResults] = useState(null);
  const [interventionResults, setInterventionResults] = useState(null);
  const [roadmapResults, setRoadmapResults] = useState(null);
  const [recoveryResults, setRecoveryResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // API call handler
  const callAgent = async (endpoint, data) => {
    setLoading(true);
    setError('');
    try {
      const response = await fetch(`http://localhost:5000/api/${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      
      if (!response.ok) throw new Error('Agent processing failed');
      return await response.json();
    } catch (err) {
      setError(err.message || 'API connection error');
      return null;
    } finally {
      setLoading(false);
    }
  };

  // Agent 1: Misconception Detection
  const runDetection = async () => {
    const data = { learnerId, code: codeInput, quizLogs };
    const result = await callAgent('detect', data);
    if (result) setDetectionResults(result);
  };

  // Agent 2: Misconception Classification
  const runClassification = async () => {
    if (!detectionResults) return;
    const data = { 
      learnerId,
      candidates: detectionResults.candidates 
    };
    const result = await callAgent('classify', data);
    if (result) setClassificationResults(result);
  };

  // Agent 3: Correction Intervention
  const runIntervention = async () => {
    if (!classificationResults) return;
    const data = { 
      learnerId,
      misconceptions: classificationResults.misconceptions 
    };
    const result = await callAgent('correct', data);
    if (result) setInterventionResults(result);
  };

  // Agent 4: Roadmap Adjustment
  const runRoadmapAdjustment = async () => {
    if (!interventionResults) return;
    const data = { 
      learnerId,
      interventions: interventionResults.interventions 
    };
    const result = await callAgent('adjust-roadmap', data);
    if (result) setRoadmapResults(result);
  };

  // Agent 5: Confidence Recovery Tracker
  const runRecoveryTracker = async () => {
    if (!roadmapResults) return;
    const data = { 
      learnerId,
      roadmap: roadmapResults.roadmap 
    };
    const result = await callAgent('track-recovery', data);
    if (result) setRecoveryResults(result);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 to-blue-100 p-6">
      <header className="max-w-6xl mx-auto mb-12 text-center">
        <h1 className="text-4xl font-bold text-indigo-800 mb-2">
          Misconception-Driven Learning Path Correction
        </h1>
        <p className="text-lg text-indigo-600">
          Detect, classify, and correct learning misconceptions
        </p>
      </header>

      <main className="max-w-6xl mx-auto space-y-12">
        {/* Learner Identification */}
        <section className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-2xl font-semibold text-gray-800 mb-4">Learner Profile</h2>
          <div className="flex flex-col md:flex-row gap-4">
            <input
              type="text"
              value={learnerId}
              onChange={(e) => setLearnerId(e.target.value)}
              placeholder="Enter Learner ID"
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none"
            />
            <button
              onClick={() => {
                setDetectionResults(null);
                setClassificationResults(null);
                setInterventionResults(null);
                setRoadmapResults(null);
                setRecoveryResults(null);
              }}
              className="px-6 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
            >
              Reset Session
            </button>
          </div>
        </section>

        {/* Agent 1: Misconception Detection */}
        <section className="bg-white rounded-xl shadow-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-semibold text-gray-800">
              <span className="inline-block w-8 h-8 bg-indigo-600 text-white rounded-full text-center mr-2">1</span>
              Misconception Detection
            </h2>
            <button
              onClick={runDetection}
              disabled={loading || !learnerId}
              className={`px-6 py-2 rounded-lg transition-colors ${
                loading || !learnerId
                  ? 'bg-gray-300 cursor-not-allowed'
                  : 'bg-indigo-600 text-white hover:bg-indigo-700'
              }`}
            >
              Detect Errors
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-4">
            <div>
              <label className="block text-gray-700 mb-2">Code Input</label>
              <textarea
                value={codeInput}
                onChange={(e) => setCodeInput(e.target.value)}
                rows={8}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none font-mono text-sm"
                placeholder="Paste learner's code here..."
              />
            </div>
            <div>
              <label className="block text-gray-700 mb-2">Quiz Logs</label>
              <textarea
                value={quizLogs}
                onChange={(e) => setQuizLogs(e.target.value)}
                rows={8}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none font-mono text-sm"
                placeholder="Enter quiz logs or behavioral patterns..."
              />
            </div>
          </div>

          {detectionResults && (
            <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
              <h3 className="text-lg font-medium text-blue-800 mb-2">Detection Results</h3>
              <div className="space-y-3">
                 {Array.isArray(detectionResults.candidates)
        ? detectionResults.candidates.map((candidate, idx) => (
             <div key={idx} className="p-3 bg-white rounded-lg border border-blue-100">
                    <div className="flex justify-between">
                      <span className="font-medium">{candidate.tag}</span>
                      <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                        Confidence: {candidate.confidence}%
                      </span>
                    </div>
                    <p className="mt-1 text-sm text-gray-600">{candidate.description}</p>
                  </div>
          ))
        : (
          <div className="p-3 bg-white rounded-lg border border-red-100">
           
            <p className="mt-1 text-sm text-gray-600">
              {detectionResults.candidates}
            </p>
          </div>
        )
      }
                
              </div>
            </div>
          )}
        </section>

        {/* Agent 2: Misconception Classification */}
        <section className="bg-white rounded-xl shadow-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-semibold text-gray-800">
              <span className="inline-block w-8 h-8 bg-indigo-600 text-white rounded-full text-center mr-2">2</span>
              Misconception Classification
            </h2>
            <button
              onClick={runClassification}
              disabled={loading || !detectionResults}
              className={`px-6 py-2 rounded-lg transition-colors ${
                loading || !detectionResults
                  ? 'bg-gray-300 cursor-not-allowed'
                  : 'bg-indigo-600 text-white hover:bg-indigo-700'
              }`}
            >
              Classify Errors
            </button>
          </div>

          {classificationResults && (
            <div className="mt-4 p-4 bg-purple-50 rounded-lg border border-purple-200">
              <h3 className="text-lg font-medium text-purple-800 mb-2">Classification Results</h3>
              <div className="space-y-4">
                {classificationResults.misconceptions.map((misconception, idx) => (
                  <div key={idx} className="p-4 bg-white rounded-lg border border-purple-100">
                    <div className="flex justify-between items-start">
                      <div>
                        <h4 className="font-semibold text-lg">{misconception.concept}</h4>
                        <span className="inline-block mt-1 px-2 py-1 bg-purple-100 text-purple-800 text-xs rounded-full">
                          {misconception.category}
                        </span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className="text-sm font-medium">Confidence: {misconception.confidence}%</span>
                        <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
                          <div 
                            className={`h-full ${
                              misconception.confidence > 75 ? 'bg-green-500' : 
                              misconception.confidence > 50 ? 'bg-yellow-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${misconception.confidence}%` }}
                          ></div>
                        </div>
                      </div>
                    </div>
                    <div className="mt-3">
                      <h5 className="font-medium text-gray-700">Explanation:</h5>
                      <p className="text-gray-600">{misconception.explanation}</p>
                    </div>
                    <div className="mt-3">
                      <h5 className="font-medium text-gray-700">RAG Sources:</h5>
                      <ul className="list-disc pl-5 text-sm text-gray-600">
                        {misconception.rag_sources.map((source, sIdx) => (
                          <li key={sIdx}>{source}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>

        {/* Agent 3: Correction Intervention */}
        <section className="bg-white rounded-xl shadow-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-semibold text-gray-800">
              <span className="inline-block w-8 h-8 bg-indigo-600 text-white rounded-full text-center mr-2">3</span>
              Correction Intervention
            </h2>
            <button
              onClick={runIntervention}
              disabled={loading || !classificationResults}
              className={`px-6 py-2 rounded-lg transition-colors ${
                loading || !classificationResults
                  ? 'bg-gray-300 cursor-not-allowed'
                  : 'bg-indigo-600 text-white hover:bg-indigo-700'
              }`}
            >
              Generate Interventions
            </button>
          </div>

          {interventionResults && (
            <div className="mt-4 p-4 bg-teal-50 rounded-lg border border-teal-200">
              <h3 className="text-lg font-medium text-teal-800 mb-2">Intervention Plan</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 bg-white rounded-lg border border-teal-100">
                  <h4 className="font-semibold text-lg flex items-center text-teal-700">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                    </svg>
                    Analogies
                  </h4>
                  <p className="mt-2 text-gray-600">{interventionResults.analogy}</p>
                </div>
                
                <div className="p-4 bg-white rounded-lg border border-teal-100">
                  <h4 className="font-semibold text-lg flex items-center text-teal-700">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-11a1 1 0 10-2 0v2H7a1 1 0 100 2h2v2a1 1 0 102 0v-2h2a1 1 0 100-2h-2V7z" clipRule="evenodd" />
                    </svg>
                    Micro-Challenges
                  </h4>
                  <ul className="mt-2 space-y-2">
                    {interventionResults.micro_challenges.map((challenge, idx) => (
                      <li key={idx} className="flex items-start">
                        <span className="inline-block mt-1 mr-2 w-2 h-2 rounded-full bg-teal-500"></span>
                        <span>{challenge}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                
                <div className="p-4 bg-white rounded-lg border border-teal-100">
                  <h4 className="font-semibold text-lg flex items-center text-teal-700">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                      <path d="M10 12a2 2 0 100-4 2 2 0 000 4z" />
                      <path fillRule="evenodd" d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clipRule="evenodd" />
                    </svg>
                    Explanations
                  </h4>
                  <p className="mt-2 text-gray-600">{interventionResults.explanation}</p>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* Agent 4: Roadmap Adjustment */}
        <section className="bg-white rounded-xl shadow-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-semibold text-gray-800">
              <span className="inline-block w-8 h-8 bg-indigo-600 text-white rounded-full text-center mr-2">4</span>
              Roadmap Adjustment
            </h2>
            <button
              onClick={runRoadmapAdjustment}
              disabled={loading || !interventionResults}
              className={`px-6 py-2 rounded-lg transition-colors ${
                loading || !interventionResults
                  ? 'bg-gray-300 cursor-not-allowed'
                  : 'bg-indigo-600 text-white hover:bg-indigo-700'
              }`}
            >
              Adjust Learning Path
            </button>
          </div>

          {roadmapResults && (
            <div className="mt-4 p-4 bg-amber-50 rounded-lg border border-amber-200">
              <h3 className="text-lg font-medium text-amber-800 mb-4">Adjusted Learning Roadmap</h3>
              <div className="relative">
                {/* Roadmap timeline */}
                <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-amber-300"></div>
                
                <div className="space-y-6 pl-10">
                  {roadmapResults.roadmap.map((module, idx) => (
                    <div key={idx} className="relative">
                      <div className="absolute -left-7 top-4 w-6 h-6 rounded-full bg-amber-500 flex items-center justify-center text-white font-bold">
                        {idx + 1}
                      </div>
                      
                      <div className={`p-4 rounded-lg border ${
                        module.status === 'completed' 
                          ? 'bg-green-50 border-green-200' 
                          : module.status === 'current'
                          ? 'bg-blue-50 border-blue-200 shadow-sm'
                          : module.status === 'adjusted'
                          ? 'bg-amber-100 border-amber-300'
                          : 'bg-white border-gray-200'
                      }`}>
                        <div className="flex justify-between">
                          <h4 className="font-semibold">{module.title}</h4>
                          {module.status === 'adjusted' && (
                            <span className="px-2 py-1 bg-amber-500 text-white text-xs rounded-full">
                              Adjusted
                            </span>
                          )}
                        </div>
                        <p className="mt-1 text-sm text-gray-600">{module.description}</p>
                        <div className="mt-3 flex flex-wrap gap-2">
                          {module.tags.map((tag, tagIdx) => (
                            <span 
                              key={tagIdx} 
                              className="px-2 py-1 bg-indigo-100 text-indigo-800 text-xs rounded-full"
                            >
                              {tag}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </section>

        {/* Agent 5: Confidence Recovery Tracker */}
        <section className="bg-white rounded-xl shadow-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-semibold text-gray-800">
              <span className="inline-block w-8 h-8 bg-indigo-600 text-white rounded-full text-center mr-2">5</span>
              Confidence Recovery Tracker
            </h2>
            <button
              onClick={runRecoveryTracker}
              disabled={loading || !roadmapResults}
              className={`px-6 py-2 rounded-lg transition-colors ${
                loading || !roadmapResults
                  ? 'bg-gray-300 cursor-not-allowed'
                  : 'bg-indigo-600 text-white hover:bg-indigo-700'
              }`}
            >
              Track Recovery Progress
            </button>
          </div>

          {recoveryResults && (
            <div className="mt-4 p-4 bg-green-50 rounded-lg border border-green-200">
              <h3 className="text-lg font-medium text-green-800 mb-4">Recovery Progress</h3>
              
              <div className="mb-6">
                <div className="flex justify-between mb-2">
                  <span className="font-medium">Confidence Index</span>
                  <span className="font-bold text-green-700">{recoveryResults.confidence_index}%</span>
                </div>
                <div className="w-full h-4 bg-gray-200 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-green-500 transition-all duration-1000 ease-out"
                    style={{ width: `${recoveryResults.confidence_index}%` }}
                  ></div>
                </div>
              </div>
              
              <h4 className="font-medium text-gray-700 mb-3">Recovery Timeline</h4>
              <div className="space-y-4">
                {recoveryResults.timeline.map((event, idx) => (
                  <div key={idx} className="flex">
                    <div className="flex flex-col items-center mr-4">
                      <div className={`w-3 h-3 rounded-full ${
                        event.status === 'recovered' ? 'bg-green-500' : 
                        event.status === 'in-progress' ? 'bg-yellow-500' : 'bg-red-500'
                      }`}></div>
                      {idx < recoveryResults.timeline.length - 1 && (
                        <div className="w-0.5 h-full bg-gray-300"></div>
                      )}
                    </div>
                    <div className="pb-4">
                      <p className="font-medium">{event.date}</p>
                      <p className="text-gray-600">{event.description}</p>
                      <div className="mt-2 flex flex-wrap gap-1">
                        {event.flags?.map((flag, flagIdx) => (
                          <span 
                            key={flagIdx} 
                            className={`px-2 py-1 text-xs rounded-full ${
                              flag.type === 'critical' 
                                ? 'bg-red-100 text-red-800' 
                                : 'bg-yellow-100 text-yellow-800'
                            }`}
                          >
                            {flag.message}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>
      </main>

      {/* Loading and Error Indicators */}
      {loading && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-xl shadow-xl flex flex-col items-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mb-4"></div>
            <p className="text-lg font-medium text-gray-800">Processing with AI agents...</p>
          </div>
        </div>
      )}

      {error && (
        <div className="fixed bottom-4 right-4 bg-red-500 text-white p-4 rounded-lg shadow-lg max-w-md z-50">
          <div className="flex justify-between items-start">
            <p>{error}</p>
            <button 
              onClick={() => setError('')}
              className="ml-4 text-white hover:text-gray-200"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;