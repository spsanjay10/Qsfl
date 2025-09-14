import { useState } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'

import Layout from '@/components/Layout'
import Dashboard from '@/pages/Dashboard'
import Clients from '@/pages/Clients'
import Security from '@/pages/Security'
import Analytics from '@/pages/Analytics'
import Settings from '@/pages/Settings'
import Login from '@/pages/Login'
import ShowcaseMode from '@/components/demo/ShowcaseMode'

import { useAuth } from '@/hooks/useAuth'
import LoadingSpinner from '@/components/ui/LoadingSpinner'

function App() {
  const { user, isLoading } = useAuth()
  const [showcaseMode, setShowcaseMode] = useState(false)

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900">
        <LoadingSpinner size="lg" />
      </div>
    )
  }

  if (!user) {
    return <Login />
  }

  return (
    <>
      <Layout>
        <AnimatePresence mode="wait">
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/clients" element={<Clients />} />
            <Route path="/security" element={<Security />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </AnimatePresence>
      </Layout>
      
      {/* Showcase Mode */}
      <ShowcaseMode 
        isActive={showcaseMode} 
        onToggle={() => setShowcaseMode(!showcaseMode)} 
      />
      
      {/* Showcase Mode Toggle Button */}
      {!showcaseMode && (
        <button
          onClick={() => setShowcaseMode(true)}
          className="fixed bottom-6 right-6 bg-gradient-to-r from-primary-600 to-secondary-600 hover:from-primary-700 hover:to-secondary-700 text-white p-3 rounded-full shadow-lg hover:shadow-xl transition-all duration-200 z-40"
          title="Start Showcase Mode"
        >
          🎭
        </button>
      )}
    </>
  )
}

export default App