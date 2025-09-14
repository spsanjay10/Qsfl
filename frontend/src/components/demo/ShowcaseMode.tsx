import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  PlayIcon, 
  PauseIcon, 
  ForwardIcon,
  SpeakerWaveIcon,
  SpeakerXMarkIcon 
} from '@heroicons/react/24/outline'
import toast from 'react-hot-toast'

interface ShowcaseModeProps {
  isActive: boolean
  onToggle: () => void
}

const SHOWCASE_STEPS = [
  {
    id: 'welcome',
    title: 'Welcome to QSFL-CAAD',
    description: 'Quantum-Safe Federated Learning with Comprehensive Anomaly and Attack Detection',
    duration: 3000,
    action: () => toast.success('🚀 Starting QSFL-CAAD Showcase'),
  },
  {
    id: 'system_overview',
    title: 'System Overview',
    description: 'Monitor federated learning clients in real-time with advanced security',
    duration: 4000,
    action: () => toast('📊 Viewing system metrics'),
  },
  {
    id: 'add_clients',
    title: 'Adding Clients',
    description: 'Easily add and manage federated learning clients',
    duration: 5000,
    action: () => toast('👥 Managing client connections'),
  },
  {
    id: 'security_monitoring',
    title: 'Security Monitoring',
    description: 'Real-time anomaly detection and threat analysis',
    duration: 4000,
    action: () => toast.error('🔒 Security monitoring active'),
  },
  {
    id: 'attack_simulation',
    title: 'Attack Simulation',
    description: 'Demonstrate system resilience against various attacks',
    duration: 6000,
    action: () => toast.error('⚠️ Simulating attack scenario'),
  },
]

export default function ShowcaseMode({ isActive, onToggle }: ShowcaseModeProps) {
  const [currentStep, setCurrentStep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isMuted, setIsMuted] = useState(false)

  useEffect(() => {
    if (!isActive || !isPlaying) return

    const step = SHOWCASE_STEPS[currentStep]
    if (!step) return

    // Execute step action
    step.action()

    const timer = setTimeout(() => {
      if (currentStep < SHOWCASE_STEPS.length - 1) {
        setCurrentStep(prev => prev + 1)
      } else {
        setIsPlaying(false)
        setCurrentStep(0)
        toast.success('🎉 Showcase completed!')
      }
    }, step.duration)

    return () => clearTimeout(timer)
  }, [isActive, isPlaying, currentStep])

  if (!isActive) return null

  const currentStepData = SHOWCASE_STEPS[currentStep]
  const progress = ((currentStep + 1) / SHOWCASE_STEPS.length) * 100

  return (
    <motion.div
      initial={{ opacity: 0, y: -100 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -100 }}
      className="fixed top-4 left-1/2 transform -translate-x-1/2 z-50"
    >
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl border border-gray-200 dark:border-gray-700 p-4 min-w-96">
        {/* Header */}
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white">
            🎭 Showcase Mode
          </h3>
          <button
            onClick={onToggle}
            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
          >
            ✕
          </button>
        </div>

        {/* Current Step */}
        <div className="mb-4">
          <h4 className="text-lg font-bold text-gray-900 dark:text-white mb-1">
            {currentStepData?.title}
          </h4>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            {currentStepData?.description}
          </p>
        </div>

        {/* Progress Bar */}
        <div className="mb-4">
          <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mb-1">
            <span>Step {currentStep + 1} of {SHOWCASE_STEPS.length}</span>
            <span>{Math.round(progress)}%</span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
            <motion.div
              className="bg-gradient-to-r from-primary-500 to-secondary-500 h-2 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center justify-center space-x-3">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="flex items-center justify-center w-10 h-10 bg-primary-600 hover:bg-primary-700 text-white rounded-full transition-colors"
          >
            {isPlaying ? (
              <PauseIcon className="h-5 w-5" />
            ) : (
              <PlayIcon className="h-5 w-5 ml-0.5" />
            )}
          </button>
          
          <button
            onClick={() => {
              if (currentStep < SHOWCASE_STEPS.length - 1) {
                setCurrentStep(prev => prev + 1)
              }
            }}
            disabled={currentStep >= SHOWCASE_STEPS.length - 1}
            className="flex items-center justify-center w-10 h-10 bg-gray-600 hover:bg-gray-700 disabled:bg-gray-400 text-white rounded-full transition-colors"
          >
            <ForwardIcon className="h-5 w-5" />
          </button>

          <button
            onClick={() => setIsMuted(!isMuted)}
            className="flex items-center justify-center w-10 h-10 bg-gray-600 hover:bg-gray-700 text-white rounded-full transition-colors"
          >
            {isMuted ? (
              <SpeakerXMarkIcon className="h-5 w-5" />
            ) : (
              <SpeakerWaveIcon className="h-5 w-5" />
            )}
          </button>
        </div>
      </div>
    </motion.div>
  )
}