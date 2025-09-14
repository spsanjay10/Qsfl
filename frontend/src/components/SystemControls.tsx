import { useState } from 'react'
import { motion } from 'framer-motion'
import {
  PlayIcon,
  PauseIcon,
  StopIcon,
  ArrowPathIcon,
  BoltIcon,
} from '@heroicons/react/24/outline'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import toast from 'react-hot-toast'

import { controlSystem, simulateAttack } from '@/api/dashboard'
import { useSocket } from '@/hooks/useSocket'
import { cn } from '@/utils/cn'

export default function SystemControls() {
  const [isSimulating, setIsSimulating] = useState(false)
  const queryClient = useQueryClient()
  const { emit } = useSocket()

  const controlMutation = useMutation({
    mutationFn: controlSystem,
    onSuccess: (_, action) => {
      queryClient.invalidateQueries({ queryKey: ['dashboard'] })
      const messages = {
        start: '🚀 System started successfully',
        stop: '🛑 System stopped',
        pause: '⏸️ System paused',
        reset: '🔄 System reset completed',
      }
      toast.success(messages[action as keyof typeof messages])
    },
    onError: (error: any) => {
      toast.error(error.message || 'System control failed')
    },
  })

  const attackMutation = useMutation({
    mutationFn: ({ type, intensity }: { type: string; intensity: string }) =>
      simulateAttack(type, intensity),
    onSuccess: () => {
      toast.error('⚠️ Attack simulation started')
      setIsSimulating(false)
    },
    onError: (error: any) => {
      toast.error(error.message || 'Attack simulation failed')
      setIsSimulating(false)
    },
  })

  const handleControl = (action: 'start' | 'stop' | 'pause' | 'reset') => {
    controlMutation.mutate(action)
    // Also emit via WebSocket for real-time updates
    emit('system_control', { action })
  }

  const handleAttackSimulation = () => {
    setIsSimulating(true)
    attackMutation.mutate({
      type: 'model_poisoning',
      intensity: 'medium',
    })
  }

  const isLoading = controlMutation.isPending || attackMutation.isPending

  return (
    <div className="flex items-center space-x-2">
      {/* Start Button */}
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => handleControl('start')}
        disabled={isLoading}
        className={cn(
          'flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all duration-200',
          'bg-green-600 hover:bg-green-700 text-white',
          'focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2',
          'disabled:opacity-50 disabled:cursor-not-allowed'
        )}
      >
        <PlayIcon className="h-4 w-4" />
        <span className="hidden sm:inline">Start</span>
      </motion.button>

      {/* Pause Button */}
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => handleControl('pause')}
        disabled={isLoading}
        className={cn(
          'flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all duration-200',
          'bg-yellow-600 hover:bg-yellow-700 text-white',
          'focus:outline-none focus:ring-2 focus:ring-yellow-500 focus:ring-offset-2',
          'disabled:opacity-50 disabled:cursor-not-allowed'
        )}
      >
        <PauseIcon className="h-4 w-4" />
        <span className="hidden sm:inline">Pause</span>
      </motion.button>

      {/* Stop Button */}
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => handleControl('stop')}
        disabled={isLoading}
        className={cn(
          'flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all duration-200',
          'bg-red-600 hover:bg-red-700 text-white',
          'focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2',
          'disabled:opacity-50 disabled:cursor-not-allowed'
        )}
      >
        <StopIcon className="h-4 w-4" />
        <span className="hidden sm:inline">Stop</span>
      </motion.button>

      {/* Reset Button */}
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => handleControl('reset')}
        disabled={isLoading}
        className={cn(
          'flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all duration-200',
          'bg-gray-600 hover:bg-gray-700 text-white',
          'focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2',
          'disabled:opacity-50 disabled:cursor-not-allowed'
        )}
      >
        <ArrowPathIcon className="h-4 w-4" />
        <span className="hidden sm:inline">Reset</span>
      </motion.button>

      {/* Divider */}
      <div className="h-6 w-px bg-gray-300 dark:bg-gray-600" />

      {/* Attack Simulation Button */}
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={handleAttackSimulation}
        disabled={isLoading || isSimulating}
        className={cn(
          'flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all duration-200',
          'bg-orange-600 hover:bg-orange-700 text-white',
          'focus:outline-none focus:ring-2 focus:ring-orange-500 focus:ring-offset-2',
          'disabled:opacity-50 disabled:cursor-not-allowed'
        )}
      >
        <BoltIcon className={cn('h-4 w-4', isSimulating && 'animate-pulse')} />
        <span className="hidden sm:inline">
          {isSimulating ? 'Simulating...' : 'Simulate Attack'}
        </span>
      </motion.button>

      {/* Loading Indicator */}
      {isLoading && (
        <div className="flex items-center space-x-2 text-sm text-gray-500 dark:text-gray-400">
          <div className="animate-spin rounded-full h-4 w-4 border-2 border-gray-300 border-t-primary-600" />
          <span className="hidden sm:inline">Processing...</span>
        </div>
      )}
    </div>
  )
}