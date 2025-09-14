import { motion } from 'framer-motion'
import { 
  ExclamationTriangleIcon, 
  ShieldCheckIcon, 
  ClockIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline'
import { cn } from '@/utils/cn'

interface SecurityEventsProps {
  data: any
}

export default function SecurityEvents({ data }: SecurityEventsProps) {
  const events = data?.metrics?.security_events || []
  const recentEvents = events.slice(-10).reverse() // Show last 10 events, most recent first

  const getSeverityConfig = (severity: string) => {
    switch (severity) {
      case 'high':
        return {
          color: 'text-red-600 dark:text-red-400',
          bgColor: 'bg-red-50 dark:bg-red-900/20',
          borderColor: 'border-red-200 dark:border-red-800',
          icon: ExclamationTriangleIcon,
        }
      case 'medium':
        return {
          color: 'text-yellow-600 dark:text-yellow-400',
          bgColor: 'bg-yellow-50 dark:bg-yellow-900/20',
          borderColor: 'border-yellow-200 dark:border-yellow-800',
          icon: ExclamationTriangleIcon,
        }
      case 'low':
        return {
          color: 'text-blue-600 dark:text-blue-400',
          bgColor: 'bg-blue-50 dark:bg-blue-900/20',
          borderColor: 'border-blue-200 dark:border-blue-800',
          icon: InformationCircleIcon,
        }
      default:
        return {
          color: 'text-gray-600 dark:text-gray-400',
          bgColor: 'bg-gray-50 dark:bg-gray-900/20',
          borderColor: 'border-gray-200 dark:border-gray-800',
          icon: InformationCircleIcon,
        }
    }
  }

  const formatTime = (timestamp: string) => {
    try {
      const date = new Date(timestamp)
      return date.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      })
    } catch {
      return 'Invalid time'
    }
  }

  const getEventTypeLabel = (eventType: string) => {
    const labels: Record<string, string> = {
      'high_anomaly': 'High Anomaly Detected',
      'quarantine': 'Client Quarantined',
      'reputation_drop': 'Reputation Decreased',
      'attack_detected': 'Attack Detected',
      'system_alert': 'System Alert',
      'client_disconnected': 'Client Disconnected',
      'model_poisoning': 'Model Poisoning Attempt',
    }
    return labels[eventType] || eventType.replace('_', ' ').toUpperCase()
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-soft p-6 border border-gray-200 dark:border-gray-700">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
          <ShieldCheckIcon className="h-5 w-5 mr-2" />
          Security Events
        </h3>
        <div className="flex items-center text-sm text-gray-500 dark:text-gray-400">
          <ClockIcon className="h-4 w-4 mr-1" />
          <span>Last 10 events</span>
        </div>
      </div>

      {recentEvents.length === 0 ? (
        <div className="text-center py-12">
          <ShieldCheckIcon className="h-12 w-12 text-gray-400 mx-auto mb-3" />
          <h4 className="text-lg font-medium text-gray-900 dark:text-white mb-2">All Clear</h4>
          <p className="text-gray-500 dark:text-gray-400">
            No security events detected. System is operating normally.
          </p>
        </div>
      ) : (
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {recentEvents.map((event, index) => {
            const config = getSeverityConfig(event.severity)
            const Icon = config.icon

            return (
              <motion.div
                key={`${event.timestamp}-${index}`}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
                className={cn(
                  'flex items-start space-x-3 p-4 rounded-lg border',
                  config.bgColor,
                  config.borderColor
                )}
              >
                <div className={cn('flex-shrink-0 p-1 rounded-full', config.bgColor)}>
                  <Icon className={cn('h-5 w-5', config.color)} />
                </div>
                
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-1">
                    <h4 className={cn('text-sm font-medium', config.color)}>
                      {getEventTypeLabel(event.event_type)}
                    </h4>
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      {formatTime(event.timestamp)}
                    </span>
                  </div>
                  
                  <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                    {event.description}
                  </p>
                  
                  <div className="flex items-center space-x-4 text-xs text-gray-500 dark:text-gray-400">
                    {event.client_id && (
                      <span className="flex items-center">
                        <span className="font-medium">Client:</span>
                        <span className="ml-1 font-mono">{event.client_id}</span>
                      </span>
                    )}
                    
                    {event.anomaly_score !== undefined && (
                      <span className="flex items-center">
                        <span className="font-medium">Score:</span>
                        <span className={cn(
                          'ml-1 font-mono',
                          event.anomaly_score > 0.7 ? 'text-red-600 dark:text-red-400' :
                          event.anomaly_score > 0.4 ? 'text-yellow-600 dark:text-yellow-400' :
                          'text-green-600 dark:text-green-400'
                        )}>
                          {event.anomaly_score.toFixed(3)}
                        </span>
                      </span>
                    )}
                    
                    <span className={cn(
                      'px-2 py-0.5 rounded-full text-xs font-medium',
                      config.color,
                      config.bgColor
                    )}>
                      {event.severity.toUpperCase()}
                    </span>
                  </div>
                </div>
              </motion.div>
            )
          })}
        </div>
      )}

      {/* Summary Stats */}
      {events.length > 0 && (
        <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-700">
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <p className="text-2xl font-bold text-red-600 dark:text-red-400">
                {events.filter((e: any) => e.severity === 'high').length}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">High Severity</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">
                {events.filter((e: any) => e.severity === 'medium').length}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Medium Severity</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                {events.filter((e: any) => e.severity === 'low').length}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Low Severity</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}