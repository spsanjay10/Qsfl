import { motion } from 'framer-motion'
import {
  EyeIcon,
  TrashIcon,
  MapPinIcon,
  ShieldCheckIcon,
  ExclamationTriangleIcon,
  SignalIcon,
  ClockIcon,
} from '@heroicons/react/24/outline'
import { Menu, Transition } from '@headlessui/react'
import { Fragment } from 'react'
import { EllipsisVerticalIcon } from '@heroicons/react/24/solid'

import { cn } from '@/utils/cn'
import { useAuth } from '@/hooks/useAuth'

interface ClientCardProps {
  clientId: string
  client: any
  onView: () => void
  onDelete: () => void
}

export default function ClientCard({ clientId, client, onView, onDelete }: ClientCardProps) {
  const { user } = useAuth()
  const canDelete = user?.permissions.includes('admin') || user?.permissions.includes('manage_clients')

  const getTypeConfig = (type: string) => {
    const configs = {
      honest: {
        color: 'text-green-600 dark:text-green-400',
        bgColor: 'bg-green-100 dark:bg-green-900',
        borderColor: 'border-green-200 dark:border-green-700',
        icon: ShieldCheckIcon,
        label: 'Honest',
      },
      suspicious: {
        color: 'text-yellow-600 dark:text-yellow-400',
        bgColor: 'bg-yellow-100 dark:bg-yellow-900',
        borderColor: 'border-yellow-200 dark:border-yellow-700',
        icon: ExclamationTriangleIcon,
        label: 'Suspicious',
      },
      malicious: {
        color: 'text-red-600 dark:text-red-400',
        bgColor: 'bg-red-100 dark:bg-red-900',
        borderColor: 'border-red-200 dark:border-red-700',
        icon: ExclamationTriangleIcon,
        label: 'Malicious',
      },
    }
    return configs[type as keyof typeof configs] || configs.honest
  }

  const typeConfig = getTypeConfig(client.type)
  const TypeIcon = typeConfig.icon

  const getStatusColor = (status: string, quarantined: boolean) => {
    if (quarantined) return 'text-orange-600 dark:text-orange-400'
    if (status === 'active') return 'text-green-600 dark:text-green-400'
    return 'text-gray-600 dark:text-gray-400'
  }

  const getReputationColor = (reputation: number) => {
    if (reputation >= 0.8) return 'text-green-600 dark:text-green-400'
    if (reputation >= 0.5) return 'text-yellow-600 dark:text-yellow-400'
    return 'text-red-600 dark:text-red-400'
  }

  const formatLastSeen = (lastUpdate: string | null) => {
    if (!lastUpdate) return 'Never'
    const date = new Date(lastUpdate)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    
    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`
    return `${Math.floor(diffMins / 1440)}d ago`
  }

  return (
    <motion.div
      whileHover={{ y: -2 }}
      className={cn(
        'bg-white dark:bg-gray-800 rounded-xl shadow-soft hover:shadow-medium transition-all duration-200 border-l-4 overflow-hidden',
        typeConfig.borderColor,
        client.quarantined && 'ring-2 ring-orange-200 dark:ring-orange-800'
      )}
    >
      <div className="p-6">
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center space-x-3">
            <div className={cn('p-2 rounded-lg', typeConfig.bgColor)}>
              <TypeIcon className={cn('h-5 w-5', typeConfig.color)} />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white truncate">
                {clientId}
              </h3>
              <div className="flex items-center space-x-2">
                <span className={cn('text-sm font-medium', typeConfig.color)}>
                  {typeConfig.label}
                </span>
                {client.quarantined && (
                  <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-orange-100 dark:bg-orange-900 text-orange-800 dark:text-orange-200">
                    Quarantined
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Actions Menu */}
          <Menu as="div" className="relative">
            <Menu.Button className="p-2 rounded-lg text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
              <EllipsisVerticalIcon className="h-5 w-5" />
            </Menu.Button>
            <Transition
              as={Fragment}
              enter="transition ease-out duration-100"
              enterFrom="transform opacity-0 scale-95"
              enterTo="transform opacity-100 scale-100"
              leave="transition ease-in duration-75"
              leaveFrom="transform opacity-100 scale-100"
              leaveTo="transform opacity-0 scale-95"
            >
              <Menu.Items className="absolute right-0 z-10 mt-2 w-48 origin-top-right rounded-lg bg-white dark:bg-gray-700 shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none">
                <div className="py-1">
                  <Menu.Item>
                    {({ active }) => (
                      <button
                        onClick={onView}
                        className={cn(
                          'flex items-center w-full px-4 py-2 text-sm',
                          active ? 'bg-gray-100 dark:bg-gray-600 text-gray-900 dark:text-white' : 'text-gray-700 dark:text-gray-200'
                        )}
                      >
                        <EyeIcon className="h-4 w-4 mr-3" />
                        View Details
                      </button>
                    )}
                  </Menu.Item>
                  {canDelete && (
                    <Menu.Item>
                      {({ active }) => (
                        <button
                          onClick={onDelete}
                          className={cn(
                            'flex items-center w-full px-4 py-2 text-sm',
                            active ? 'bg-red-100 dark:bg-red-900 text-red-900 dark:text-red-100' : 'text-red-700 dark:text-red-400'
                          )}
                        >
                          <TrashIcon className="h-4 w-4 mr-3" />
                          Delete Client
                        </button>
                      )}
                    </Menu.Item>
                  )}
                </div>
              </Menu.Items>
            </Transition>
          </Menu>
        </div>

        {/* Location */}
        {client.location && (
          <div className="flex items-center text-sm text-gray-600 dark:text-gray-400 mb-3">
            <MapPinIcon className="h-4 w-4 mr-2" />
            <span>{client.location.city}, {client.location.country}</span>
          </div>
        )}

        {/* Metrics */}
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Reputation</p>
            <div className="flex items-center space-x-2">
              <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div
                  className={cn(
                    'h-2 rounded-full transition-all duration-300',
                    client.reputation >= 0.8 ? 'bg-green-500' :
                    client.reputation >= 0.5 ? 'bg-yellow-500' : 'bg-red-500'
                  )}
                  style={{ width: `${client.reputation * 100}%` }}
                />
              </div>
              <span className={cn('text-sm font-medium', getReputationColor(client.reputation))}>
                {(client.reputation * 100).toFixed(0)}%
              </span>
            </div>
          </div>

          <div>
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Anomaly Score</p>
            <div className="flex items-center space-x-2">
              <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div
                  className={cn(
                    'h-2 rounded-full transition-all duration-300',
                    client.last_anomaly_score <= 0.3 ? 'bg-green-500' :
                    client.last_anomaly_score <= 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                  )}
                  style={{ width: `${client.last_anomaly_score * 100}%` }}
                />
              </div>
              <span className={cn(
                'text-sm font-medium',
                client.last_anomaly_score <= 0.3 ? 'text-green-600 dark:text-green-400' :
                client.last_anomaly_score <= 0.6 ? 'text-yellow-600 dark:text-yellow-400' : 'text-red-600 dark:text-red-400'
              )}>
                {(client.last_anomaly_score * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <p className="text-lg font-semibold text-gray-900 dark:text-white">
              {client.updates_sent || 0}
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400">Updates</p>
          </div>
          <div>
            <p className="text-lg font-semibold text-gray-900 dark:text-white">
              {client.model_accuracy ? `${(client.model_accuracy * 100).toFixed(1)}%` : 'N/A'}
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400">Accuracy</p>
          </div>
          <div>
            <div className="flex items-center justify-center space-x-1">
              <div className={cn(
                'w-2 h-2 rounded-full',
                client.quarantined ? 'bg-orange-500 animate-pulse' :
                client.status === 'active' ? 'bg-green-500 animate-pulse' : 'bg-gray-400'
              )} />
              <span className={cn(
                'text-xs font-medium',
                getStatusColor(client.status, client.quarantined)
              )}>
                {client.quarantined ? 'Quarantined' : client.status === 'active' ? 'Online' : 'Offline'}
              </span>
            </div>
            <p className="text-xs text-gray-500 dark:text-gray-400">Status</p>
          </div>
        </div>

        {/* Last Seen */}
        <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
            <div className="flex items-center">
              <ClockIcon className="h-3 w-3 mr-1" />
              <span>Last seen: {formatLastSeen(client.last_update)}</span>
            </div>
            <div className="flex items-center">
              <SignalIcon className="h-3 w-3 mr-1" />
              <span>Round {client.last_round || 0}</span>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  )
}