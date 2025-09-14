import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  UsersIcon,
  PlusIcon,
  MapPinIcon,
  ShieldCheckIcon,
  ExclamationTriangleIcon,
  ComputerDesktopIcon,
  TrashIcon,
  LockClosedIcon,
  LockOpenIcon,
} from '@heroicons/react/24/outline'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import toast from 'react-hot-toast'

import AddClientModal from '@/components/clients/AddClientModal'
import ClientDetailsModal from '@/components/clients/ClientDetailsModal'
import { deleteClient, toggleClientQuarantine } from '@/api/clients'
import { cn } from '@/utils/cn'

interface ConnectedClientsProps {
  data: any
}

export default function ConnectedClients({ data }: ConnectedClientsProps) {
  const [isAddModalOpen, setIsAddModalOpen] = useState(false)
  const [selectedClient, setSelectedClient] = useState<any>(null)
  const [isDetailsModalOpen, setIsDetailsModalOpen] = useState(false)
  
  const queryClient = useQueryClient()

  const deleteClientMutation = useMutation({
    mutationFn: deleteClient,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['dashboard'] })
      toast.success('Client deleted successfully')
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to delete client')
    },
  })

  const quarantineMutation = useMutation({
    mutationFn: toggleClientQuarantine,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['dashboard'] })
      toast.success('Client status updated')
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to update client')
    },
  })

  const clients = data?.clients || {}
  const clientEntries = Object.entries(clients)

  const getTypeConfig = (type: string) => {
    const configs = {
      honest: {
        color: 'text-green-600 dark:text-green-400',
        bgColor: 'bg-green-100 dark:bg-green-900/20',
        borderColor: 'border-green-200 dark:border-green-700',
        icon: ShieldCheckIcon,
        label: 'Honest',
      },
      suspicious: {
        color: 'text-yellow-600 dark:text-yellow-400',
        bgColor: 'bg-yellow-100 dark:bg-yellow-900/20',
        borderColor: 'border-yellow-200 dark:border-yellow-700',
        icon: ExclamationTriangleIcon,
        label: 'Suspicious',
      },
      malicious: {
        color: 'text-red-600 dark:text-red-400',
        bgColor: 'bg-red-100 dark:bg-red-900/20',
        borderColor: 'border-red-200 dark:border-red-700',
        icon: ExclamationTriangleIcon,
        label: 'Malicious',
      },
    }
    return configs[type as keyof typeof configs] || configs.honest
  }

  const handleClientClick = (clientId: string, client: any) => {
    setSelectedClient({ id: clientId, ...client })
    setIsDetailsModalOpen(true)
  }

  const handleDeleteClient = (clientId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    if (confirm(`Are you sure you want to delete client "${clientId}"?`)) {
      deleteClientMutation.mutate(clientId)
    }
  }

  const handleToggleQuarantine = (clientId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    quarantineMutation.mutate(clientId)
  }

  return (
    <>
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-soft p-6 border border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
            <UsersIcon className="h-5 w-5 mr-2" />
            Connected Clients ({clientEntries.length})
          </h3>
          <button
            onClick={() => setIsAddModalOpen(true)}
            className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-primary-600 to-secondary-600 text-white rounded-lg hover:from-primary-700 hover:to-secondary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 transition-all duration-200"
          >
            <PlusIcon className="h-4 w-4" />
            <span>Add Client</span>
          </button>
        </div>

        {clientEntries.length === 0 ? (
          <div className="text-center py-12">
            <UsersIcon className="h-16 w-16 text-gray-400 mx-auto mb-4" />
            <h4 className="text-lg font-medium text-gray-900 dark:text-white mb-2">No Clients Connected</h4>
            <p className="text-gray-500 dark:text-gray-400 mb-6">
              Add your first client to start monitoring the federated learning network.
            </p>
            <button
              onClick={() => setIsAddModalOpen(true)}
              className="inline-flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-primary-600 to-secondary-600 text-white rounded-lg hover:from-primary-700 hover:to-secondary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 transition-all duration-200"
            >
              <PlusIcon className="h-5 w-5" />
              <span>Add First Client</span>
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {clientEntries.map(([clientId, client]: [string, any]) => {
              const typeConfig = getTypeConfig(client.type)
              const TypeIcon = typeConfig.icon

              return (
                <motion.div
                  key={clientId}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={cn(
                    'relative bg-white dark:bg-gray-700 rounded-lg border-2 p-4 cursor-pointer transition-all duration-200 hover:shadow-md',
                    typeConfig.borderColor,
                    client.quarantined && 'opacity-60'
                  )}
                  onClick={() => handleClientClick(clientId, client)}
                >
                  {/* Status indicator */}
                  <div className="absolute top-3 right-3">
                    <div className={cn(
                      'w-3 h-3 rounded-full',
                      client.quarantined ? 'bg-red-500' :
                      client.status === 'active' ? 'bg-green-500 animate-pulse' : 'bg-gray-400'
                    )} />
                  </div>

                  {/* Client header */}
                  <div className="flex items-start space-x-3 mb-3">
                    <div className={cn('p-2 rounded-lg', typeConfig.bgColor)}>
                      <TypeIcon className={cn('h-5 w-5', typeConfig.color)} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <h4 className="text-sm font-semibold text-gray-900 dark:text-white truncate">
                        {clientId}
                      </h4>
                      <p className={cn('text-xs font-medium', typeConfig.color)}>
                        {typeConfig.label} Client
                      </p>
                    </div>
                  </div>

                  {/* Client metrics */}
                  <div className="space-y-2 mb-4">
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-500 dark:text-gray-400">Reputation</span>
                      <span className="font-medium text-gray-900 dark:text-white">
                        {(client.reputation * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-500 dark:text-gray-400">Anomaly Score</span>
                      <span className={cn(
                        'font-medium',
                        client.last_anomaly_score > 0.6 ? 'text-red-600 dark:text-red-400' :
                        client.last_anomaly_score > 0.3 ? 'text-yellow-600 dark:text-yellow-400' :
                        'text-green-600 dark:text-green-400'
                      )}>
                        {(client.last_anomaly_score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-500 dark:text-gray-400">Updates</span>
                      <span className="font-medium text-gray-900 dark:text-white">
                        {client.updates_sent || 0}
                      </span>
                    </div>
                  </div>

                  {/* Location */}
                  {client.location && (
                    <div className="flex items-center text-xs text-gray-500 dark:text-gray-400 mb-3">
                      <MapPinIcon className="h-3 w-3 mr-1" />
                      <span>{client.location.city}, {client.location.country}</span>
                    </div>
                  )}

                  {/* Capabilities */}
                  {client.capabilities && client.capabilities.length > 0 && (
                    <div className="flex items-center text-xs text-gray-500 dark:text-gray-400 mb-3">
                      <ComputerDesktopIcon className="h-3 w-3 mr-1" />
                      <span>{client.capabilities.length} capabilities</span>
                    </div>
                  )}

                  {/* Status badges */}
                  <div className="flex items-center justify-between">
                    <div className="flex space-x-1">
                      {client.quarantined && (
                        <span className="px-2 py-1 text-xs font-medium bg-red-100 dark:bg-red-900/20 text-red-800 dark:text-red-400 rounded-full">
                          Quarantined
                        </span>
                      )}
                    </div>
                    
                    {/* Action buttons */}
                    <div className="flex space-x-1">
                      <button
                        onClick={(e) => handleToggleQuarantine(clientId, e)}
                        className={cn(
                          'p-1.5 rounded-md transition-colors',
                          client.quarantined 
                            ? 'text-green-600 hover:bg-green-100 dark:hover:bg-green-900/20' 
                            : 'text-yellow-600 hover:bg-yellow-100 dark:hover:bg-yellow-900/20'
                        )}
                        title={client.quarantined ? 'Unquarantine' : 'Quarantine'}
                      >
                        {client.quarantined ? (
                          <LockOpenIcon className="h-4 w-4" />
                        ) : (
                          <LockClosedIcon className="h-4 w-4" />
                        )}
                      </button>
                      <button
                        onClick={(e) => handleDeleteClient(clientId, e)}
                        className="p-1.5 text-red-600 hover:bg-red-100 dark:hover:bg-red-900/20 rounded-md transition-colors"
                        title="Delete Client"
                      >
                        <TrashIcon className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                </motion.div>
              )
            })}
          </div>
        )}
      </div>

      {/* Add Client Modal */}
      <AddClientModal
        isOpen={isAddModalOpen}
        onClose={() => setIsAddModalOpen(false)}
      />

      {/* Client Details Modal */}
      <ClientDetailsModal
        client={selectedClient}
        isOpen={isDetailsModalOpen}
        onClose={() => {
          setIsDetailsModalOpen(false)
          setSelectedClient(null)
        }}
      />
    </>
  )
}