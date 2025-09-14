import { useMemo } from 'react'
import { MapPinIcon, UsersIcon } from '@heroicons/react/24/outline'

interface ClientMapProps {
  data: any
}

export default function ClientMap({ data }: ClientMapProps) {
  const clientStats = useMemo(() => {
    if (!data?.clients) return { total: 0, active: 0, quarantined: 0, locations: [] }

    const clients = Object.entries(data.clients)
    const locations = clients.map(([id, client]: [string, any]) => ({
      id,
      ...client,
      location: client.location || { city: 'Unknown', country: 'Unknown' }
    }))

    return {
      total: clients.length,
      active: clients.filter(([, c]: [string, any]) => c.status === 'active' && !c.quarantined).length,
      quarantined: clients.filter(([, c]: [string, any]) => c.quarantined).length,
      locations
    }
  }, [data])

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'honest': return 'text-green-600 bg-green-100 border-green-200'
      case 'suspicious': return 'text-yellow-600 bg-yellow-100 border-yellow-200'
      case 'malicious': return 'text-red-600 bg-red-100 border-red-200'
      default: return 'text-gray-600 bg-gray-100 border-gray-200'
    }
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-soft p-6 border border-gray-200 dark:border-gray-700">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
          <MapPinIcon className="h-5 w-5 mr-2" />
          Client Distribution
        </h3>
        <div className="flex items-center space-x-4 text-sm">
          <div className="flex items-center">
            <div className="w-3 h-3 bg-green-500 rounded-full mr-1"></div>
            <span className="text-gray-600 dark:text-gray-400">Active: {clientStats.active}</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-red-500 rounded-full mr-1"></div>
            <span className="text-gray-600 dark:text-gray-400">Quarantined: {clientStats.quarantined}</span>
          </div>
        </div>
      </div>

      {/* World Map Placeholder */}
      <div className="relative bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-700 dark:to-gray-600 rounded-lg p-6 mb-4 min-h-48">
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <div className="w-16 h-16 bg-blue-500 rounded-full flex items-center justify-center mx-auto mb-3">
              <MapPinIcon className="h-8 w-8 text-white" />
            </div>
            <p className="text-gray-600 dark:text-gray-300 font-medium">Interactive World Map</p>
            <p className="text-sm text-gray-500 dark:text-gray-400">Showing {clientStats.total} clients globally</p>
          </div>
        </div>
        
        {/* Sample location pins */}
        <div className="absolute top-4 left-8">
          <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
        </div>
        <div className="absolute top-8 right-12">
          <div className="w-3 h-3 bg-yellow-500 rounded-full animate-pulse"></div>
        </div>
        <div className="absolute bottom-8 left-16">
          <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
        </div>
        <div className="absolute bottom-12 right-8">
          <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
        </div>
      </div>

      {/* Client List */}
      <div className="space-y-2 max-h-64 overflow-y-auto">
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3 flex items-center">
          <UsersIcon className="h-4 w-4 mr-2" />
          Connected Clients
        </h4>
        
        {clientStats.locations.length === 0 ? (
          <div className="text-center py-8">
            <UsersIcon className="h-12 w-12 text-gray-400 mx-auto mb-3" />
            <p className="text-gray-500 dark:text-gray-400">No clients connected</p>
          </div>
        ) : (
          clientStats.locations.map((client) => (
            <div
              key={client.id}
              className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg border border-gray-200 dark:border-gray-600"
            >
              <div className="flex items-center space-x-3">
                <div className={`w-2 h-2 rounded-full ${
                  client.quarantined ? 'bg-red-500' : 
                  client.status === 'active' ? 'bg-green-500' : 'bg-gray-400'
                } ${client.status === 'active' ? 'animate-pulse' : ''}`}></div>
                <div>
                  <p className="text-sm font-medium text-gray-900 dark:text-white">{client.id}</p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {client.location.city}, {client.location.country}
                  </p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <span className={`px-2 py-1 text-xs font-medium rounded-full border ${
                  getTypeColor(client.type)
                }`}>
                  {client.type}
                </span>
                {client.quarantined && (
                  <span className="px-2 py-1 text-xs font-medium bg-red-100 text-red-800 rounded-full border border-red-200">
                    Quarantined
                  </span>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}