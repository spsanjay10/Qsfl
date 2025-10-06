import { Fragment } from 'react'
import { Dialog, Transition } from '@headlessui/react'
import {
    XMarkIcon,
    MapPinIcon,
    ShieldCheckIcon,
    ExclamationTriangleIcon,
    ComputerDesktopIcon,
    ClockIcon,
    ChartBarIcon,
} from '@heroicons/react/24/outline'

import { cn } from '@/utils/cn'

interface ClientLocation {
    city: string
    country: string
    lat?: number
    lng?: number
}

interface Client {
    id: string
    type: 'honest' | 'suspicious' | 'malicious'
    status: 'active' | 'offline'
    quarantined: boolean
    reputation: number
    location?: ClientLocation
    updates_sent?: number
    model_accuracy?: number
    last_anomaly_score: number
    capabilities?: string[]
    description?: string
    created_at?: string
    last_update?: string
    last_round?: number
}

interface ClientDetailsModalProps {
    client: Client | null
    isOpen: boolean
    onClose: () => void
}

export default function ClientDetailsModal({ client, isOpen, onClose }: ClientDetailsModalProps) {
    if (!client) return null

    const getTypeConfig = (type: Client['type']) => {
        const configs = {
            honest: {
                color: 'text-green-600 dark:text-green-400',
                bgColor: 'bg-green-100 dark:bg-green-900',
                borderColor: 'border-green-200 dark:border-green-700',
                icon: ShieldCheckIcon,
                label: 'Honest Client',
            },
            suspicious: {
                color: 'text-yellow-600 dark:text-yellow-400',
                bgColor: 'bg-yellow-100 dark:bg-yellow-900',
                borderColor: 'border-yellow-200 dark:border-yellow-700',
                icon: ExclamationTriangleIcon,
                label: 'Suspicious Client',
            },
            malicious: {
                color: 'text-red-600 dark:text-red-400',
                bgColor: 'bg-red-100 dark:bg-red-900',
                borderColor: 'border-red-200 dark:border-red-700',
                icon: ExclamationTriangleIcon,
                label: 'Malicious Client',
            },
        } as const
        return configs[type] || configs.honest
    }

    const typeConfig = getTypeConfig(client.type)
    const TypeIcon = typeConfig.icon

    const formatDate = (dateString: string) => {
        try {
            return new Date(dateString).toLocaleString()
        } catch {
            return 'Unknown'
        }
    }

    return (
        <Transition appear show={isOpen} as={Fragment}>
            <Dialog as="div" className="relative z-50" onClose={onClose}>
                <Transition.Child
                    as={Fragment}
                    enter="ease-out duration-300"
                    enterFrom="opacity-0"
                    enterTo="opacity-100"
                    leave="ease-in duration-200"
                    leaveFrom="opacity-100"
                    leaveTo="opacity-0"
                >
                    <div className="fixed inset-0 bg-black bg-opacity-25 backdrop-blur-sm" />
                </Transition.Child>

                <div className="fixed inset-0 overflow-y-auto">
                    <div className="flex min-h-full items-center justify-center p-4 text-center">
                        <Transition.Child
                            as={Fragment}
                            enter="ease-out duration-300"
                            enterFrom="opacity-0 scale-95"
                            enterTo="opacity-100 scale-100"
                            leave="ease-in duration-200"
                            leaveFrom="opacity-100 scale-100"
                            leaveTo="opacity-0 scale-95"
                        >
                            <Dialog.Panel className="w-full max-w-2xl transform overflow-hidden rounded-2xl bg-white dark:bg-gray-800 text-left align-middle shadow-xl transition-all">
                                {/* Header */}
                                <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
                                    <div className="flex items-center space-x-3">
                                        <div className={cn('p-3 rounded-lg', typeConfig.bgColor)}>
                                            <TypeIcon className={cn('h-6 w-6', typeConfig.color)} />
                                        </div>
                                        <div>
                                            <Dialog.Title className="text-lg font-semibold text-gray-900 dark:text-white">
                                                {client.id}
                                            </Dialog.Title>
                                            <p className={cn('text-sm font-medium', typeConfig.color)}>
                                                {typeConfig.label}
                                            </p>
                                        </div>
                                    </div>
                                    <button
                                        onClick={onClose}
                                        className="rounded-lg p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                                    >
                                        <XMarkIcon className="h-5 w-5" />
                                    </button>
                                </div>

                                <div className="p-6 space-y-6">
                                    {/* Status Overview */}
                                    <div className="grid grid-cols-2 gap-4">
                                        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                                            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Status</h4>
                                            <div className="flex items-center space-x-2">
                                                <div className={cn(
                                                    'w-3 h-3 rounded-full',
                                                    client.quarantined ? 'bg-red-500' :
                                                        client.status === 'active' ? 'bg-green-500 animate-pulse' : 'bg-gray-400'
                                                )} />
                                                <span className="text-sm font-medium text-gray-900 dark:text-white">
                                                    {client.quarantined ? 'Quarantined' : client.status === 'active' ? 'Active' : 'Offline'}
                                                </span>
                                            </div>
                                        </div>

                                        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                                            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Reputation</h4>
                                            <div className="flex items-center space-x-2">
                                                <div className="flex-1 bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                                                    <div
                                                        className={cn(
                                                            'h-2 rounded-full transition-all duration-300',
                                                            client.reputation >= 0.8 ? 'bg-green-500' :
                                                                client.reputation >= 0.5 ? 'bg-yellow-500' : 'bg-red-500'
                                                        )}
                                                        style={{ width: `${client.reputation * 100}%` }}
                                                    />
                                                </div>
                                                <span className="text-sm font-medium text-gray-900 dark:text-white">
                                                    {(client.reputation * 100).toFixed(1)}%
                                                </span>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Location */}
                                    {client.location && (
                                        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                                            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3 flex items-center">
                                                <MapPinIcon className="h-4 w-4 mr-2" />
                                                Location
                                            </h4>
                                            <p className="text-gray-900 dark:text-white">
                                                {client.location.city}, {client.location.country}
                                            </p>
                                            {client.location.lat !== undefined && client.location.lng !== undefined && (
                                                <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                                                    Coordinates: {client.location.lat.toFixed(4)}, {client.location.lng.toFixed(4)}
                                                </p>
                                            )}
                                        </div>
                                    )}

                                    {/* Performance Metrics */}
                                    <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                                        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3 flex items-center">
                                            <ChartBarIcon className="h-4 w-4 mr-2" />
                                            Performance Metrics
                                        </h4>
                                        <div className="grid grid-cols-3 gap-4">
                                            <div className="text-center">
                                                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                                                    {client.updates_sent || 0}
                                                </p>
                                                <p className="text-xs text-gray-500 dark:text-gray-400">Updates Sent</p>
                                            </div>
                                            <div className="text-center">
                                                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                                                    {client.model_accuracy ? `${(client.model_accuracy * 100).toFixed(1)}%` : 'N/A'}
                                                </p>
                                                <p className="text-xs text-gray-500 dark:text-gray-400">Model Accuracy</p>
                                            </div>
                                            <div className="text-center">
                                                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                                                    {(client.last_anomaly_score * 100).toFixed(1)}%
                                                </p>
                                                <p className="text-xs text-gray-500 dark:text-gray-400">Anomaly Score</p>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Capabilities */}
                                    {client.capabilities && client.capabilities.length > 0 && (
                                        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                                            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3 flex items-center">
                                                <ComputerDesktopIcon className="h-4 w-4 mr-2" />
                                                Capabilities
                                            </h4>
                                            <div className="flex flex-wrap gap-2">
                                                {client.capabilities.map((capability: string, index: number) => (
                                                    <span
                                                        key={index}
                                                        className="px-3 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 text-sm rounded-full"
                                                    >
                                                        {capability}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {/* Description */}
                                    {client.description && (
                                        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                                            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                                Description
                                            </h4>
                                            <p className="text-gray-900 dark:text-white">{client.description}</p>
                                        </div>
                                    )}

                                    {/* Timestamps */}
                                    <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                                        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3 flex items-center">
                                            <ClockIcon className="h-4 w-4 mr-2" />
                                            Timeline
                                        </h4>
                                        <div className="space-y-2 text-sm">
                                            {client.created_at && (
                                                <div className="flex justify-between">
                                                    <span className="text-gray-500 dark:text-gray-400">Created:</span>
                                                    <span className="text-gray-900 dark:text-white">{formatDate(client.created_at)}</span>
                                                </div>
                                            )}
                                            {client.last_update && (
                                                <div className="flex justify-between">
                                                    <span className="text-gray-500 dark:text-gray-400">Last Update:</span>
                                                    <span className="text-gray-900 dark:text-white">{formatDate(client.last_update)}</span>
                                                </div>
                                            )}
                                            {client.last_round && (
                                                <div className="flex justify-between">
                                                    <span className="text-gray-500 dark:text-gray-400">Last Round:</span>
                                                    <span className="text-gray-900 dark:text-white">#{client.last_round}</span>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>

                                {/* Footer */}
                                <div className="flex items-center justify-end space-x-3 p-6 border-t border-gray-200 dark:border-gray-700">
                                    <button
                                        onClick={onClose}
                                        className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 transition-colors"
                                    >
                                        Close
                                    </button>
                                </div>
                            </Dialog.Panel>
                        </Transition.Child>
                    </div>
                </div>
            </Dialog>
        </Transition>
    )
}