import { Fragment, useState } from 'react'
import { Dialog, Transition, Listbox } from '@headlessui/react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import {
  XMarkIcon,
  CheckIcon,
  ChevronUpDownIcon,
  MapPinIcon,
  UserIcon,
  ShieldCheckIcon,
  ExclamationTriangleIcon,
  ComputerDesktopIcon,
} from '@heroicons/react/24/outline'
import { motion } from 'framer-motion'
import toast from 'react-hot-toast'

import LoadingSpinner from '@/components/ui/LoadingSpinner'
import { addClient } from '@/api/clients'
import { cn } from '@/utils/cn'

const clientSchema = z.object({
  name: z.string().min(1, 'Client name is required').max(50, 'Name too long'),
  type: z.enum(['honest', 'suspicious', 'malicious'], {
    required_error: 'Please select a client type',
  }),
  location: z.object({
    city: z.string().min(1, 'City is required'),
    country: z.string().min(1, 'Country is required'),
    lat: z.number().min(-90).max(90),
    lng: z.number().min(-180).max(180),
  }),
  description: z.string().optional(),
  capabilities: z.array(z.string()).optional(),
})

type ClientFormData = z.infer<typeof clientSchema>

const CLIENT_TYPES = [
  {
    value: 'honest',
    label: 'Honest Client',
    description: 'Trustworthy client that follows protocol',
    icon: ShieldCheckIcon,
    color: 'text-green-600 dark:text-green-400',
    bgColor: 'bg-green-100 dark:bg-green-900',
  },
  {
    value: 'suspicious',
    label: 'Suspicious Client',
    description: 'Client with questionable behavior patterns',
    icon: ExclamationTriangleIcon,
    color: 'text-yellow-600 dark:text-yellow-400',
    bgColor: 'bg-yellow-100 dark:bg-yellow-900',
  },
  {
    value: 'malicious',
    label: 'Malicious Client',
    description: 'Client designed to attack the system',
    icon: ExclamationTriangleIcon,
    color: 'text-red-600 dark:text-red-400',
    bgColor: 'bg-red-100 dark:bg-red-900',
  },
]

const PREDEFINED_LOCATIONS = [
  { city: 'New York', country: 'USA', lat: 40.7128, lng: -74.0060 },
  { city: 'London', country: 'UK', lat: 51.5074, lng: -0.1278 },
  { city: 'Tokyo', country: 'Japan', lat: 35.6762, lng: 139.6503 },
  { city: 'Sydney', country: 'Australia', lat: -33.8688, lng: 151.2093 },
  { city: 'Berlin', country: 'Germany', lat: 52.5200, lng: 13.4050 },
  { city: 'Toronto', country: 'Canada', lat: 43.6532, lng: -79.3832 },
  { city: 'Singapore', country: 'Singapore', lat: 1.3521, lng: 103.8198 },
  { city: 'São Paulo', country: 'Brazil', lat: -23.5505, lng: -46.6333 },
  { city: 'Mumbai', country: 'India', lat: 19.0760, lng: 72.8777 },
  { city: 'Dubai', country: 'UAE', lat: 25.2048, lng: 55.2708 },
]

const CAPABILITIES = [
  'High Performance Computing',
  'GPU Acceleration',
  'Large Dataset Processing',
  'Real-time Analytics',
  'Edge Computing',
  'IoT Integration',
  'Mobile Computing',
  'Cloud Integration',
]

interface AddClientModalProps {
  isOpen: boolean
  onClose: () => void
}

export default function AddClientModal({ isOpen, onClose }: AddClientModalProps) {
  const [selectedType, setSelectedType] = useState(CLIENT_TYPES[0])
  const [selectedLocation, setSelectedLocation] = useState(PREDEFINED_LOCATIONS[0])
  const [selectedCapabilities, setSelectedCapabilities] = useState<string[]>([])
  
  const queryClient = useQueryClient()

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
    reset,
    setValue,
    watch,
  } = useForm<ClientFormData>({
    resolver: zodResolver(clientSchema),
    defaultValues: {
      type: 'honest',
      location: PREDEFINED_LOCATIONS[0],
      capabilities: [],
    },
  })

  const addClientMutation = useMutation({
    mutationFn: addClient,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['clients'] })
      toast.success('Client added successfully!')
      handleClose()
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to add client')
    },
  })

  const onSubmit = async (data: ClientFormData) => {
    try {
      await addClientMutation.mutateAsync({
        ...data,
        type: selectedType.value as any,
        location: selectedLocation,
        capabilities: selectedCapabilities,
      })
    } catch (error) {
      // Error handled by mutation
    }
  }

  const handleClose = () => {
    reset()
    setSelectedType(CLIENT_TYPES[0])
    setSelectedLocation(PREDEFINED_LOCATIONS[0])
    setSelectedCapabilities([])
    onClose()
  }

  const toggleCapability = (capability: string) => {
    setSelectedCapabilities(prev =>
      prev.includes(capability)
        ? prev.filter(c => c !== capability)
        : [...prev, capability]
    )
  }

  return (
    <Transition appear show={isOpen} as={Fragment}>
      <Dialog as="div" className="relative z-50" onClose={handleClose}>
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
              <Dialog.Panel className="w-full max-w-2xl transform overflow-hidden rounded-2xl bg-white dark:bg-gray-800 p-6 text-left align-middle shadow-xl transition-all">
                <div className="flex items-center justify-between mb-6">
                  <Dialog.Title className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
                    <UserIcon className="h-6 w-6 mr-2 text-primary-600" />
                    Add New Client
                  </Dialog.Title>
                  <button
                    onClick={handleClose}
                    className="rounded-lg p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                  >
                    <XMarkIcon className="h-5 w-5" />
                  </button>
                </div>

                <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
                  {/* Client Name */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Client Name
                    </label>
                    <input
                      {...register('name')}
                      type="text"
                      placeholder="e.g., client_001, hospital_ny, bank_london"
                      className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
                    />
                    {errors.name && (
                      <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.name.message}</p>
                    )}
                  </div>

                  {/* Client Type */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Client Type
                    </label>
                    <Listbox value={selectedType} onChange={setSelectedType}>
                      <div className="relative">
                        <Listbox.Button className="relative w-full cursor-pointer rounded-lg bg-white dark:bg-gray-700 py-3 pl-4 pr-10 text-left border border-gray-300 dark:border-gray-600 focus:outline-none focus:ring-2 focus:ring-primary-500">
                          <div className="flex items-center">
                            <div className={cn('flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center', selectedType.bgColor)}>
                              <selectedType.icon className={cn('w-5 h-5', selectedType.color)} />
                            </div>
                            <div className="ml-3">
                              <span className="block truncate text-gray-900 dark:text-white font-medium">
                                {selectedType.label}
                              </span>
                              <span className="block truncate text-sm text-gray-500 dark:text-gray-400">
                                {selectedType.description}
                              </span>
                            </div>
                          </div>
                          <span className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-2">
                            <ChevronUpDownIcon className="h-5 w-5 text-gray-400" />
                          </span>
                        </Listbox.Button>
                        <Transition
                          as={Fragment}
                          leave="transition ease-in duration-100"
                          leaveFrom="opacity-100"
                          leaveTo="opacity-0"
                        >
                          <Listbox.Options className="absolute z-10 mt-1 max-h-60 w-full overflow-auto rounded-lg bg-white dark:bg-gray-700 py-1 shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none">
                            {CLIENT_TYPES.map((type) => (
                              <Listbox.Option
                                key={type.value}
                                className={({ active }) =>
                                  cn(
                                    'relative cursor-pointer select-none py-3 pl-4 pr-10',
                                    active ? 'bg-primary-100 dark:bg-primary-900' : 'text-gray-900 dark:text-white'
                                  )
                                }
                                value={type}
                              >
                                {({ selected }) => (
                                  <div className="flex items-center">
                                    <div className={cn('flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center', type.bgColor)}>
                                      <type.icon className={cn('w-5 h-5', type.color)} />
                                    </div>
                                    <div className="ml-3">
                                      <span className={cn('block truncate font-medium', selected ? 'text-primary-600 dark:text-primary-400' : 'text-gray-900 dark:text-white')}>
                                        {type.label}
                                      </span>
                                      <span className="block truncate text-sm text-gray-500 dark:text-gray-400">
                                        {type.description}
                                      </span>
                                    </div>
                                    {selected && (
                                      <span className="absolute inset-y-0 right-0 flex items-center pr-3 text-primary-600 dark:text-primary-400">
                                        <CheckIcon className="h-5 w-5" />
                                      </span>
                                    )}
                                  </div>
                                )}
                              </Listbox.Option>
                            ))}
                          </Listbox.Options>
                        </Transition>
                      </div>
                    </Listbox>
                  </div>

                  {/* Location */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Location
                    </label>
                    <Listbox value={selectedLocation} onChange={setSelectedLocation}>
                      <div className="relative">
                        <Listbox.Button className="relative w-full cursor-pointer rounded-lg bg-white dark:bg-gray-700 py-3 pl-4 pr-10 text-left border border-gray-300 dark:border-gray-600 focus:outline-none focus:ring-2 focus:ring-primary-500">
                          <div className="flex items-center">
                            <MapPinIcon className="h-5 w-5 text-gray-400 mr-3" />
                            <span className="block truncate text-gray-900 dark:text-white">
                              {selectedLocation.city}, {selectedLocation.country}
                            </span>
                          </div>
                          <span className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-2">
                            <ChevronUpDownIcon className="h-5 w-5 text-gray-400" />
                          </span>
                        </Listbox.Button>
                        <Transition
                          as={Fragment}
                          leave="transition ease-in duration-100"
                          leaveFrom="opacity-100"
                          leaveTo="opacity-0"
                        >
                          <Listbox.Options className="absolute z-10 mt-1 max-h-60 w-full overflow-auto rounded-lg bg-white dark:bg-gray-700 py-1 shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none">
                            {PREDEFINED_LOCATIONS.map((location, index) => (
                              <Listbox.Option
                                key={index}
                                className={({ active }) =>
                                  cn(
                                    'relative cursor-pointer select-none py-2 pl-10 pr-4',
                                    active ? 'bg-primary-100 dark:bg-primary-900 text-primary-900 dark:text-primary-100' : 'text-gray-900 dark:text-white'
                                  )
                                }
                                value={location}
                              >
                                {({ selected }) => (
                                  <>
                                    <span className={cn('block truncate', selected ? 'font-medium' : 'font-normal')}>
                                      {location.city}, {location.country}
                                    </span>
                                    {selected && (
                                      <span className="absolute inset-y-0 left-0 flex items-center pl-3 text-primary-600 dark:text-primary-400">
                                        <CheckIcon className="h-5 w-5" />
                                      </span>
                                    )}
                                  </>
                                )}
                              </Listbox.Option>
                            ))}
                          </Listbox.Options>
                        </Transition>
                      </div>
                    </Listbox>
                  </div>

                  {/* Capabilities */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Capabilities (Optional)
                    </label>
                    <div className="grid grid-cols-2 gap-2">
                      {CAPABILITIES.map((capability) => (
                        <button
                          key={capability}
                          type="button"
                          onClick={() => toggleCapability(capability)}
                          className={cn(
                            'flex items-center px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                            selectedCapabilities.includes(capability)
                              ? 'bg-primary-100 dark:bg-primary-900 text-primary-700 dark:text-primary-300 border-primary-200 dark:border-primary-700'
                              : 'bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-300 border-gray-200 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-600',
                            'border'
                          )}
                        >
                          <ComputerDesktopIcon className="h-4 w-4 mr-2" />
                          {capability}
                          {selectedCapabilities.includes(capability) && (
                            <CheckIcon className="h-4 w-4 ml-auto" />
                          )}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Description */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Description (Optional)
                    </label>
                    <textarea
                      {...register('description')}
                      rows={3}
                      placeholder="Additional information about this client..."
                      className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 resize-none"
                    />
                  </div>

                  {/* Actions */}
                  <div className="flex items-center justify-end space-x-3 pt-6 border-t border-gray-200 dark:border-gray-700">
                    <button
                      type="button"
                      onClick={handleClose}
                      className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 transition-colors"
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      disabled={isSubmitting || addClientMutation.isPending}
                      className="px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-primary-600 to-secondary-600 rounded-lg hover:from-primary-700 hover:to-secondary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center"
                    >
                      {(isSubmitting || addClientMutation.isPending) ? (
                        <>
                          <LoadingSpinner size="sm" />
                          <span className="ml-2">Adding...</span>
                        </>
                      ) : (
                        'Add Client'
                      )}
                    </button>
                  </div>
                </form>
              </Dialog.Panel>
            </Transition.Child>
          </div>
        </div>
      </Dialog>
    </Transition>
  )
}