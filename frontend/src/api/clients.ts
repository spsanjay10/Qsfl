import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:5000',
  timeout: 10000,
})

// Add auth token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('qsfl_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

export interface Client {
  id: string
  name: string
  type: 'honest' | 'suspicious' | 'malicious'
  status: 'active' | 'offline' | 'quarantined'
  reputation: number
  last_anomaly_score: number
  updates_sent: number
  quarantined: boolean
  location: {
    city: string
    country: string
    lat: number
    lng: number
  }
  capabilities?: string[]
  description?: string
  created_at: string
  last_update?: string
  model_accuracy?: number
}

export interface AddClientRequest {
  name: string
  type: 'honest' | 'suspicious' | 'malicious'
  location: {
    city: string
    country: string
    lat: number
    lng: number
  }
  capabilities?: string[]
  description?: string
}

// Get all clients
export async function getClients(): Promise<{ clients: Record<string, Client> }> {
  try {
    const response = await api.get('/api/clients')
    return { clients: response.data }
  } catch (error) {
    // Fallback to dashboard data if clients endpoint doesn't exist
    try {
      const response = await api.get('/api/dashboard_data')
      return { clients: response.data.clients || {} }
    } catch (fallbackError) {
      console.error('Failed to fetch clients:', fallbackError)
      throw new Error('Failed to fetch clients')
    }
  }
}

// Get single client
export async function getClient(clientId: string): Promise<Client> {
  try {
    const response = await api.get(`/api/clients/${clientId}`)
    return response.data
  } catch (error) {
    console.error('Failed to fetch client:', error)
    throw new Error('Failed to fetch client details')
  }
}

// Add new client
export async function addClient(clientData: AddClientRequest): Promise<Client> {
  try {
    const response = await api.post('/api/clients', clientData)
    return response.data.client
  } catch (error: any) {
    console.error('Failed to add client:', error)
    const message = error.response?.data?.error || 'Failed to add client'
    throw new Error(message)
  }
}

// Update client
export async function updateClient(clientId: string, updates: Partial<Client>): Promise<Client> {
  try {
    const response = await api.put(`/api/clients/${clientId}`, updates)
    return response.data
  } catch (error) {
    console.error('Failed to update client:', error)
    throw new Error('Failed to update client')
  }
}

// Delete client
export async function deleteClient(clientId: string): Promise<void> {
  try {
    await api.delete(`/api/clients/${clientId}`)
  } catch (error: any) {
    console.error('Failed to delete client:', error)
    const message = error.response?.data?.error || 'Failed to delete client'
    throw new Error(message)
  }
}

// Quarantine/unquarantine client
export async function toggleClientQuarantine(clientId: string): Promise<void> {
  try {
    await api.post(`/api/clients/${clientId}/quarantine`)
  } catch (error: any) {
    console.error('Failed to toggle client quarantine:', error)
    const message = error.response?.data?.error || 'Failed to update client status'
    throw new Error(message)
  }
}

// Reset client reputation
export async function resetClientReputation(clientId: string): Promise<void> {
  try {
    await api.post(`/api/clients/${clientId}/reset-reputation`)
  } catch (error) {
    console.error('Failed to reset client reputation:', error)
    throw new Error('Failed to reset client reputation')
  }
}