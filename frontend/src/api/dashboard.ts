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

export interface DashboardData {
  clients: Record<string, any>
  metrics: {
    timestamps: string[]
    anomaly_scores: Record<string, number[]>
    reputation_scores: Record<string, number[]>
    model_accuracy: number[]
    security_events: any[]
    system_performance: any[]
  }
  system_status: string
  current_round: number
}

// Get dashboard data
export async function getDashboardData(): Promise<DashboardData> {
  try {
    const response = await api.get('/api/dashboard_data')
    
    // Merge with demo clients from localStorage
    const demoClients = JSON.parse(localStorage.getItem('demo_clients') || '{}')
    const mergedClients = { ...response.data.clients, ...demoClients }
    
    return {
      ...response.data,
      clients: mergedClients,
    }
  } catch (error) {
    console.error('Failed to fetch dashboard data:', error)
    
    // Return demo data if API is not available
    const demoClients = JSON.parse(localStorage.getItem('demo_clients') || '{}')
    
    return {
      clients: demoClients,
      metrics: {
        timestamps: [],
        anomaly_scores: {},
        reputation_scores: {},
        model_accuracy: [],
        security_events: [],
        system_performance: [],
      },
      system_status: 'stopped',
      current_round: 0,
    }
  }
}

// System control
export async function controlSystem(action: 'start' | 'stop' | 'pause' | 'reset'): Promise<void> {
  try {
    await api.post(`/api/control/${action}`)
  } catch (error) {
    console.error(`Failed to ${action} system:`, error)
    throw new Error(`Failed to ${action} system`)
  }
}

// Simulate attack
export async function simulateAttack(attackType: string, intensity: string): Promise<void> {
  try {
    await api.post('/api/simulate_attack', {
      type: attackType,
      intensity: intensity,
    })
  } catch (error) {
    console.error('Failed to simulate attack:', error)
    throw new Error('Failed to simulate attack')
  }
}