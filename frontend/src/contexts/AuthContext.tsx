import { createContext, useContext, useEffect, useState, ReactNode } from 'react'
import toast from 'react-hot-toast'

interface User {
  id: string
  name: string
  email: string
  role: string
  avatar?: string
  permissions: string[]
}

interface AuthContextType {
  user: User | null
  login: (email: string, password: string, rememberMe?: boolean) => Promise<void>
  logout: () => void
  isLoading: boolean
  isAuthenticated: boolean
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

interface AuthProviderProps {
  children: ReactNode
}

// Demo users database
const DEMO_USERS = [
  {
    id: '1',
    name: 'Dr. Sarah Chen',
    email: 'admin@qsfl-caad.com',
    password: 'admin123',
    role: 'Administrator',
    avatar: 'https://images.unsplash.com/photo-1494790108755-2616b612b786?w=150',
    permissions: ['read', 'write', 'admin', 'manage_clients', 'system_control'],
  },
  {
    id: '2',
    name: 'Alex Rodriguez',
    email: 'security@qsfl-caad.com',
    password: 'security123',
    role: 'Security Analyst',
    avatar: 'https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=150',
    permissions: ['read', 'write', 'security_analysis', 'manage_clients'],
  },
  {
    id: '3',
    name: 'Dr. Emily Watson',
    email: 'researcher@qsfl-caad.com',
    password: 'research123',
    role: 'Data Scientist',
    avatar: 'https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=150',
    permissions: ['read', 'analytics', 'research'],
  },
]

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Check for stored authentication
    const checkAuth = () => {
      try {
        const storedUser = localStorage.getItem('qsfl_user')
        const storedToken = localStorage.getItem('qsfl_token')
        
        if (storedUser && storedToken) {
          const userData = JSON.parse(storedUser)
          // Verify token is still valid (in real app, check with server)
          const tokenData = JSON.parse(atob(storedToken.split('.')[1]))
          const isExpired = tokenData.exp * 1000 < Date.now()
          
          if (!isExpired) {
            setUser(userData)
          } else {
            // Token expired, clear storage
            localStorage.removeItem('qsfl_user')
            localStorage.removeItem('qsfl_token')
          }
        }
      } catch (error) {
        console.error('Auth check error:', error)
        localStorage.removeItem('qsfl_user')
        localStorage.removeItem('qsfl_token')
      } finally {
        setIsLoading(false)
      }
    }

    checkAuth()
  }, [])

  const login = async (email: string, password: string, rememberMe = false) => {
    setIsLoading(true)
    
    try {
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      // Find user in demo database
      const demoUser = DEMO_USERS.find(u => u.email === email && u.password === password)
      
      if (!demoUser) {
        throw new Error('Invalid credentials')
      }

      // Create user object (exclude password)
      const { password: _, ...userWithoutPassword } = demoUser
      const userData: User = userWithoutPassword

      // Create mock JWT token
      const tokenPayload = {
        userId: userData.id,
        email: userData.email,
        role: userData.role,
        exp: Math.floor(Date.now() / 1000) + (rememberMe ? 30 * 24 * 60 * 60 : 24 * 60 * 60), // 30 days or 1 day
      }
      
      const mockToken = `header.${btoa(JSON.stringify(tokenPayload))}.signature`

      // Store authentication data
      localStorage.setItem('qsfl_user', JSON.stringify(userData))
      localStorage.setItem('qsfl_token', mockToken)
      
      setUser(userData)
      
      // Welcome message based on role
      const welcomeMessages = {
        'Administrator': 'Welcome back! You have full system access.',
        'Security Analyst': 'Security dashboard ready. All systems monitored.',
        'Data Scientist': 'Analytics environment loaded. Ready for research.',
      }
      
      toast.success(welcomeMessages[userData.role as keyof typeof welcomeMessages] || 'Welcome back!')
      
    } catch (error) {
      toast.error('Invalid email or password. Please try again.')
      throw error
    } finally {
      setIsLoading(false)
    }
  }

  const logout = () => {
    localStorage.removeItem('qsfl_user')
    localStorage.removeItem('qsfl_token')
    setUser(null)
    toast.success('Logged out successfully')
  }

  const value = {
    user,
    login,
    logout,
    isLoading,
    isAuthenticated: !!user,
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}