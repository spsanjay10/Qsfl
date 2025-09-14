# 🎨 QSFL-CAAD Frontend Development Guide

This guide covers everything you need to know about developing the modern React frontend for QSFL-CAAD.

## 🚀 **Quick Start**

### **1. Setup Frontend Environment**
```bash
# Install frontend dependencies
make frontend-install
# or
cd frontend && npm install

# Start development server
make frontend-dev
# or
cd frontend && npm run dev
```

### **2. Start Full Stack Development**
```bash
# Start both backend and frontend
make dev-full

# Access applications:
# Backend API: http://localhost:5000
# Frontend App: http://localhost:3000
```

## 🏗️ **Architecture Overview**

### **Technology Stack**
- **React 18** - Modern React with concurrent features
- **TypeScript** - Type safety and better DX
- **Vite** - Fast build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **Framer Motion** - Smooth animations
- **React Query** - Server state management
- **Socket.IO** - Real-time communication
- **Plotly.js** - Interactive charts and visualizations

### **Project Structure**
```
frontend/
├── src/
│   ├── components/          # Reusable UI components
│   │   ├── ui/             # Basic UI components (buttons, cards, etc.)
│   │   ├── charts/         # Chart components (Plotly, custom)
│   │   └── Layout/         # Layout components (sidebar, header)
│   ├── pages/              # Page components (Dashboard, Clients, etc.)
│   ├── hooks/              # Custom React hooks
│   ├── contexts/           # React contexts (Theme, Socket, Auth)
│   ├── api/                # API functions and types
│   ├── utils/              # Utility functions
│   ├── types/              # TypeScript type definitions
│   └── assets/             # Static assets (images, icons)
├── public/                 # Public static files
├── tests/                  # Test files
└── stories/                # Storybook stories
```

## 🎨 **Design System**

### **Color Palette**
```css
/* Primary Colors */
primary-50: #f0f9ff    primary-500: #0ea5e9    primary-900: #0c4a6e
secondary-50: #fdf4ff  secondary-500: #d946ef  secondary-900: #701a75

/* Status Colors */
success: #22c55e       warning: #f59e0b       danger: #ef4444
```

### **Typography**
- **Font Family**: Inter (primary), JetBrains Mono (code)
- **Font Weights**: 300, 400, 500, 600, 700
- **Responsive scaling** with Tailwind's text utilities

### **Spacing & Layout**
- **Grid System**: CSS Grid and Flexbox
- **Responsive Breakpoints**: sm (640px), md (768px), lg (1024px), xl (1280px)
- **Consistent spacing** using Tailwind's spacing scale

## 🧩 **Component Development**

### **Component Structure**
```tsx
// components/ui/MetricCard.tsx
import { motion } from 'framer-motion'
import { cn } from '@/utils/cn'

interface MetricCardProps {
  title: string
  value: string | number
  change?: string
  changeType?: 'positive' | 'negative' | 'neutral'
  icon?: React.ComponentType<{ className?: string }>
  loading?: boolean
}

export default function MetricCard({
  title,
  value,
  change,
  changeType = 'neutral',
  icon: Icon,
  loading = false,
}: MetricCardProps) {
  if (loading) {
    return <MetricCardSkeleton />
  }

  return (
    <motion.div
      whileHover={{ y: -2 }}
      className="bg-white dark:bg-gray-800 rounded-lg shadow-soft p-6"
    >
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-600 dark:text-gray-400">{title}</p>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">
            {typeof value === 'number' ? value.toLocaleString() : value}
          </p>
        </div>
        {Icon && (
          <div className="p-3 bg-gradient-to-br from-primary-500 to-secondary-500 rounded-lg">
            <Icon className="h-6 w-6 text-white" />
          </div>
        )}
      </div>
      
      {change && (
        <div className={cn('mt-2 text-sm', getChangeColor(changeType))}>
          {change}
        </div>
      )}
    </motion.div>
  )
}
```

### **Component Guidelines**
1. **TypeScript interfaces** for all props
2. **Default props** using destructuring
3. **Loading states** for async components
4. **Error boundaries** for error handling
5. **Accessibility** with proper ARIA labels
6. **Responsive design** with Tailwind classes

### **Storybook Stories**
```tsx
// stories/MetricCard.stories.tsx
import type { Meta, StoryObj } from '@storybook/react'
import { UsersIcon } from '@heroicons/react/24/outline'
import MetricCard from '@/components/ui/MetricCard'

const meta: Meta<typeof MetricCard> = {
  title: 'UI/MetricCard',
  component: MetricCard,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    title: 'Active Clients',
    value: 42,
    change: '+5%',
    changeType: 'positive',
    icon: UsersIcon,
  },
}

export const Loading: Story = {
  args: {
    title: 'Active Clients',
    value: 0,
    loading: true,
  },
}
```

## 📊 **Data Management**

### **React Query for Server State**
```tsx
// hooks/useDashboardData.ts
import { useQuery } from '@tanstack/react-query'
import { getDashboardData } from '@/api/dashboard'

export function useDashboardData() {
  return useQuery({
    queryKey: ['dashboard'],
    queryFn: getDashboardData,
    refetchInterval: 5000, // Refetch every 5 seconds
    staleTime: 1000 * 60, // Consider data stale after 1 minute
    retry: (failureCount, error) => {
      if (error?.status === 404) return false
      return failureCount < 3
    },
  })
}
```

### **Socket.IO for Real-time Updates**
```tsx
// hooks/useSocket.ts
import { useContext, useEffect, useState } from 'react'
import { SocketContext } from '@/contexts/SocketContext'

export function useSocket(event?: string) {
  const { socket, isConnected } = useContext(SocketContext)
  const [data, setData] = useState<any>(null)

  useEffect(() => {
    if (!socket || !event) return

    const handleData = (newData: any) => {
      setData(newData)
    }

    socket.on(event, handleData)
    return () => socket.off(event, handleData)
  }, [socket, event])

  const emit = (eventName: string, data?: any) => {
    socket?.emit(eventName, data)
  }

  return { socket, isConnected, data, emit }
}
```

### **Form Handling with React Hook Form**
```tsx
// components/forms/ClientForm.tsx
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'

const clientSchema = z.object({
  name: z.string().min(1, 'Name is required'),
  type: z.enum(['honest', 'malicious', 'suspicious']),
  location: z.string().optional(),
})

type ClientFormData = z.infer<typeof clientSchema>

export default function ClientForm({ onSubmit }: { onSubmit: (data: ClientFormData) => void }) {
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<ClientFormData>({
    resolver: zodResolver(clientSchema),
  })

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Client Name
        </label>
        <input
          {...register('name')}
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
        />
        {errors.name && (
          <p className="mt-1 text-sm text-red-600">{errors.name.message}</p>
        )}
      </div>
      
      <button
        type="submit"
        disabled={isSubmitting}
        className="btn-primary"
      >
        {isSubmitting ? 'Creating...' : 'Create Client'}
      </button>
    </form>
  )
}
```

## 📈 **Charts & Visualizations**

### **Plotly.js Integration**
```tsx
// components/charts/AnomalyChart.tsx
import Plot from 'react-plotly.js'
import { useMemo } from 'react'

interface AnomalyChartProps {
  data: any
}

export default function AnomalyChart({ data }: AnomalyChartProps) {
  const plotData = useMemo(() => {
    if (!data?.metrics?.anomaly_scores) return []

    const traces = []
    const timestamps = data.metrics.timestamps || []

    // Threshold line
    traces.push({
      x: timestamps,
      y: Array(timestamps.length).fill(0.6),
      name: 'Threshold',
      type: 'scatter',
      mode: 'lines',
      line: { color: '#ef4444', dash: 'dash' },
    })

    // Client traces
    Object.entries(data.metrics.anomaly_scores).forEach(([clientId, scores]: [string, any]) => {
      if (scores.length > 0) {
        const client = data.clients[clientId] || {}
        const color = getClientColor(client.type)

        traces.push({
          x: timestamps.slice(-scores.length),
          y: scores,
          name: clientId,
          type: 'scatter',
          mode: 'lines+markers',
          line: { color },
          opacity: client.quarantined ? 0.5 : 1.0,
        })
      }
    })

    return traces
  }, [data])

  const layout = {
    title: 'Real-time Anomaly Detection',
    xaxis: { title: 'Time' },
    yaxis: { title: 'Anomaly Score', range: [0, 1] },
    showlegend: true,
    margin: { t: 50, r: 50, b: 50, l: 60 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-soft p-6">
      <Plot
        data={plotData}
        layout={layout}
        config={{
          responsive: true,
          displayModeBar: true,
          displaylogo: false,
        }}
        className="w-full h-96"
      />
    </div>
  )
}
```

### **Custom Chart Components**
```tsx
// components/charts/MetricChart.tsx
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

interface MetricChartProps {
  data: Array<{ time: string; value: number }>
  color?: string
  title?: string
}

export default function MetricChart({ data, color = '#0ea5e9', title }: MetricChartProps) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-soft p-6">
      {title && (
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          {title}
        </h3>
      )}
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="time" stroke="#6b7280" />
          <YAxis stroke="#6b7280" />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1f2937',
              border: 'none',
              borderRadius: '8px',
              color: '#f9fafb',
            }}
          />
          <Line
            type="monotone"
            dataKey="value"
            stroke={color}
            strokeWidth={2}
            dot={{ fill: color, strokeWidth: 2, r: 4 }}
            activeDot={{ r: 6 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
```

## 🎭 **Animations with Framer Motion**

### **Page Transitions**
```tsx
// components/PageTransition.tsx
import { motion } from 'framer-motion'

const pageVariants = {
  initial: { opacity: 0, y: 20 },
  in: { opacity: 1, y: 0 },
  out: { opacity: 0, y: -20 },
}

const pageTransition = {
  type: 'tween',
  ease: 'anticipate',
  duration: 0.5,
}

export default function PageTransition({ children }: { children: React.ReactNode }) {
  return (
    <motion.div
      initial="initial"
      animate="in"
      exit="out"
      variants={pageVariants}
      transition={pageTransition}
    >
      {children}
    </motion.div>
  )
}
```

### **List Animations**
```tsx
// components/AnimatedList.tsx
import { motion, AnimatePresence } from 'framer-motion'

const listVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
    },
  },
}

const itemVariants = {
  hidden: { y: 20, opacity: 0 },
  visible: {
    y: 0,
    opacity: 1,
    transition: {
      type: 'spring',
      stiffness: 100,
    },
  },
}

export default function AnimatedList({ items }: { items: any[] }) {
  return (
    <motion.div variants={listVariants} initial="hidden" animate="visible">
      <AnimatePresence>
        {items.map((item, index) => (
          <motion.div
            key={item.id}
            variants={itemVariants}
            layout
            exit={{ opacity: 0, scale: 0.8 }}
          >
            {/* Item content */}
          </motion.div>
        ))}
      </AnimatePresence>
    </motion.div>
  )
}
```

## 🧪 **Testing Strategy**

### **Unit Testing with Vitest**
```tsx
// tests/components/MetricCard.test.tsx
import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { UsersIcon } from '@heroicons/react/24/outline'
import MetricCard from '@/components/ui/MetricCard'

describe('MetricCard', () => {
  it('renders metric data correctly', () => {
    render(
      <MetricCard
        title="Active Clients"
        value={42}
        change="+5%"
        changeType="positive"
        icon={UsersIcon}
      />
    )

    expect(screen.getByText('Active Clients')).toBeInTheDocument()
    expect(screen.getByText('42')).toBeInTheDocument()
    expect(screen.getByText('+5%')).toBeInTheDocument()
  })

  it('shows loading state', () => {
    render(<MetricCard title="Test" value={0} loading />)
    
    expect(screen.getByTestId('metric-card-skeleton')).toBeInTheDocument()
  })
})
```

### **Integration Testing**
```tsx
// tests/pages/Dashboard.test.tsx
import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { describe, it, expect, vi } from 'vitest'
import Dashboard from '@/pages/Dashboard'

// Mock API
vi.mock('@/api/dashboard', () => ({
  getDashboardData: vi.fn(() => Promise.resolve({
    clients: {},
    metrics: {},
    system_status: 'running',
    current_round: 5,
  })),
}))

describe('Dashboard Page', () => {
  it('renders dashboard with data', async () => {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })

    render(
      <QueryClientProvider client={queryClient}>
        <Dashboard />
      </QueryClientProvider>
    )

    await waitFor(() => {
      expect(screen.getByText('Dashboard')).toBeInTheDocument()
      expect(screen.getByText('RUNNING')).toBeInTheDocument()
    })
  })
})
```

### **E2E Testing with Playwright**
```typescript
// tests/e2e/dashboard.spec.ts
import { test, expect } from '@playwright/test'

test('dashboard loads and displays metrics', async ({ page }) => {
  await page.goto('/')
  
  // Check page title
  await expect(page).toHaveTitle(/QSFL-CAAD Dashboard/)
  
  // Check navigation
  await expect(page.locator('nav')).toContainText('Dashboard')
  await expect(page.locator('nav')).toContainText('Clients')
  
  // Check metrics cards
  await expect(page.locator('[data-testid="metric-card"]')).toHaveCount(4)
  
  // Check charts are rendered
  await expect(page.locator('.plotly')).toBeVisible()
})

test('real-time updates work', async ({ page }) => {
  await page.goto('/')
  
  // Wait for WebSocket connection
  await page.waitForFunction(() => window.io && window.io.connected)
  
  // Trigger system start
  await page.click('button:has-text("Start")')
  
  // Check status updates
  await expect(page.locator('text=RUNNING')).toBeVisible()
})
```

## 🚀 **Performance Optimization**

### **Code Splitting**
```tsx
// Lazy load pages
import { lazy, Suspense } from 'react'
import LoadingSpinner from '@/components/ui/LoadingSpinner'

const Dashboard = lazy(() => import('@/pages/Dashboard'))
const Clients = lazy(() => import('@/pages/Clients'))
const Analytics = lazy(() => import('@/pages/Analytics'))

function App() {
  return (
    <Suspense fallback={<LoadingSpinner />}>
      <Routes>
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/clients" element={<Clients />} />
        <Route path="/analytics" element={<Analytics />} />
      </Routes>
    </Suspense>
  )
}
```

### **Memoization**
```tsx
// Memoize expensive calculations
import { useMemo } from 'react'

function ExpensiveComponent({ data }: { data: any[] }) {
  const processedData = useMemo(() => {
    return data
      .filter(item => item.active)
      .map(item => ({
        ...item,
        computed: expensiveCalculation(item),
      }))
      .sort((a, b) => b.computed - a.computed)
  }, [data])

  return <div>{/* Render processed data */}</div>
}
```

### **Virtual Scrolling**
```tsx
// For large lists
import { FixedSizeList as List } from 'react-window'

function VirtualizedList({ items }: { items: any[] }) {
  const Row = ({ index, style }: { index: number; style: React.CSSProperties }) => (
    <div style={style}>
      <ClientCard client={items[index]} />
    </div>
  )

  return (
    <List
      height={600}
      itemCount={items.length}
      itemSize={80}
      width="100%"
    >
      {Row}
    </List>
  )
}
```

## 🔧 **Development Workflow**

### **Daily Development**
```bash
# 1. Start development environment
make dev-full

# 2. Make changes to components
# Edit files in src/

# 3. Run tests
make frontend-test

# 4. Check code quality
make frontend-lint
make frontend-format

# 5. Build for production
make frontend-build
```

### **Component Development Workflow**
```bash
# 1. Create component
mkdir src/components/ui/NewComponent
touch src/components/ui/NewComponent/index.tsx

# 2. Write Storybook story
touch src/components/ui/NewComponent/NewComponent.stories.tsx

# 3. Write tests
touch src/components/ui/NewComponent/NewComponent.test.tsx

# 4. Start Storybook for development
make frontend-storybook

# 5. Run tests
npm run test NewComponent
```

### **Git Workflow**
```bash
# 1. Create feature branch
git checkout -b feature/new-dashboard-widget

# 2. Make changes and commit
git add .
git commit -m "feat(dashboard): add new performance widget"

# 3. Run pre-commit checks
make frontend-lint
make frontend-test

# 4. Push and create PR
git push origin feature/new-dashboard-widget
```

## 📱 **Responsive Design**

### **Breakpoint Strategy**
```tsx
// Use Tailwind responsive classes
<div className="
  grid 
  grid-cols-1 
  sm:grid-cols-2 
  lg:grid-cols-3 
  xl:grid-cols-4 
  gap-6
">
  {/* Cards */}
</div>

// Custom breakpoints with hooks
import { useMediaQuery } from '@/hooks/useMediaQuery'

function ResponsiveComponent() {
  const isMobile = useMediaQuery('(max-width: 768px)')
  const isTablet = useMediaQuery('(max-width: 1024px)')
  
  return (
    <div>
      {isMobile ? <MobileLayout /> : <DesktopLayout />}
    </div>
  )
}
```

### **Mobile-First Approach**
```css
/* Base styles for mobile */
.card {
  @apply p-4 rounded-lg;
}

/* Tablet and up */
@screen md {
  .card {
    @apply p-6;
  }
}

/* Desktop and up */
@screen lg {
  .card {
    @apply p-8;
  }
}
```

## 🔒 **Security Best Practices**

### **Input Validation**
```tsx
// Always validate user inputs
import { z } from 'zod'

const userInputSchema = z.object({
  message: z.string().max(1000).regex(/^[a-zA-Z0-9\s]*$/),
  email: z.string().email(),
})

function validateInput(input: unknown) {
  try {
    return userInputSchema.parse(input)
  } catch (error) {
    throw new Error('Invalid input')
  }
}
```

### **XSS Prevention**
```tsx
// Sanitize HTML content
import DOMPurify from 'dompurify'

function SafeHTML({ content }: { content: string }) {
  const sanitizedContent = DOMPurify.sanitize(content)
  
  return (
    <div dangerouslySetInnerHTML={{ __html: sanitizedContent }} />
  )
}
```

### **CSRF Protection**
```tsx
// Include CSRF tokens in API calls
const api = axios.create({
  baseURL: '/api',
  headers: {
    'X-CSRF-Token': getCsrfToken(),
  },
})
```

## 🚀 **Deployment**

### **Build Process**
```bash
# Development build
npm run build -- --mode development

# Production build
npm run build -- --mode production

# Analyze bundle
npm run analyze
```

### **Docker Deployment**
```bash
# Build Docker image
docker build -t qsfl-caad-frontend .

# Run container
docker run -p 80:80 qsfl-caad-frontend

# With docker-compose
docker-compose up frontend
```

### **Environment Configuration**
```bash
# .env.local
VITE_API_URL=http://localhost:5000
VITE_WS_URL=ws://localhost:5000
VITE_DEBUG=false

# .env.production
VITE_API_URL=https://api.qsfl-caad.com
VITE_WS_URL=wss://api.qsfl-caad.com
VITE_DEBUG=false
```

## 📚 **Resources & Learning**

### **Documentation**
- [React Documentation](https://react.dev/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Tailwind CSS Docs](https://tailwindcss.com/docs)
- [Framer Motion Guide](https://www.framer.com/motion/)
- [React Query Docs](https://tanstack.com/query/latest)

### **Tools & Extensions**
- **VS Code Extensions**:
  - ES7+ React/Redux/React-Native snippets
  - Tailwind CSS IntelliSense
  - TypeScript Importer
  - Auto Rename Tag
  - Bracket Pair Colorizer

### **Best Practices**
- Follow React best practices and patterns
- Use TypeScript for type safety
- Write comprehensive tests
- Optimize for performance
- Ensure accessibility compliance
- Follow semantic HTML structure

---

This frontend provides a modern, scalable, and maintainable foundation for your QSFL-CAAD project with professional-grade development practices! 🎉